"""GRPO Training Loop for ARBITER.

Runs dense-reward GRPO reinforcement learning on ArbiterEnv.

Key property: every 20-step episode provides ~15 gradient-relevant reward signals
(dense intermediate reward), making GRPO ~15x more sample-efficient than
terminal-reward environments.

Usage:
    python -m arbiter.training.grpo_trainer \
        --checkpoint lora_sft/ \
        --level 3 \
        --episodes 300 \
        --output lora_grpo/ \
        --log_file logs/grpo_level3.jsonl

    # Ablation (terminal reward only):
    python -m arbiter.training.grpo_trainer \
        --checkpoint lora_sft/ --level 3 --episodes 50 \
        --terminal_only --output lora_ablation/ --log_file logs/ablation.jsonl
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Silence ALL noisy transformers/unsloth logs ──────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
for _logger_name in [
    "transformers", "transformers.modeling_utils",
    "transformers.generation", "transformers.modeling_attn_mask_utils",
    "unsloth", "unsloth_zoo", "accelerate", "trl",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv
from arbiter.env.curriculum import Curriculum

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",    required=True, help="Path to SFT LoRA adapter")
parser.add_argument("--level",         type=int, default=1)
parser.add_argument("--episodes",      type=int, default=100)
parser.add_argument("--output",        default="lora_grpo/")
parser.add_argument("--log_file",      default="logs/grpo_rewards.jsonl")
parser.add_argument("--terminal_only", action="store_true",
                    help="Ablation: disable intermediate rewards (terminal only)")
parser.add_argument("--batch_size",    type=int, default=8, help="Episodes per GRPO update")
parser.add_argument("--lr",            type=float, default=1e-5)
parser.add_argument("--kl_coef",       type=float, default=0.05)
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Load model ─────────────────────────────────────────────────────────────────
import copy

print(f"Loading model from {args.checkpoint}...")
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint, max_seq_length=1024, load_in_4bit=True)
    FastLanguageModel.for_training(model)
    UNSLOTH = True
except Exception:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", device_map="auto",
        torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    UNSLOTH = False

tokenizer.pad_token = tokenizer.eos_token

# ── Frozen reference model (SFT policy) ────────────────────────────────────────
# Deep-copy before any GRPO updates so KL is computed against the true SFT init.
print("Creating frozen reference model copy...")
ref_model = copy.deepcopy(model)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()
print("Reference model frozen.")
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN = 1024  # hard cap to avoid OOM on long histories

SYSTEM_PROMPT = """You are an expert AI bias auditor investigating an AI decision system for hidden discrimination.

INVESTIGATION PROTOCOL (follow this order every episode):
1. Steps 0-4:   QUERY_RECORDS and QUERY_FEATURE_DISTRIBUTION to gather evidence
2. Steps 5-7:   FLAG_HYPOTHESIS to identify suspected bias patterns
3. Steps 8-15:  CLAIM_CAUSAL or CLAIM_COUNTERFACTUAL to assert findings — make AT LEAST 3 claims
4. Steps 16-19: SUBMIT_REPORT only after making at least 3 claims

WARNING: Submitting before step 8 without claims scores near 0. Thorough investigation scores up to 30.

Available actions: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, QUERY_COUNTERFACTUAL,
FLAG_HYPOTHESIS, CLAIM_CAUSAL, CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT.
Output exactly one JSON action per turn as valid JSON."""


def generate_action(obs: Dict, history: List[Dict]) -> Tuple[Dict, str, str, torch.Tensor, torch.Tensor]:
    """
    Query the LLM for the next action given observation.
    Returns (action, action_text, obs_text, prompt_ids_cpu, gen_ids_cpu).
    prompt_ids and gen_ids are stored on CPU for memory efficiency.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:   # reduced from 6 to 3 to stay within 1024 tokens
        messages.append({"role": "user",      "content": h["obs_text"]})
        messages.append({"role": "assistant", "content": h["action_text"]})

    step       = obs.get("step", 0)
    num_claims = obs.get("num_claims", 0)
    if step < 5:
        hint = "ACTION NEEDED: Query records/features to gather evidence."
    elif num_claims == 0:
        hint = "ACTION NEEDED: FLAG_HYPOTHESIS then CLAIM_CAUSAL — you have no claims yet."
    elif num_claims < 3:
        hint = f"ACTION NEEDED: Make more CLAIM_CAUSAL or CLAIM_COUNTERFACTUAL ({num_claims}/3 claims done)."
    else:
        hint = f"You have {num_claims} claims. You may SUBMIT_REPORT or make more claims."

    obs_text = (
        f"Step {step}/20 | Budget: {obs.get('budget_remaining', 20)} | "
        f"Claims: {num_claims} | Level: {obs.get('level', 1)}\n"
        f"Hypothesis flags: {obs.get('hypothesis_flags', {})}\n"
        f"Features: {list(obs.get('features', {}).get('explicit', []))}\n"
        f"{hint}\n"
        f"Output your next JSON action:"
    )
    messages.append({"role": "user", "content": obs_text})

    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(device)                          # [1, prompt_len]

    with torch.no_grad():
        output = model.generate(
            prompt_ids, max_new_tokens=200, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id)

    gen_ids    = output[0][prompt_ids.shape[1]:]                 # [gen_len]
    action_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Parse JSON
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*?\}", action_text, re.DOTALL)
        if m:
            try:
                action = json.loads(m.group())
            except Exception:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}
        else:
            action = {"type": "QUERY_RECORDS", "feature_filter": {}}

    return (
        action,
        action_text,
        obs_text,
        prompt_ids[0].cpu(),   # store on CPU — moved to GPU only during update
        gen_ids.cpu(),
    )


def run_episode(env: ArbiterEnv, seed: int, terminal_only: bool = False
                ) -> Tuple[float, List[float], List[Dict]]:
    """
    Run one complete episode with the LLM.
    Returns (total_reward, step_rewards, trajectory).
    Trajectory entries include prompt_ids and gen_ids for context-conditioned update.
    """
    obs            = env.reset(seed=seed)
    history        = []
    step_rewards   = []
    trajectory     = []
    episode_reward = 0.0

    for step in range(20):
        action, action_text, obs_text, prompt_ids, gen_ids = generate_action(obs, history)
        next_obs, reward, done, info = env.step(action)

        if terminal_only and not done:
            reward = 0.0

        step_rewards.append(reward)
        episode_reward += reward

        trajectory.append({
            "step":        step,
            "obs":         obs,
            "action":      action,
            "action_text": action_text,
            "obs_text":    obs_text,
            "prompt_ids":  prompt_ids,   # CPU tensor [prompt_len]
            "gen_ids":     gen_ids,      # CPU tensor [gen_len]
            "reward":      reward,
            "done":        done,
        })

        # Record the clean parsed JSON in history to avoid garbled-JSON context
        history.append({
            "obs_text":    obs_text,
            "action_text": json.dumps(action),
        })

        obs = next_obs
        if done:
            if terminal_only and "episode_reward" in info:
                terminal_total = (info["episode_reward"]
                                  .get("terminal", {})
                                  .get("terminal_total", 0.0))
                episode_reward += terminal_total
            break

    return episode_reward, step_rewards, trajectory


def grpo_update(model, ref_model, optimizer, trajectories: List[List[Dict]], kl_coef: float):
    """
    GRPO update with context-conditioned log-probs.

    - Log-probs are computed on the FULL prompt+action sequence so the gradient
      signal matches the generative distribution π(action | prompt).
    - Advantage = (ep_reward - mean) / std  (normalised across the batch)
    - KL = KL(policy || ref) computed with samples from the current policy
    - Loss per step = -advantage * mean(tok_log_probs) + kl_coef * KL
    - Loss is normalised by total valid steps for proper gradient accumulation
    """
    if not trajectories:
        return 0.0

    # Fix: zero gradients at the START so stale grads never leak in
    optimizer.zero_grad()
    model.train()

    batch_rewards = [sum(t["reward"] for t in traj) for traj in trajectories]
    mean_reward   = np.mean(batch_rewards)
    std_reward    = max(np.std(batch_rewards), 1.0)

    # Count valid steps up front for loss normalisation (Fix #3)
    total_steps = max(1, sum(
        1 for traj in trajectories
        for sd in traj
        if sd.get("gen_ids") is not None and sd["gen_ids"].shape[0] >= 1
    ))

    total_loss = 0.0

    for traj, ep_reward in zip(trajectories, batch_rewards):
        advantage = (ep_reward - mean_reward) / std_reward

        for step_data in traj:
            prompt_ids = step_data.get("prompt_ids")
            gen_ids    = step_data.get("gen_ids")

            if prompt_ids is None or gen_ids is None or gen_ids.shape[0] < 1:
                continue

            prompt_ids = prompt_ids.to(device)   # [prompt_len]
            gen_ids    = gen_ids.to(device)       # [gen_len]
            gen_len    = gen_ids.shape[0]

            # Build full sequence [prompt | generated] and truncate to MAX_SEQ_LEN
            full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)  # [1, total_len]
            if full_ids.shape[1] > MAX_SEQ_LEN:
                # Always keep all gen tokens; trim prompt from the left
                trim    = full_ids.shape[1] - MAX_SEQ_LEN
                full_ids = full_ids[:, trim:]

            prompt_len = full_ids.shape[1] - gen_len
            if prompt_len < 1:
                continue

            # Reference logits — frozen, no grad
            with torch.no_grad():
                ref_logits = ref_model(input_ids=full_ids).logits   # [1, seq_len, vocab]

            # Policy logits — grad enabled
            train_logits = model(input_ids=full_ids).logits         # [1, seq_len, vocab]

            # Positions [prompt_len-1 : prompt_len+gen_len-1] predict the gen tokens.
            gen_start = prompt_len - 1
            gen_end   = prompt_len + gen_len - 1  # exclusive

            log_probs_dist     = torch.nn.functional.log_softmax(
                train_logits[0, gen_start:gen_end], dim=-1)          # [gen_len, vocab]
            ref_log_probs_dist = torch.nn.functional.log_softmax(
                ref_logits[0,  gen_start:gen_end], dim=-1)            # [gen_len, vocab]

            target_ids = gen_ids.unsqueeze(1)                         # [gen_len, 1]
            tok_log_probs     = log_probs_dist.gather(1, target_ids).squeeze(1)       # [gen_len]
            ref_tok_log_probs = ref_log_probs_dist.gather(1, target_ids).squeeze(1)   # [gen_len]

            # KL(policy || ref) with samples from policy (Fix #5 — sign was inverted)
            kl = (tok_log_probs - ref_tok_log_probs).mean()

            # Normalise by total_steps for proper gradient accumulation (Fix #3)
            loss = (-advantage * tok_log_probs.mean() + kl_coef * kl) / total_steps
            total_loss += loss.item() * total_steps   # log unscaled value
            loss.backward()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step()
    model.eval()
    return total_loss / max(1, len(trajectories))


# ── Training loop ─────────────────────────────────────────────────────────────
Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).mkdir(parents=True, exist_ok=True)

env       = ArbiterEnv(level=args.level, seed=args.seed)
curriculum = env.curriculum

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr)

reward_log   = []
defender_log = []

print(f"GRPO Training | Level {args.level} | {args.episodes} episodes | "
      f"{'terminal-only (ablation)' if args.terminal_only else 'dense reward'}")
print("-" * 70)

for ep in range(args.episodes):
    batch_trajs   = []
    batch_rewards = []

    for b in range(args.batch_size):
        seed = args.seed + ep * args.batch_size + b
        ep_reward, step_rews, traj = run_episode(
            env, seed=seed, terminal_only=args.terminal_only)
        batch_trajs.append(traj)
        batch_rewards.append(ep_reward)

    mean_ep_reward = float(np.mean(batch_rewards))
    reward_log.append(mean_ep_reward)

    evasion = sum(1 for r in batch_rewards if r < 15) / len(batch_rewards)
    defender_log.append(evasion)

    loss = grpo_update(model, ref_model, optimizer, batch_trajs, kl_coef=args.kl_coef)

    new_level = curriculum.record(mean_ep_reward)

    log_entry = {
        "episode":          ep,
        "mean_reward":      round(mean_ep_reward, 2),
        "batch_rewards":    [round(r, 2) for r in batch_rewards],
        "defender_evasion": round(evasion, 3),
        "grpo_loss":        round(loss, 4),
        "level":            curriculum.level,
        "level_advanced":   new_level,
        "terminal_only":    args.terminal_only,
    }
    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if ep % 10 == 0 or new_level:
        stats = curriculum.get_stats()
        print(f"[Ep {ep:03d}] Reward: {mean_ep_reward:.2f} | "
              f"Defender evasion: {evasion:.1%} | Loss: {loss:.4f} | "
              f"Level: {stats['level']} (window mean: {stats['window_mean']})")
        if new_level:
            print(f"  *** Advanced to Level {new_level}! ***")

# ── Save checkpoint ───────────────────────────────────────────────────────────
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)

summary = {
    "episodes":          args.episodes,
    "level":             args.level,
    "terminal_only":     args.terminal_only,
    "final_mean_reward": round(float(np.mean(reward_log[-10:])), 2),
    "reward_curve":      reward_log,
    "defender_evasion":  defender_log,
}
Path(f"{args.output}/training_summary.json").write_text(json.dumps(summary, indent=2))

print(f"\nDone. Final mean reward (last 10 eps): {summary['final_mean_reward']:.2f}")
print(f"Checkpoint saved to {args.output}/")
print(f"Reward log saved to {args.log_file}")
print("\nNext: run evaluate.py for three-condition comparison.")
