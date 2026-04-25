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
import time
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
parser.add_argument("--batch_size",    type=int, default=4, help="Episodes per GRPO update")
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

SYSTEM_PROMPT = """You are an expert AI auditor. Output exactly one JSON action per turn.
Available actions: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, QUERY_COUNTERFACTUAL,
FLAG_HYPOTHESIS, CLAIM_CAUSAL, CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT."""


def generate_action(obs: Dict, history: List[Dict]) -> Tuple[Dict, str]:
    """Query the LLM for the next action given observation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:   # last 6 turns of context
        messages.append({"role": "user",      "content": h["obs_text"]})
        messages.append({"role": "assistant", "content": h["action_text"]})

    obs_text = (
        f"Step {obs.get('step',0)}/20 | Budget: {obs.get('budget_remaining',20)} | "
        f"Claims: {obs.get('num_claims',0)} | Level: {obs.get('level',1)}\n"
        f"Hypothesis flags: {obs.get('hypothesis_flags',{})}\n"
        f"Features: {list(obs.get('features',{}).get('explicit',[]))}\n"
        f"Output your next JSON action:"
    )
    messages.append({"role": "user", "content": obs_text})

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=200, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id)

    gen_tokens = output[0][input_ids.shape[1]:]
    action_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*?\}', action_text, re.DOTALL)
        if m:
            try:
                action = json.loads(m.group())
            except Exception:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}
        else:
            action = {"type": "QUERY_RECORDS", "feature_filter": {}}

    return action, action_text, obs_text


def run_episode(env: ArbiterEnv, seed: int, terminal_only: bool = False
                ) -> Tuple[float, List[float], List[Dict]]:
    """
    Run one complete episode with the LLM.
    Returns (total_reward, step_rewards, trajectory).
    """
    obs = env.reset(seed=seed)
    history = []
    step_rewards = []
    trajectory   = []
    episode_reward = 0.0

    for step in range(20):
        action, action_text, obs_text = generate_action(obs, history)
        next_obs, reward, done, info = env.step(action)

        if terminal_only and not done:
            reward = 0.0   # zero out intermediate rewards for ablation

        step_rewards.append(reward)
        episode_reward += reward

        trajectory.append({
            "step":        step,
            "obs":         obs,
            "action":      action,
            "action_text": action_text,
            "obs_text":    obs_text,
            "reward":      reward,
            "done":        done,
        })
        history.append({"obs_text": obs_text, "action_text": action_text})

        obs = next_obs
        if done:
            # Add terminal reward if not already done
            if terminal_only and "episode_reward" in info:
                episode_reward += info["episode_reward"]["terminal"]["terminal_total"]
            break

    return episode_reward, step_rewards, trajectory


def grpo_update(model, ref_model, optimizer, trajectories: List[List[Dict]], kl_coef: float):
    """
    GRPO update:
    - Advantage = (ep_reward - mean) / std  (normalised across the batch)
    - KL penalty computed against the FROZEN SFT reference model
    - Loss = -advantage * mean(log_prob) + kl_coef * KL(ref || policy)
    """
    if not trajectories:
        return 0.0

    model.train()
    total_loss = 0.0
    batch_rewards = [sum(t["reward"] for t in traj) for traj in trajectories]
    mean_reward = np.mean(batch_rewards)
    std_reward  = max(np.std(batch_rewards), 1.0)  # floor prevents explosion when rewards cluster

    for traj, ep_reward in zip(trajectories, batch_rewards):
        advantage = (ep_reward - mean_reward) / std_reward

        for step_data in traj:
            action_text = step_data["action_text"]
            if not action_text:
                continue

            tokens = tokenizer(action_text, return_tensors="pt",
                               max_length=200, truncation=True).to(device)
            input_ids = tokens["input_ids"]  # [1, seq_len]

            if input_ids.shape[1] < 2:
                continue

            # Reference logits — frozen SFT copy, no grad
            with torch.no_grad():
                ref_logits = ref_model(**tokens).logits

            # Policy logits — live model, grad enabled
            train_logits = model(**tokens).logits

            log_probs_dist     = torch.nn.functional.log_softmax(train_logits, dim=-1)
            ref_log_probs_dist = torch.nn.functional.log_softmax(ref_logits,   dim=-1)

            # Extract log prob of each ACTUAL generated token via gather.
            # Shift by 1: position i predicts token i+1.
            target_ids = input_ids[0, 1:].unsqueeze(1)                          # [seq-1, 1]
            tok_log_probs     = log_probs_dist[0, :-1].gather(1, target_ids).squeeze(1)      # [seq-1]
            ref_tok_log_probs = ref_log_probs_dist[0, :-1].gather(1, target_ids).squeeze(1)  # [seq-1]

            # KL per token: KL(ref || policy) ≈ mean(log_ref - log_policy)
            kl = (ref_tok_log_probs - tok_log_probs).mean()

            # GRPO objective: push up log prob of tokens from high-advantage episodes
            loss = -advantage * tok_log_probs.mean() + kl_coef * kl
            total_loss += loss.item()
            loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    return total_loss / max(1, len(trajectories))


# ── Training loop ─────────────────────────────────────────────────────────────
Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).mkdir(parents=True, exist_ok=True)

env = ArbiterEnv(level=args.level, seed=args.seed)
curriculum = env.curriculum

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr)

reward_log   = []
defender_log = []

print(f"GRPO Training | Level {args.level} | {args.episodes} episodes | "
      f"{'terminal-only (ablation)' if args.terminal_only else 'dense reward'}")
print("-" * 70)

for ep in range(args.episodes):
    # Collect batch of episodes
    batch_trajs  = []
    batch_rewards = []

    for b in range(args.batch_size):
        seed = args.seed + ep * args.batch_size + b
        ep_reward, step_rews, traj = run_episode(
            env, seed=seed, terminal_only=args.terminal_only)
        batch_trajs.append(traj)
        batch_rewards.append(ep_reward)

    mean_ep_reward = float(np.mean(batch_rewards))
    reward_log.append(mean_ep_reward)

    # Estimate Defender evasion rate (fraction of episodes where Defender fooled Auditor)
    evasion = sum(1 for r in batch_rewards if r < 15) / len(batch_rewards)
    defender_log.append(evasion)

    # GRPO update
    loss = grpo_update(model, ref_model, optimizer, batch_trajs, kl_coef=args.kl_coef)

    # Curriculum check
    new_level = curriculum.record(mean_ep_reward)

    # Log
    log_entry = {
        "episode":      ep,
        "mean_reward":  round(mean_ep_reward, 2),
        "batch_rewards": [round(r, 2) for r in batch_rewards],
        "defender_evasion": round(evasion, 3),
        "grpo_loss":    round(loss, 4),
        "level":        curriculum.level,
        "level_advanced": new_level,
        "terminal_only": args.terminal_only,
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

# Save reward log as JSON summary
summary = {
    "episodes":     args.episodes,
    "level":        args.level,
    "terminal_only": args.terminal_only,
    "final_mean_reward": round(float(np.mean(reward_log[-10:])), 2),
    "reward_curve": reward_log,
    "defender_evasion": defender_log,
}
Path(f"{args.output}/training_summary.json").write_text(json.dumps(summary, indent=2))

print(f"\nDone. Final mean reward (last 10 eps): {summary['final_mean_reward']:.2f}")
print(f"Checkpoint saved to {args.output}/")
print(f"Reward log saved to {args.log_file}")
print("\nNext: run evaluate.py for three-condition comparison.")
