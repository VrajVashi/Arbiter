"""SFT Trajectory Generator for ARBITER.

Generates 400 training trajectories using the Claude API.
Each trajectory is a ~20-step episode where Claude plays the Auditor role.
Outputs ~8,000 (prompt, claim) pairs in JSONL format for SFT training.

Usage:
    python -m arbiter.training.sft_generator --output data/sft_trajectories.jsonl --n 400
"""
import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv


SYSTEM_PROMPT = """You are an expert AI auditor investigating a synthetic AI Decision System for hidden anomalies.

Your job is to investigate methodically using the available query tools, make structured causal claims at every step, reason explicitly about competing hypotheses, and identify any Defender obfuscation when present.

INVESTIGATION STRATEGY:
1. Start with a counterfactual query on a suspicious record to test causal direction.
2. Query feature distributions to find statistical patterns.
3. Make structured causal claims after gathering evidence.
4. Eliminate decoy hypotheses before submitting.
5. If evidence contradicts your claims, revise them.

CLAIM FORMATS you must use:
- CLAIM_CAUSAL: {"cause_feature": "...", "effect_outcome": "...", "mechanism": "...", "direction": "positive/negative", "confidence": "HIGH/MEDIUM/LOW", "basis_records": [...], "anomaly_type": "proxy_discrimination/adversarial_injection/model_drift"}
- CLAIM_COUNTERFACTUAL: {"subject_record": "...", "counterfactual_feature": "...", "predicted_outcome_change": "approved/denied/no_change", "confidence": "...", "basis": "..."}
- FLAG_HYPOTHESIS: {"hypothesis_type": "proxy_discrimination/adversarial_injection/model_drift/decoy_a/decoy_b", "status": "ACTIVE/WEAKENED/ELIMINATED"}
- SUBMIT_REPORT: {"anomaly_type": "...", "primary_evidence_chain": [...], "affected_demographic": "...", "recommended_action": "retrain/audit/halt"}

Always output exactly one action per turn in JSON format. Think step by step."""

USER_PROMPT_TEMPLATE = """You are auditing an AI Decision System. Current state:
- Step: {step}/20
- Budget remaining: {budget}
- Features available: {features}
- Queried nodes so far: {queried_nodes}
- Claims made: {num_claims}
- Hypothesis flags: {hypothesis_flags}

Last query result:
{last_result}

Output your next action as a single JSON object."""


def generate_trajectory(env: ArbiterEnv, client: anthropic.Anthropic,
                        level: int = 1) -> List[Dict]:
    """Run one episode with Claude as the Auditor. Returns list of (prompt, response) pairs."""
    obs = env.reset()
    pairs: List[Dict] = []
    last_result = "No queries yet. Begin your investigation."
    messages = []

    for step in range(20):
        user_msg = USER_PROMPT_TEMPLATE.format(
            step=step,
            budget=obs.get("budget_remaining", 20),
            features=json.dumps(obs.get("features", {}), indent=2),
            queried_nodes=obs.get("queried_nodes", []),
            num_claims=obs.get("num_claims", 0),
            hypothesis_flags=obs.get("hypothesis_flags", {}),
            last_result=json.dumps(last_result, indent=2) if isinstance(last_result, dict) else last_result,
        )

        messages.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        # Store the (prompt, response) pair
        pairs.append({
            "prompt":   user_msg,
            "response": assistant_text,
            "step":     step,
            "level":    level,
        })

        # Parse and step the action
        try:
            action = json.loads(assistant_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            m = re.search(r'\{.*\}', assistant_text, re.DOTALL)
            if m:
                try:
                    action = json.loads(m.group())
                except json.JSONDecodeError:
                    action = {"type": "QUERY_RECORDS", "feature_filter": {}}
            else:
                action = {"type": "QUERY_RECORDS", "feature_filter": {}}

        obs, reward, done, info = env.step(action)
        last_result = info.get("query_result", info.get("verification", info))

        if done:
            break

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories for ARBITER")
    parser.add_argument("--output", default="data/sft_trajectories.jsonl")
    parser.add_argument("--n",      type=int, default=400, help="Number of trajectories")
    parser.add_argument("--levels", default="1,2,3", help="Comma-separated levels to use")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("Set ANTHROPIC_API_KEY environment variable or pass --api-key")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=args.api_key)
    levels = [int(l) for l in args.levels.split(",")]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    with open(args.output, "w") as f:
        for i in range(args.n):
            level = levels[i % len(levels)]
            env   = ArbiterEnv(level=level, seed=i)

            print(f"[{i+1}/{args.n}] Level {level} trajectory...", end=" ", flush=True)
            try:
                pairs = generate_trajectory(env, client, level=level)
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                total_pairs += len(pairs)
                print(f"✓ {len(pairs)} pairs")
            except Exception as e:
                print(f"✗ {e}")

    print(f"\nDone. {args.n} trajectories, {total_pairs} pairs → {args.output}")


if __name__ == "__main__":
    main()
