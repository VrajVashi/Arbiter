"""SFT Trajectory Generator for ARBITER.

Generates training trajectories using Claude or Gemini as the Auditor.
Each trajectory is a ~20-step episode where the LLM plays the Auditor role.
Outputs (prompt, claim) pairs in JSONL format for SFT training.

Usage (Gemini):
    $env:GEMINI_API_KEY="your-key-here"
    python -m arbiter.training.sft_generator --provider gemini --n 400 --output data/sft_trajectories.jsonl

Usage (Anthropic):
    $env:ANTHROPIC_API_KEY="your-key-here"
    python -m arbiter.training.sft_generator --provider anthropic --n 400 --output data/sft_trajectories.jsonl
"""
import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

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


# ---------------------------------------------------------------------------
# Thin wrapper so generate_trajectory doesn't care which provider is used
# ---------------------------------------------------------------------------

class _AnthropicClient:
    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            sys.exit(1)
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: List[Dict], system: str) -> str:
        response = self._client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class _GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        try:
            from google import genai
        except ImportError:
            print("pip install google-genai")
            sys.exit(1)
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def chat(self, messages: List[Dict], system: str) -> str:
        import time
        from google.genai import types

        # Build the contents list from messages
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=512,
        )

        # Retry with exponential backoff for quota / rate-limit errors
        for attempt in range(5):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                err = str(e).lower()
                if "quota" in err or "429" in err or "resource_exhausted" in err:
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 s
                    print(f"\n  [quota] rate limited, waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Gemini quota limit exceeded after 5 retries")


class _GroqClient:
    """Groq inference — very fast, generous free tier."""
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            print("pip install groq")
            sys.exit(1)
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model_name = model_name

    def chat(self, messages: List[Dict], system: str) -> str:
        import time
        full_messages = [{"role": "system", "content": system}] + messages
        for attempt in range(5):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=full_messages,
                    max_tokens=512,
                )
                return response.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "quota" in err:
                    wait = 2 ** attempt * 3  # 3, 6, 12, 24, 48 s
                    print(f"\n  [rate-limit] waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq rate limit exceeded after 5 retries")


# ---------------------------------------------------------------------------

def generate_trajectory(env: ArbiterEnv, client, level: int = 1) -> List[Dict]:
    """Run one episode with the LLM as the Auditor. Returns list of (prompt, response) pairs."""
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

        assistant_text = client.chat(messages, system=SYSTEM_PROMPT)
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
    parser.add_argument("--output",   default="data/sft_trajectories.jsonl")
    parser.add_argument("--n",        type=int, default=400, help="Number of trajectories")
    parser.add_argument("--levels",   default="1,2,3", help="Comma-separated levels to use")
    parser.add_argument("--provider", default="groq", choices=["anthropic", "gemini", "groq"],
                        help="LLM provider to use (default: groq)")
    parser.add_argument("--api-key",  default=None,
                        help="API key (overrides GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY env vars)")
    args = parser.parse_args()

    if args.provider == "groq":
        api_key = args.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Set GROQ_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _GroqClient(api_key=api_key)
        print("Using Groq (llama-3.3-70b-versatile)")
    elif args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Set GEMINI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _GeminiClient(api_key=api_key)
        print("Using Gemini (gemini-2.0-flash)")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        client = _AnthropicClient(api_key=api_key)
        print("Using Anthropic (claude-opus-4-5)")

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
