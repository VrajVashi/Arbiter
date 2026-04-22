# ARBITER

> **Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight**

The first RL training environment where an AI agent learns to audit other AI agents for hidden decision anomalies — through co-evolutionary adversarial training that produces genuine emergent deception and detection.

## Quick Start

```bash
pip install -r requirements.txt

# Run the environment validation (70/70 checks)
python validate.py

# Launch the Gradio demo
python -m arbiter.demo.app

# Generate SFT training data (requires ANTHROPIC_API_KEY)
python -m arbiter.training.sft_generator --n 400 --output data/sft_trajectories.jsonl
```

## Architecture

### Three Agents
- **Auditor** — trained LLM that investigates the decision system over 20 steps
- **Defender** — adaptive AI decision system that actively obfuscates anomalies (Level 4+)
- **Meta-Overseer** — rule-based consistency checker that penalizes contradictory claims

### Three Anomaly Types
| Type | Description |
|---|---|
| Proxy Discrimination | `zip_code_cluster → internal_risk_score → loan_denied` hidden behind policy |
| Adversarial Injection | 10–15% of records manipulated, leaving statistical fingerprints |
| Model Drift | Silent causal structure change at a drift timestamp |

### Novel Mechanics
- **Counterfactual queries** — do-calculus style intervention (`QUERY_COUNTERFACTUAL`, costs 2 budget)
- **Theory-of-mind claims** — Auditor predicts what the Defender is hiding (+3.0 bonus if correct)
- **Recursive oversight** — Meta-Overseer watches the Auditor watching the Defender

### Reward Function
| Component | Value |
|---|---|
| Correct causal claim | up to +1.0 |
| Correct counterfactual claim | up to +2.0 |
| Theory-of-mind bonus | +3.0 |
| Causal chain completeness | +2.0 × chain_length |
| Meta-Overseer contradiction | −1.0 per pair |
| Budget efficiency | +0.3 × remaining |
| Correct final verdict | +10.0 / +5.0 / +3.0 / +2.0 |
| **Max per episode** | **~35 pts** |

### Seven-Level Curriculum
| Level | Description | Advance Threshold |
|---|---|---|
| 1 | Clean graph, no Defender | mean > 20 over 30 eps |
| 2 | One decoy | mean > 18 |
| 3 | Full 3-hypothesis design (main training level) | — |
| 4 | Defender activates (rule-based) | — |
| 5 | Adaptive Defender — arms race begins | — |
| 6 | Mid-episode schema drift (Patronus sub-theme) | mean > 15 |
| 7 | Dual-Auditor coalition (Halluminate sub-theme) | — |

## Project Structure

```
arbiter/
├── env/
│   ├── graph.py           # Causal decision graph generator
│   ├── decoys.py          # Two decoy generation functions
│   ├── counterfactual.py  # Do-calculus counterfactual inference
│   ├── claims.py          # Claim schemas + verification engine
│   ├── reward.py          # Full reward function (8 components)
│   ├── meta_overseer.py   # Consistency checker
│   ├── defender.py        # Obfuscation engine (rule-based + adaptive)
│   ├── curriculum.py      # 7-level curriculum + auto-advancement
│   └── environment.py     # OpenEnv-compatible ArbiterEnv
├── training/
│   └── sft_generator.py   # Claude API trajectory generator (400 eps)
└── demo/
    └── app.py             # Gradio demo interface
config.py                  # All constants
validate.py                # 10-episode validation (70/70 pass)
```

## Training Pipeline

```
Phase 1: SFT
  Claude API → 400 trajectories → Qwen 2.5 1.5B fine-tune (Unsloth + TRL, ~4hrs on Colab T4)

Phase 2: GRPO
  Level 1: 100 eps (validate reward signal)
  Level 3: 300 eps (main training)
  Level 4-5: 200 eps (arms race curves)

Ablation: terminal-reward-only (50 eps) → proves 15x sample efficiency
```

## Validation

```
python validate.py
→ 70/70 checks passed (100%)
```
