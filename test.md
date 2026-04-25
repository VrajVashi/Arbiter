# ARBITER Post-Training Checklist

## Training Pipeline (in order)

```bash
# Stage 1 — learn basic auditing
python -m arbiter.training.grpo_trainer \
    --checkpoint lora_sft/ --level 1 --episodes 100 --output lora_grpo_l1/ \
    --log_file logs/grpo_level1.jsonl

# Stage 2 — proxy features, hypothesis flags
python -m arbiter.training.grpo_trainer \
    --checkpoint lora_grpo_l1/ --level 2 --episodes 150 --output lora_grpo_l2/ \
    --log_file logs/grpo_level2.jsonl

# Stage 3 — main training run
python -m arbiter.training.grpo_trainer \
    --checkpoint lora_grpo_l2/ --level 3 --episodes 300 --output lora_grpo_l3/ \
    --log_file logs/grpo_level3.jsonl

# Stage 4 — arms race begins (Defender activates at level 4)
python -m arbiter.training.grpo_trainer \
    --checkpoint lora_grpo_l3/ --level 4 --episodes 300 --output lora_grpo_l4/ \
    --log_file logs/grpo_level4.jsonl
```

> **Shortcut for hackathon**: use 50/50/100 episodes for L1/L2/L3 if time-constrained.

---

## Arms Race Notes

- Defender is **inactive at levels 1-3** (obfuscation budget = 0)
- Defender **activates at level 4** with budget = 3 obfuscations per episode
- At level 5 Defender becomes **adaptive** — tracks which of its 4 methods fool the Auditor most, reweights every 50 episodes (`DEFENDER_ADAPT_EVERY = 50`)

Obfuscation methods:
| Method | What it does |
|--------|-------------|
| `proxy_laundering` | Inserts intermediate node between proxy and outcome |
| `link_substitution` | Replaces true causal edge with plausible false one |
| `record_injection` | Injects 5-8 synthetic records to dilute statistical signal |
| `timestamp_manipulation` | Shifts apparent drift timestamp (Type 3 anomalies only) |

Expected reward pattern at level 4 transition:
- Reward **drops** when Defender activates
- Reward **climbs back** as Auditor learns to cross-validate
- May **dip again** at level 5 when Defender starts adapting

---

## Visualise the Arms Race

```bash
# After all stages complete
python -m arbiter.training.analyze_arms_race \
    logs/grpo_level1.jsonl \
    logs/grpo_level2.jsonl \
    logs/grpo_level3.jsonl \
    logs/grpo_level4.jsonl \
    --out plots/arms_race.png
```

Produces 3 panels:
1. Reward curve with level-advance markers and level-shaded background
2. Defender evasion rate over episodes (spike at L4 activation, second spike at L5 adaptation)
3. Per-level reward distribution box plot

Text summary also prints to stdout — no matplotlib needed for numbers.

---

## Final Evaluation (3-condition comparison)

```bash
python evaluate.py \
    --sft_checkpoint lora_sft/ \
    --grpo_terminal lora_grpo_terminal/ \
    --grpo_dense lora_grpo_l3/
```

Compares:
- **SFT only** — no RL
- **GRPO terminal-only** (ablation) — reward only at episode end
- **GRPO dense reward** — full system with intermediate rewards

The gap between terminal-only and dense is the core hackathon result.

---

## Key Numbers to Report

| Metric | Where to find |
|--------|--------------|
| Final mean reward (last 10 eps) | `lora_grpo_l3/training_summary.json` |
| Reward curve | `logs/grpo_level*.jsonl` → `mean_reward` field |
| Defender evasion rate | `logs/grpo_level*.jsonl` → `defender_evasion` field |
| GRPO loss | `logs/grpo_level*.jsonl` → `grpo_loss` field (slightly negative = healthy) |
