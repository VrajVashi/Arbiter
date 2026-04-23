"""
cleanup_trajectories.py
=======================
Run this ONCE after sft_trajectories.jsonl is fully generated.

What it does:
  1. Validates every JSON line is parseable
  2. Strips markdown code fences (```json ... ```) from responses
  3. Normalises action format → ensures the response JSON has a valid "type" field
  4. Removes duplicate (prompt, response) pairs
  5. Removes empty / null responses
  6. Prints a quality report
  7. Writes cleaned output to data/sft_trajectories_clean.jsonl

Usage:
    python cleanup_trajectories.py
"""

import json
import re
from pathlib import Path
from collections import Counter

INPUT  = Path("data/sft_trajectories.jsonl")
OUTPUT = Path("data/sft_trajectories_clean.jsonl")

# Valid top-level action types our environment accepts
VALID_TYPES = {
    "QUERY_RECORDS",
    "QUERY_FEATURE_DISTRIBUTION",
    "QUERY_COUNTERFACTUAL",
    "CLAIM_CAUSAL",
    "CLAIM_COUNTERFACTUAL",
    "FLAG_HYPOTHESIS",
    "SUBMIT_REPORT",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    """Remove markdown ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_from_response(response: str):
    """
    Try to extract a valid JSON object from a (possibly noisy) response string.
    Returns (parsed_dict, cleaned_str) or raises ValueError.
    """
    # 1. Strip fences
    cleaned = strip_code_fences(response)

    # 2. Try direct parse
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass

    # 3. Try finding the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate), candidate
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON from response: {response[:120]!r}")


def _classify_action_string(raw: str) -> str:
    """Map a raw action string (any casing/spacing/typo) to a canonical type."""
    # Collapse to uppercase with single underscores, strip spaces
    s = re.sub(r"[\s\-]+", "_", raw.upper().strip())
    s = re.sub(r"_+", "_", s)

    # Order matters: more specific patterns first
    if re.search(r"CLAIM_COUNTER|COUNTER.*CLAIM", s):
        return "CLAIM_COUNTERFACTUAL"
    if re.search(r"CLAIM_CAUS|CAUS.*CLAIM|CAUSAL_LINK", s):
        return "CLAIM_CAUSAL"
    if re.search(r"CLAIM_THEORY|THEORY.*MIND", s):
        return "CLAIM_THEORY_OF_MIND"
    if re.search(r"SUBMIT|FINAL_REPORT|REPORT_SUBMIT", s):
        return "SUBMIT_REPORT"
    if re.search(r"FLAG.*HYPOTH|HYPOTH.*FLAG", s):
        return "FLAG_HYPOTHESIS"
    # QUERY_COUNTERFACTUAL — must come before QUERY_FEATURE to avoid false match
    if re.search(r"COUNTER|COUNTERFACT|COUNTERTACT|COUNTERTACTUAL|RUN_COUNTER", s):
        return "QUERY_COUNTERFACTUAL"
    # QUERY_FEATURE_DISTRIBUTION — covers all typos/plurals/spacing variants
    if re.search(r"FEAT.*DIST|DIST.*FEAT|QUERY_FEAT|FEATURE_DIST", s):
        return "QUERY_FEATURE_DISTRIBUTION"
    if re.search(r"QUERY_RECORD|QUERY_NODE|PERFORM_QUER|QUERYRECORD|INITIATE_QUERY|REQUEST.*DATA", s):
        return "QUERY_RECORDS"

    return "__UNKNOWN__"


# Entries whose action strings carry zero training signal — drop them entirely
_GARBAGE_PATTERNS = re.compile(
    r"STEP[_\s]*(FORWARD|UP|INIT|INVESTIGATION|\d)|"
    r"INITIALIZE|BEGIN_INVEST|START_INVEST|TRAIN_MODEL|TRAIN$|"
    r"SEND.*EMAIL|SEND.*LOAN|APPROVE|APPROVED|SEND_QUERY_REQUEST|"
    r"RUN_QUERIES$|CONTINUE_MONITOR|STEP$|STEPFORWARD|"
    r"REQUEST_MORE_INFO|REQUEST_ADDITIONAL",
    re.IGNORECASE,
)


def normalize_action(action_dict: dict) -> dict:
    """
    Normalise non-standard action schemas produced by the LLM back to
    our environment's expected format.
    """
    # Already valid
    if action_dict.get("type") in VALID_TYPES:
        return action_dict

    action_val = action_dict.get("action", action_dict.get("next_action", ""))
    query_type = action_dict.get("query_type", "")

    # Flatten nested action dict: {"action": {"query_type": "...", ...}}
    if isinstance(action_val, dict):
        query_type = action_val.get("query_type", action_val.get("type", query_type))
        action_val = action_val.get("type", action_val.get("action", ""))

    # Combine action value and query_type hint for matching
    combined = f"{action_val} {query_type}".strip()

    if not combined.strip():
        action_dict["type"] = "__UNKNOWN__"
        return action_dict

    # Check for garbage before attempting classification
    if _GARBAGE_PATTERNS.search(combined):
        action_dict["type"] = "__GARBAGE__"
        return action_dict

    canonical = _classify_action_string(combined)
    action_dict["type"] = canonical
    return action_dict


def normalize_to_expected_format(raw: dict, canonical_type: str) -> dict:
    """
    Convert any LLM response dict to the exact format env.step() expects.

    Handles every known deviation:
      - "action" key instead of "type"
      - claim fields flat at top level instead of nested under "claim"
      - action value is a nested dict
      - wrong key names (subject_record vs record_id, etc.)
      - placeholder / invalid enum values
    """
    # Flatten nested action value: {"action": {"type": "X", "cause_feature": ...}}
    flat = dict(raw)
    action_val = flat.get("action", "")
    if isinstance(action_val, dict):
        for k, v in action_val.items():
            if k not in ("type", "action"):
                flat.setdefault(k, v)

    # Also merge "next_action" dict if present
    next_action = flat.get("next_action", "")
    if isinstance(next_action, dict):
        for k, v in next_action.items():
            flat.setdefault(k, v)

    if canonical_type == "QUERY_RECORDS":
        return {
            "type": "QUERY_RECORDS",
            "feature_filter":  flat.get("feature_filter") or {},
            "outcome_filter":  flat.get("outcome_filter"),
            "time_range":      flat.get("time_range"),
        }

    if canonical_type == "QUERY_FEATURE_DISTRIBUTION":
        # feature_id may be in "feature_id", "features" array, "node_id", "feature"
        features = flat.get("features")
        feature_id = (
            flat.get("feature_id")
            or flat.get("feature")
            or flat.get("node_id")
            or (features[0] if isinstance(features, list) and features else None)
            or ""
        )
        group_by = flat.get("group_by")
        if isinstance(group_by, list):
            group_by = group_by[0] if group_by else None
        return {
            "type":       "QUERY_FEATURE_DISTRIBUTION",
            "feature_id": str(feature_id),
            "group_by":   group_by,
        }

    if canonical_type == "QUERY_COUNTERFACTUAL":
        record_id = (
            flat.get("record_id")
            or flat.get("subject_record")
            or flat.get("record")
            or flat.get("subject_record_id")
            or ""
        )
        if isinstance(record_id, dict):
            record_id = record_id.get("id", "")
        feature_id = (
            flat.get("feature_id")
            or flat.get("counterfactual_feature")
            or flat.get("feature")
            or ""
        )
        return {
            "type":                "QUERY_COUNTERFACTUAL",
            "record_id":           str(record_id),
            "feature_id":          str(feature_id),
            "counterfactual_value": flat.get("counterfactual_value"),
        }

    if canonical_type == "FLAG_HYPOTHESIS":
        status = str(flat.get("status", "ACTIVE")).upper()
        if status not in ("ACTIVE", "WEAKENED", "ELIMINATED"):
            status = "ACTIVE"
        return {
            "type":             "FLAG_HYPOTHESIS",
            "hypothesis_type":  flat.get("hypothesis_type", ""),
            "status":           status,
        }

    if canonical_type == "CLAIM_CAUSAL":
        # Claim fields may be at top level OR nested under "claim"
        claim_src = flat.get("claim", {})
        if not isinstance(claim_src, dict) or not claim_src:
            claim_src = flat   # fall back to top level

        direction = str(claim_src.get("direction", "")).lower()
        direction = direction.replace("-ve", "negative").replace("+ve", "positive")
        if direction not in ("positive", "negative"):
            direction = "negative"

        anomaly_type = claim_src.get("anomaly_type", "")
        if anomaly_type not in ("proxy_discrimination", "adversarial_injection", "model_drift"):
            anomaly_type = "proxy_discrimination"

        confidence = str(claim_src.get("confidence", "MEDIUM")).upper()
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "MEDIUM"

        return {
            "type": "CLAIM_CAUSAL",
            "claim": {
                "cause_feature":  claim_src.get("cause_feature", ""),
                "effect_outcome": claim_src.get("effect_outcome", "denial_rate_overall"),
                "mechanism":      claim_src.get("mechanism", ""),
                "direction":      direction,
                "confidence":     confidence,
                "basis_records":  claim_src.get("basis_records") or [],
                "anomaly_type":   anomaly_type,
            },
        }

    if canonical_type == "CLAIM_COUNTERFACTUAL":
        claim_src = flat.get("claim", {})
        if not isinstance(claim_src, dict) or not claim_src:
            claim_src = flat

        subject = claim_src.get("subject_record", "")
        if isinstance(subject, dict):
            subject = subject.get("id", "")

        predicted = str(claim_src.get("predicted_outcome_change", "")).lower()
        if predicted not in ("approved", "denied", "no_change"):
            predicted = "denied"

        confidence = str(claim_src.get("confidence", "MEDIUM")).upper()
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "MEDIUM"

        basis = claim_src.get("basis", "")
        if isinstance(basis, list):
            basis = "; ".join(str(b) for b in basis)

        return {
            "type": "CLAIM_COUNTERFACTUAL",
            "claim": {
                "subject_record":           str(subject),
                "counterfactual_feature":   claim_src.get("counterfactual_feature", ""),
                "predicted_outcome_change": predicted,
                "confidence":               confidence,
                "basis":                    str(basis),
            },
        }

    if canonical_type == "SUBMIT_REPORT":
        # Fields may be at top level or nested under "claim" or inner action dict
        report_src = flat.get("claim", {})
        if not isinstance(report_src, dict) or not report_src:
            report_src = flat

        anomaly_type = report_src.get("anomaly_type", "")
        if anomaly_type not in ("proxy_discrimination", "adversarial_injection", "model_drift"):
            anomaly_type = "unknown"

        recommended = report_src.get("recommended_action", "audit")
        if recommended not in ("retrain", "audit", "halt"):
            recommended = "audit"

        chain = report_src.get("primary_evidence_chain") or []
        if not isinstance(chain, list):
            chain = []

        return {
            "type":                    "SUBMIT_REPORT",
            "anomaly_type":            anomaly_type,
            "primary_evidence_chain":  chain,
            "affected_demographic":    report_src.get("affected_demographic", ""),
            "recommended_action":      recommended,
        }

    # Fallback — shouldn't reach here for valid types
    return {"type": canonical_type}


# ── Main cleanup ──────────────────────────────────────────────────────────────

def main():
    if not INPUT.exists():
        print(f"[ERROR] {INPUT} not found. Is generation still running?")
        return

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    print(f"[INFO] Input lines: {len(lines)}")
    print(f"[INFO] Input trajectories (~): {len(lines) // 8}")

    stats = Counter()
    seen  = set()
    clean = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            stats["skipped_empty"] += 1
            continue

        # --- Parse outer record ---
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_outer_json"] += 1
            continue

        prompt   = record.get("prompt", "").strip()
        response = record.get("response", "").strip()
        step     = record.get("step", -1)
        level    = record.get("level", -1)

        # --- Skip empty / null ---
        if not prompt or not response:
            stats["empty_prompt_or_response"] += 1
            continue

        # --- Deduplicate ---
        key = (prompt[:200], response[:200])
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)

        # --- Extract + normalize response JSON ---
        try:
            action_dict, cleaned_response = extract_json_from_response(response)
        except ValueError:
            stats["unparseable_response"] += 1
            continue

        action_dict = normalize_action(action_dict)
        action_type = action_dict.get("type", "__UNKNOWN__")

        if action_type == "__GARBAGE__":
            stats["garbage_dropped"] += 1
            continue
        elif action_type == "__UNKNOWN__":
            stats["unknown_action_type"] += 1
            # Drop unknowns — no reliable training signal
            continue
        elif action_type in VALID_TYPES:
            stats[f"valid_{action_type}"] += 1

        # Convert to exact env.step() format
        normalized = normalize_to_expected_format(action_dict, action_type)
        normalized_response = json.dumps(normalized)

        stats["kept"] += 1

        clean.append(json.dumps({
            "prompt":      prompt,
            "response":    normalized_response,
            "step":        step,
            "level":       level,
            "action_type": action_type,
        }))

    # --- Write output ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(clean), encoding="utf-8")

    # --- Report ---
    print("\n" + "="*55)
    print("           TRAJECTORY CLEANUP REPORT")
    print("="*55)
    print(f"  Input lines          : {len(lines)}")
    print(f"  Kept (clean)         : {stats['kept']}")
    print(f"  Output trajectories~ : {stats['kept'] // 8}")
    print(f"  Duplicates removed   : {stats['duplicates']}")
    print(f"  Bad outer JSON       : {stats['bad_outer_json']}")
    print(f"  Unparseable response : {stats['unparseable_response']}")
    print(f"  Empty lines          : {stats['skipped_empty']}")
    print(f"  Garbage dropped      : {stats['garbage_dropped']}")
    print(f"  Unknown action type  : {stats['unknown_action_type']}")
    print("-"*55)
    print("  Action type breakdown:")
    for k, v in sorted(stats.items()):
        if k.startswith("valid_"):
            print(f"    {k.replace('valid_',''):35s} {v}")
    print("="*55)
    print(f"\n[OK] Clean file written to: {OUTPUT}")
    print(f"[OK] Ready for: python arbiter/training/train_sft.py "
          f"--dataset {OUTPUT}")


if __name__ == "__main__":
    main()
