"""Meta-Overseer Consistency Checker for ARBITER.

Watches the Auditor's claim chain and flags internally contradictory pairs.
Does NOT know ground truth — only checks logical consistency.

Contradiction rules:
  1. Claim A asserts X → Y (HIGH), Claim B asserts Y → X (HIGH)  [directional reversal]
  2. Claim A says feature F has NO causal effect; Claim B says F → outcome  [existence conflict]
  3. Claim A says anomaly_type = T1; Claim B says anomaly_type = T2  [type conflict]
  4. Two HIGH-confidence counterfactuals on the same record predict opposite outcomes  [cf conflict]
"""
from typing import Dict, List, Tuple


def check_consistency(claims: List[Dict]) -> Dict:
    """
    Check the Auditor's full claim chain for contradictions.

    Parameters
    ----------
    claims : list of claim dicts (with 'claim_type' key)

    Returns
    -------
    dict:
        violations      – list of (claim_idx_a, claim_idx_b, reason)
        num_violations  – int
        penalty         – float (sum of -1.0 per violation)
    """
    violations: List[Tuple[int, int, str]] = []

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            reason = _detect_contradiction(a, b)
            if reason:
                violations.append((i, j, reason))

    num = len(violations)
    return {
        "violations":     violations,
        "num_violations": num,
        "penalty":        -1.0 * num,
    }


def _detect_contradiction(a: Dict, b: Dict) -> str:
    """Return a contradiction reason string, or '' if no contradiction."""

    a_type = a.get("claim_type", "")
    b_type = b.get("claim_type", "")

    # ── Rule 1: Directional reversal (both causal claims, HIGH confidence) ──
    if a_type == "causal" and b_type == "causal":
        if a.get("confidence") == "HIGH" and b.get("confidence") == "HIGH":
            if (a.get("cause_feature") == b.get("effect_outcome") and
                    a.get("effect_outcome") == b.get("cause_feature")):
                return (f"Directional reversal: claim {a.get('cause_feature')}→"
                        f"{a.get('effect_outcome')} contradicts "
                        f"{b.get('cause_feature')}→{b.get('effect_outcome')}")

    # ── Rule 2: Anomaly type conflict ───────────────────────────────────────
    if a_type == "causal" and b_type == "causal":
        at_a = a.get("anomaly_type", "")
        at_b = b.get("anomaly_type", "")
        if at_a and at_b and at_a != at_b and \
                a.get("confidence") in ("HIGH", "MEDIUM") and \
                b.get("confidence") in ("HIGH", "MEDIUM"):
            return (f"Anomaly type conflict: claim asserts '{at_a}' "
                    f"while another asserts '{at_b}'")

    # ── Rule 3: Counterfactual contradiction on same record ─────────────────
    if a_type == "counterfactual" and b_type == "counterfactual":
        if (a.get("subject_record") == b.get("subject_record") and
                a.get("counterfactual_feature") == b.get("counterfactual_feature") and
                a.get("predicted_outcome_change") != b.get("predicted_outcome_change")):
            return (f"Counterfactual conflict on {a.get('subject_record')}: "
                    f"predicts '{a.get('predicted_outcome_change')}' vs "
                    f"'{b.get('predicted_outcome_change')}'")

    # ── Rule 4: Causal claim says feature has no effect, CF says it does ───
    if a_type == "causal" and b_type == "counterfactual":
        if (a.get("cause_feature") == b.get("counterfactual_feature") and
                a.get("direction") == "none" and
                b.get("predicted_outcome_change") not in ("no_change", None)):
            return (f"Existence conflict: causal claim says "
                    f"{a.get('cause_feature')} has no effect, but "
                    f"counterfactual predicts outcome change")

    # Symmetric check for rule 4
    if b_type == "causal" and a_type == "counterfactual":
        if (b.get("cause_feature") == a.get("counterfactual_feature") and
                b.get("direction") == "none" and
                a.get("predicted_outcome_change") not in ("no_change", None)):
            return (f"Existence conflict: causal claim says "
                    f"{b.get('cause_feature')} has no effect, but "
                    f"counterfactual predicts outcome change")

    # ── Rule 5: Hypothesis flip-flop (ELIMINATED → ACTIVE for same type) ────
    if a_type == "hypothesis_flag" and b_type == "hypothesis_flag":
        if (a.get("hypothesis_type") == b.get("hypothesis_type") and
                a.get("status") == "ELIMINATED" and b.get("status") == "ACTIVE"):
            return (f"Hypothesis flip-flop: re-activated eliminated "
                    f"hypothesis {a.get('hypothesis_type')}")

    return ""
