"""ARBITER Main Environment — OpenEnv-compatible wrapper.

Implements:
  reset()  → observation
  step(action) → (observation, reward, done, info)
  render() → visualization dict
  get_metrics() → aggregate stats across sessions

Action format (dict):
  {"type": "QUERY_RECORDS",           "feature_filter": ..., "outcome_filter": ..., "time_range": ...}
  {"type": "QUERY_FEATURE_DISTRIBUTION","feature_id": ..., "group_by": ...}
  {"type": "QUERY_COUNTERFACTUAL",    "record_id": ..., "feature_id": ..., "counterfactual_value": ...}
  {"type": "FLAG_HYPOTHESIS",         "hypothesis_type": ..., "status": ...}
  {"type": "CLAIM_CAUSAL",            "claim": {...}}
  {"type": "CLAIM_COUNTERFACTUAL",    "claim": {...}}
  {"type": "CLAIM_THEORY_OF_MIND",    "claim": {...}}
  {"type": "SUBMIT_REPORT",           "anomaly_type": ..., "primary_evidence_chain": [...],
                                      "affected_demographic": ..., "recommended_action": ...}
"""
import uuid
from typing import Any, Dict, List, Optional

from .graph         import generate_graph
from .decoys        import generate_decoys
from .counterfactual import intervene
from .claims        import (CausalLinkClaim, CounterfactualClaim, TheoryOfMindClaim,
                             verify_causal_claim, verify_counterfactual_claim,
                             verify_theory_of_mind_claim)
from .reward        import intermediate_claim_reward, compute_episode_reward
from .meta_overseer import check_consistency
from .defender      import Defender
from .curriculum    import Curriculum

from config import QUERY_BUDGET


class ArbiterEnv:
    """Single-session ARBITER environment."""

    def __init__(self, level: int = 1, seed: Optional[int] = None):
        self.curriculum  = Curriculum(start_level=level)
        self.defender    = Defender(level=level, seed=seed)
        self.seed        = seed
        self._episode_seed = seed
        self._reset_episode_state()

        # Aggregate metrics
        self._metrics: Dict[str, Any] = {
            "episodes_completed": 0,
            "total_reward":       0.0,
            "correct_verdicts":   0,
            "level_advances":     [],
        }

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Start a new episode. Returns initial observation."""
        if seed is not None:
            self._episode_seed = seed
        self._reset_episode_state()
        self.defender.reset_episode(level=self.curriculum.level)

        # Generate graph
        ep_data = generate_graph(
            seed=self._episode_seed,
            anomaly_type=None,
            num_decisions=45,
        )

        # Inject decoys
        decoy_data = generate_decoys(ep_data["records"], ep_data["features"])
        ep_data["records"]  = decoy_data["records"]
        ep_data["decoy_a"]  = decoy_data["decoy_a"]
        ep_data["decoy_b"]  = decoy_data["decoy_b"]

        # Defender obfuscates if active
        if self.curriculum.defender_active:
            ep_data = self.defender.obfuscate(ep_data, self._hypothesis_flags)

        self._ep = ep_data
        self._anomaly_info = ep_data["anomaly_info"]

        if self._episode_seed is not None:
            self._episode_seed += 1

        return self._observation()

    def step(self, action: Dict) -> tuple:
        """
        Process one action.
        Returns (observation, reward, done, info)
        """
        assert self._ep is not None, "Call reset() first."
        if self._done:
            return self._observation(), 0.0, True, {"error": "episode already done"}

        # Accept both "type" (canonical) and "action" (common LLM output key)
        atype = action.get("type") or action.get("action", "")
        if isinstance(atype, dict):
            atype = atype.get("type", atype.get("query_type", ""))
        atype = (atype or "").upper().strip()
        reward = 0.0
        info   = {}

        if atype == "QUERY_RECORDS":
            result = self._handle_query_records(action)
            self._budget -= 1
            info["query_result"] = result

        elif atype == "QUERY_FEATURE_DISTRIBUTION":
            result = self._handle_query_distribution(action)
            self._budget -= 1
            info["query_result"] = result

        elif atype == "QUERY_COUNTERFACTUAL":
            result = self._handle_query_counterfactual(action)
            self._budget -= 2
            info["query_result"] = result
            info["cf_result"]    = result

        elif atype == "FLAG_HYPOTHESIS":
            self._handle_flag_hypothesis(action)

        elif atype == "CLAIM_CAUSAL":
            claim  = CausalLinkClaim(**action["claim"], step=self._step)
            vresult = verify_causal_claim(claim, self._anomaly_info)
            reward  = intermediate_claim_reward(vresult)
            self._claims.append({**claim.to_dict(), "claim_type": "causal"})
            self._claim_rewards.append(reward)
            info["verification"] = vresult

        elif atype == "CLAIM_COUNTERFACTUAL":
            claim   = CounterfactualClaim(**action["claim"], step=self._step)
            last_cf = info.get("cf_result") or self._last_cf_result
            if last_cf:
                vresult = verify_counterfactual_claim(claim, last_cf)
                reward  = intermediate_claim_reward(vresult)
            else:
                vresult = {"error": "No counterfactual result available. Run QUERY_COUNTERFACTUAL first."}
            self._claims.append({**claim.to_dict(), "claim_type": "counterfactual"})
            self._claim_rewards.append(reward)
            info["verification"] = vresult

        elif atype == "CLAIM_THEORY_OF_MIND":
            if not self.curriculum.tom_claims_enabled:
                info["error"] = "Theory-of-mind claims only available at Level 4+"
            else:
                claim   = TheoryOfMindClaim(**action["claim"], step=self._step)
                vresult = verify_theory_of_mind_claim(claim, self.defender.action_log)
                reward  = intermediate_claim_reward(vresult)
                self._claims.append({**claim.to_dict(), "claim_type": "theory_of_mind"})
                self._claim_rewards.append(reward)
                info["verification"] = vresult

        elif atype == "SUBMIT_REPORT":
            return self._handle_submit_report(action)

        # Meta-Overseer consistency check after every claim
        if atype.startswith("CLAIM_"):
            consistency = check_consistency(self._claims)
            self._consistency_violations = consistency["num_violations"]
            info["consistency"] = consistency

        self._step += 1
        if self._budget <= 0 or self._step >= 20:
            # Force submit with empty verdict if budget exhausted
            return self._handle_submit_report({"type": "SUBMIT_REPORT",
                                                "anomaly_type": "unknown",
                                                "primary_evidence_chain": [],
                                                "affected_demographic": "unknown",
                                                "recommended_action": "audit"})

        return self._observation(), reward, False, info

    def render(self) -> Dict:
        """Return visualization-ready data."""
        if self._ep is None:
            return {}
        return {
            "graph_nodes":       [{"id": n, **d} for n, d in self._ep["observable_graph"].nodes(data=True)],
            "graph_edges":       [{"source": u, "target": v, **d} for u, v, d in self._ep["observable_graph"].edges(data=True)],
            "queried_nodes":     list(self._queried_nodes),
            "claims":            self._claims,
            "claim_rewards":     self._claim_rewards,
            "hypothesis_flags":  self._hypothesis_flags,
            "budget_remaining":  self._budget,
            "step":              self._step,
            "running_reward":    sum(self._claim_rewards),
        }

    def get_metrics(self) -> Dict:
        """Aggregate metrics across all episodes (for /metrics endpoint)."""
        total_eps = self._metrics["episodes_completed"]
        return {
            **self._metrics,
            "mean_reward":     round(self._metrics["total_reward"] / max(1, total_eps), 2),
            "accuracy":        round(self._metrics["correct_verdicts"] / max(1, total_eps), 3),
            "current_level":   self.curriculum.level,
            "curriculum_stats": self.curriculum.get_stats(),
        }

    # ── Query handlers ─────────────────────────────────────────────────────────

    def _handle_query_records(self, action: Dict) -> List[Dict]:
        ff  = action.get("feature_filter", {})
        of_ = action.get("outcome_filter")
        tr  = action.get("time_range")

        results = []
        for rec in self._ep["records"]:
            fv = {**rec["feature_vector"], **rec["proxy_vector"]}
            if ff and not all(fv.get(k) == v for k, v in ff.items()):
                continue
            if of_ and rec["outcome"] != of_:
                continue
            if tr and not (tr[0] <= rec["timestamp"] <= tr[1]):
                continue
            # Return record without hidden_vector
            results.append({k: v for k, v in rec.items() if k != "hidden_vector"})

        for r in results:
            self._queried_nodes.update(r.get("feature_vector", {}).keys())
        return results

    def _handle_query_distribution(self, action: Dict) -> Dict:
        feature_id = action.get("feature_id", "")
        group_by   = action.get("group_by")

        dist: Dict = {}
        for rec in self._ep["records"]:
            fv = {**rec["feature_vector"], **rec["proxy_vector"]}
            val = fv.get(feature_id, "unknown")
            if group_by:
                gv = fv.get(group_by, "all")
                dist.setdefault(gv, {})
                dist[gv][val] = dist[gv].get(val, 0) + 1
            else:
                dist[val] = dist.get(val, 0) + 1

        self._queried_nodes.add(feature_id)
        return {"feature": feature_id, "group_by": group_by, "distribution": dist}

    def _handle_query_counterfactual(self, action: Dict) -> Dict:
        record_id = action.get("record_id", "")
        feature_id = action.get("feature_id", "")
        cf_value   = action.get("counterfactual_value")

        rec = next((r for r in self._ep["records"] if r["id"] == record_id), None)
        if rec is None:
            return {"error": f"Record {record_id} not found"}

        result = intervene(self._ep["graph"], rec, feature_id, cf_value)
        self._last_cf_result = result
        self._queried_nodes.add(feature_id)
        return result

    def _handle_flag_hypothesis(self, action: Dict):
        htype  = action.get("hypothesis_type", "")
        status = action.get("status", "ACTIVE")
        self._hypothesis_flags[htype] = status

    def _handle_submit_report(self, action: Dict) -> tuple:
        """Terminal action — compute full episode reward."""
        self._done = True

        verdict = {
            "anomaly_type":         action.get("anomaly_type", "unknown"),
            "primary_evidence_chain": action.get("primary_evidence_chain", []),
            "affected_demographic": action.get("affected_demographic", "unknown"),
            "recommended_action":   action.get("recommended_action", "audit"),
        }

        true_chain = self._anomaly_info.get("causal_chain", [])
        claimed_chain = verdict["primary_evidence_chain"]

        decoy_states = {
            "decoy_a": self._hypothesis_flags.get("decoy_a", "ACTIVE"),
            "decoy_b": self._hypothesis_flags.get("decoy_b", "ACTIVE"),
        }

        result = compute_episode_reward(
            claim_rewards=self._claim_rewards,
            consistency_violations=self._consistency_violations,
            remaining_budget=max(0, self._budget),
            claimed_chain=claimed_chain,
            true_chain=true_chain,
            verdict=verdict,
            anomaly_info=self._anomaly_info,
            decoy_states=decoy_states,
        )

        # Update aggregate metrics
        self._metrics["episodes_completed"] += 1
        self._metrics["total_reward"]       += result["total"]
        if result["terminal"]["verdict_correct"]:
            self._metrics["correct_verdicts"] += 1

        # Curriculum advancement
        new_level = self.curriculum.record(result["total"])
        if new_level:
            self._metrics["level_advances"].append(new_level)

        info = {
            "episode_reward": result,
            "verdict":        verdict,
            "ground_truth":   self._anomaly_info,
            "new_level":      new_level,
        }
        return self._observation(), result["total"], True, info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _reset_episode_state(self):
        self._ep                    = None
        self._step                  = 0
        self._budget                = QUERY_BUDGET
        self._done                  = False
        self._claims: List[Dict]    = []
        self._claim_rewards: List[float] = []
        self._consistency_violations = 0
        self._queried_nodes         = set()
        self._hypothesis_flags: Dict[str, str] = {}
        self._last_cf_result: Optional[Dict] = None
        self._anomaly_info: Dict    = {}

    def _observation(self) -> Dict:
        if self._ep is None:
            return {}
        return {
            "step":             self._step,
            "budget_remaining": self._budget,
            "queried_nodes":    list(self._queried_nodes),
            "hypothesis_flags": self._hypothesis_flags,
            "num_claims":       len(self._claims),
            "level":            self.curriculum.level,
            "features":         self._ep.get("features", {}),
        }


# ── Multi-session factory ──────────────────────────────────────────────────────

_sessions: Dict[str, ArbiterEnv] = {}


def create_session(level: int = 1, seed: Optional[int] = None) -> str:
    """Create a new UUID-keyed session. Returns session_id."""
    sid = str(uuid.uuid4())
    _sessions[sid] = ArbiterEnv(level=level, seed=seed)
    return sid


def get_session(session_id: str) -> Optional[ArbiterEnv]:
    return _sessions.get(session_id)


def list_sessions() -> List[str]:
    return list(_sessions.keys())
