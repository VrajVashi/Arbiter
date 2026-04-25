"""OpenEnv-compliant wrapper for ARBITER.

Subclasses openenv.core.Environment so that ArbiterEnv is a proper
first-class OpenEnv environment, not just "API-compatible".

Usage (server):
    from arbiter.env.openenv_wrapper import ArbiterEnvironment, ArbiterAction
    from openenv.core import create_app

    env = ArbiterEnvironment()
    app = create_app(env, ArbiterAction)

Usage (direct):
    env = ArbiterEnvironment(level=1)
    obs = env.reset(seed=0)
    obs = env.step(ArbiterAction(type="QUERY_RECORDS", feature_filter={}))
    print(obs.reward, obs.done)
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core import Action, Environment, Observation, State

from .environment import ArbiterEnv  # existing logic stays untouched


# ── Typed Models ──────────────────────────────────────────────────────────────

class ArbiterAction(Action):
    """Any action the Auditor agent can take inside an ARBITER episode.

    The ``type`` field selects the action kind; all other fields are
    passed through to the underlying ArbiterEnv.step() as keyword args.
    """
    type: str = Field(
        description=(
            "Action type. One of: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, "
            "QUERY_COUNTERFACTUAL, FLAG_HYPOTHESIS, CLAIM_CAUSAL, "
            "CLAIM_COUNTERFACTUAL, CLAIM_THEORY_OF_MIND, SUBMIT_REPORT"
        )
    )
    # Optional payload fields — present only for the relevant action type
    feature_filter:       Optional[Dict[str, Any]] = None
    outcome_filter:       Optional[str]            = None
    time_range:           Optional[List[float]]    = None
    feature_id:           Optional[str]            = None
    group_by:             Optional[str]            = None
    record_id:            Optional[str]            = None
    counterfactual_value: Optional[Any]            = None
    hypothesis_type:      Optional[str]            = None
    status:               Optional[str]            = None
    claim:                Optional[Dict[str, Any]] = None
    # SUBMIT_REPORT fields
    anomaly_type:            Optional[str]       = None
    primary_evidence_chain:  Optional[List[str]] = None
    affected_demographic:    Optional[str]       = None
    recommended_action:      Optional[str]       = None

    model_config = {"extra": "allow"}  # allow extra fields for forward compat


class ArbiterObservation(Observation):
    """Observation returned after every reset() or step()."""
    step:             int              = 0
    budget_remaining: int              = 20
    queried_nodes:    List[str]        = Field(default_factory=list)
    hypothesis_flags: Dict[str, str]   = Field(default_factory=dict)
    num_claims:       int              = 0
    level:            int              = 1
    features:         Dict[str, Any]   = Field(default_factory=dict)
    # Populated after step() only
    query_result:     Optional[Any]    = None
    verification:     Optional[Any]    = None
    episode_reward:   Optional[Any]    = None
    ground_truth:     Optional[Any]    = None

    model_config = {"extra": "allow"}


class ArbiterState(State):
    """Internal episode state exposed via env.state."""
    level:               int  = 1
    episodes_completed:  int  = 0
    total_reward:        float = 0.0
    correct_verdicts:    int  = 0
    current_level:       int  = 1


# ── OpenEnv Environment ───────────────────────────────────────────────────────

class ArbiterEnvironment(Environment):
    """ARBITER wrapped as a proper openenv.core.Environment subclass.

    Delegates all game logic to the existing ArbiterEnv, but exposes the
    OpenEnv-standard interface: reset() / step() return typed Observations,
    and state is a typed ArbiterState.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, level: int = 1, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._env = ArbiterEnv(level=level, seed=seed)
        self._episode_id: Optional[str] = None

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ArbiterObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._reset_rubric()
        raw_obs = self._env.reset(seed=seed)
        return self._to_observation(raw_obs, reward=None, done=False)

    def step(
        self,
        action: ArbiterAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ArbiterObservation:
        # Convert typed Action → plain dict that ArbiterEnv.step() expects
        action_dict = action.model_dump(exclude_none=True, exclude={"metadata"})

        raw_obs, reward, done, info = self._env.step(action_dict)

        obs = self._to_observation(raw_obs, reward=reward, done=done)
        obs.query_result   = info.get("query_result")
        obs.verification   = info.get("verification")
        obs.episode_reward = info.get("episode_reward")
        obs.ground_truth   = info.get("ground_truth")
        return obs

    @property
    def state(self) -> ArbiterState:
        metrics = self._env.get_metrics()
        return ArbiterState(
            episode_id=self._episode_id,
            step_count=self._env._step,
            level=self._env.curriculum.level,
            episodes_completed=metrics["episodes_completed"],
            total_reward=metrics["total_reward"],
            correct_verdicts=metrics["correct_verdicts"],
            current_level=metrics["current_level"],
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_observation(
        self,
        raw: Dict[str, Any],
        reward: Optional[float],
        done: bool,
    ) -> ArbiterObservation:
        return ArbiterObservation(
            done=done,
            reward=reward,
            step=raw.get("step", 0),
            budget_remaining=raw.get("budget_remaining", 20),
            queried_nodes=raw.get("queried_nodes", []),
            hypothesis_flags=raw.get("hypothesis_flags", {}),
            num_claims=raw.get("num_claims", 0),
            level=raw.get("level", self._env.curriculum.level),
            features=raw.get("features", {}),
        )
