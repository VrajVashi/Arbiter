"""FastAPI REST Server for ARBITER — OpenEnv-compliant.

Endpoints:
  POST /sessions                    → create session, returns session_id
  POST /sessions/{id}/reset         → reset episode
  POST /sessions/{id}/step          → take action, returns (obs, reward, done, info)
  GET  /sessions/{id}/render        → graph + claim chain visualization data
  GET  /sessions/{id}/metrics       → session aggregate metrics
  GET  /metrics                     → global aggregate metrics
  GET  /explain/{session_id}        → post-hoc ground truth explanation
  GET  /health                      → health check

Usage:
    pip install fastapi uvicorn
    python -m arbiter.server
    # Then open http://localhost:8000/docs for auto-generated Swagger UI
"""
from __future__ import annotations

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import time

from arbiter.env.environment import (
    ArbiterEnv, create_session, get_session, list_sessions
)
from arbiter.env.openenv_wrapper import ArbiterEnvironment, ArbiterAction, ArbiterObservation

# ── OpenEnv-native app (for TRL / Unsloth GRPO integration) ───────────────
try:
    from openenv.core import create_app as _openenv_create_app
    _arbiter_openenv_env = ArbiterEnvironment(level=1)
    openenv_app = _openenv_create_app(_arbiter_openenv_env, ArbiterAction)
except Exception as _e:
    openenv_app = None
    print(f"[server] OpenEnv native app skipped: {_e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARBITER — AI Oversight Training Environment",
    description=(
        "OpenEnv-compatible REST API for the ARBITER multi-agent RL environment. "
        "Implements co-evolutionary auditing: Auditor vs Adaptive Defender over 7 curriculum levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ──────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "demo", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        """Serve the React frontend index.html."""
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

_global_start = time.time()
_global_stats: Dict[str, Any] = {
    "total_sessions":  0,
    "total_episodes":  0,
    "total_rewards":   [],
    "verdicts_correct": 0,
}


# ── Schemas ────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    level: int = 1
    seed: Optional[int] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class SessionResponse(BaseModel):
    session_id: str
    level: int
    created: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - _global_start, 1)}


@app.post("/sessions", response_model=SessionResponse)
def create_session_endpoint(req: CreateSessionRequest):
    sid = create_session(level=req.level, seed=req.seed)
    _global_stats["total_sessions"] += 1
    return {"session_id": sid, "level": req.level, "created": time.time()}


@app.get("/sessions")
def list_sessions_endpoint():
    return {"sessions": list_sessions(), "count": len(list_sessions())}


@app.post("/sessions/{session_id}/reset")
def reset_endpoint(session_id: str, req: ResetRequest):
    env = _get_env(session_id)
    obs = env.reset(seed=req.seed)
    _global_stats["total_episodes"] += 1
    return {"observation": obs}


@app.post("/sessions/{session_id}/step")
def step_endpoint(session_id: str, req: StepRequest):
    env = _get_env(session_id)
    obs, reward, done, info = env.step(req.action)

    if done and "episode_reward" in info:
        r = info["episode_reward"]["total"]
        _global_stats["total_rewards"].append(r)
        if info["episode_reward"]["terminal"].get("verdict_correct"):
            _global_stats["verdicts_correct"] += 1

    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        _serialize(info),
    }


@app.get("/sessions/{session_id}/render")
def render_endpoint(session_id: str):
    env = _get_env(session_id)
    return env.render()


@app.get("/sessions/{session_id}/metrics")
def session_metrics_endpoint(session_id: str):
    env = _get_env(session_id)
    return env.get_metrics()


@app.get("/explain/{session_id}")
def explain_endpoint(session_id: str):
    """Post-hoc ground truth explanation for the current episode."""
    env = _get_env(session_id)
    if env._anomaly_info is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return {
        "anomaly_info":    env._anomaly_info,
        "decoy_a":         env._ep.get("decoy_a", {}),
        "decoy_b":         env._ep.get("decoy_b", {}),
        "defender_log":    env.defender.action_log,
        "curriculum":      env.curriculum.get_stats(),
    }


@app.get("/metrics")
def global_metrics():
    """Global aggregate metrics across all sessions."""
    rewards = _global_stats["total_rewards"]
    return {
        "total_sessions":   _global_stats["total_sessions"],
        "total_episodes":   _global_stats["total_episodes"],
        "mean_reward":      round(sum(rewards) / max(1, len(rewards)), 2),
        "verdict_accuracy": round(_global_stats["verdicts_correct"] / max(1, len(rewards)), 3),
        "uptime_s":         round(time.time() - _global_start, 1),
    }


@app.get("/leaderboard")
def leaderboard():
    """Top-10 sessions by mean episode reward."""
    from arbiter.env.environment import _sessions
    rows = []
    for sid, env in _sessions.items():
        m = env.get_metrics()
        rows.append({
            "session_id":   sid,
            "level":        m["current_level"],
            "mean_reward":  m["mean_reward"],
            "accuracy":     m["accuracy"],
            "episodes":     m["episodes_completed"],
        })
    rows.sort(key=lambda r: r["mean_reward"], reverse=True)
    return {"leaderboard": rows[:10]}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_env(session_id: str) -> ArbiterEnv:
    env = get_session(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env


def _serialize(obj):
    """Make objects JSON-safe."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "item"):   # numpy scalars
        return obj.item()
    return obj


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Starting ARBITER API server...")
    print("REST API docs:   http://localhost:8000/docs")
    if openenv_app:
        print("OpenEnv app:     http://localhost:8000/openenv")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# Mount OpenEnv sub-app if available (enables /openenv/* WebSocket endpoints)
if openenv_app:
    app.mount("/openenv", openenv_app)


if __name__ == "__main__":
    main()
