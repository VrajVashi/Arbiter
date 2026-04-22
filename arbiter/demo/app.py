"""ARBITER Gradio Demo Interface.

Panels:
  Left:   Causal decision graph (NetworkX → Matplotlib → Gradio)
  Right:  Claim chain with real-time green/red correctness coloring
  Bottom: Running reward breakdown
  Tab 2:  Arms race dual-curve graph (Auditor reward vs Defender evasion)

Usage:
    python -m arbiter.demo.app
"""
import sys
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arbiter.env.environment import ArbiterEnv

# ── Global demo state ─────────────────────────────────────────────────────────
_env: ArbiterEnv = None
_obs = None
_render = None
_arms_race_data = {"auditor": [], "defender": []}


def _get_env(level: int = 3) -> ArbiterEnv:
    global _env, _obs, _render
    _env = ArbiterEnv(level=level)
    _obs = _env.reset()
    _render = _env.render()
    return _env


# ── Graph visualization ────────────────────────────────────────────────────────

def draw_graph(render_data: dict) -> plt.Figure:
    """Render the observable causal graph with color-coded nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not render_data or not render_data.get("graph_nodes"):
        ax.text(0.5, 0.5, "No graph data", color="white", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    G = nx.DiGraph()
    for node in render_data["graph_nodes"]:
        G.add_node(node["id"], **node)
    for edge in render_data["graph_edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)

    queried = set(render_data.get("queried_nodes", []))

    # Color coding
    colors = []
    for n, d in G.nodes(data=True):
        if n in queried:
            colors.append("#fbbf24")   # yellow — queried
        elif d.get("proxy"):
            colors.append("#f87171")   # red-ish — proxy (potentially anomalous)
        elif d.get("node_type") == "outcome":
            colors.append("#60a5fa")   # blue — outcome
        elif d.get("node_type") == "policy":
            colors.append("#a78bfa")   # purple — policy
        else:
            colors.append("#4ade80")   # green — benign explicit feature

    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = nx.random_layout(G)

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=600,
        font_color="white", font_size=6, font_weight="bold",
        edge_color="#64748b", arrows=True, arrowsize=12,
        width=1.2,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#fbbf24", label="Queried"),
        mpatches.Patch(color="#f87171", label="Proxy/Anomalous"),
        mpatches.Patch(color="#60a5fa", label="Outcome"),
        mpatches.Patch(color="#a78bfa", label="Policy"),
        mpatches.Patch(color="#4ade80", label="Benign Feature"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", facecolor="#1e293b",
              labelcolor="white", fontsize=7, framealpha=0.8)
    ax.axis("off")
    ax.set_title("Causal Decision Graph", color="white", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


# ── Claim chain display ────────────────────────────────────────────────────────

def format_claim_chain(render_data: dict) -> str:
    """Format claims as colored HTML."""
    claims  = render_data.get("claims", [])
    rewards = render_data.get("claim_rewards", [])

    if not claims:
        return "<p style='color:#94a3b8; font-style:italic;'>No claims yet.</p>"

    html = "<div style='font-family: monospace; font-size: 12px;'>"
    for i, (claim, r) in enumerate(zip(claims, rewards)):
        color  = "#4ade80" if r > 0 else "#f87171"
        bg     = "#14532d" if r > 0 else "#450a0a"
        ctype  = claim.get("claim_type", "causal").upper()
        html += (
            f"<div style='background:{bg}; border-left: 3px solid {color}; "
            f"padding:6px 10px; margin:4px 0; border-radius:4px;'>"
            f"<b style='color:{color};'>[{ctype}]</b> "
            f"<span style='color:#e2e8f0;'>{_format_claim_text(claim)}</span>"
            f"<span style='color:{color}; float:right;'>+{r:.2f}</span>"
            f"</div>"
        )
    html += "</div>"
    return html


def _format_claim_text(claim: dict) -> str:
    ctype = claim.get("claim_type", "causal")
    if ctype == "causal":
        return (f"{claim.get('cause_feature','?')} → {claim.get('effect_outcome','?')} "
                f"via {claim.get('mechanism','?')} [{claim.get('confidence','?')}]")
    elif ctype == "counterfactual":
        return (f"If {claim.get('counterfactual_feature','?')} = {claim.get('counterfactual_value','?')} "
                f"on {claim.get('subject_record','?')} → {claim.get('predicted_outcome_change','?')}")
    elif ctype == "theory_of_mind":
        return (f"Defender {claim.get('defender_action','?')} "
                f"{claim.get('target_link','?')} via {claim.get('obfuscation_method','?')}")
    return json.dumps(claim)


# ── Reward breakdown panel ─────────────────────────────────────────────────────

def draw_reward_panel(render_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    rewards = render_data.get("claim_rewards", [])
    if not rewards:
        ax.text(0.5, 0.5, "No rewards yet", color="#94a3b8", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    cumulative = np.cumsum(rewards)
    colors = ["#4ade80" if r > 0 else "#f87171" for r in rewards]

    ax.bar(range(len(rewards)), rewards, color=colors, alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(range(len(cumulative)), cumulative, color="#fbbf24", linewidth=2, marker="o", markersize=3)
    ax2.tick_params(colors="white")

    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.set_xlabel("Claim #", color="white", fontsize=8)
    ax.set_ylabel("Claim Reward", color="white", fontsize=8)
    ax2.set_ylabel("Cumulative", color="#fbbf24", fontsize=8)
    ax.set_title(f"Reward Breakdown  |  Running Total: {cumulative[-1]:.2f}",
                 color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


# ── Arms race graph ────────────────────────────────────────────────────────────

def draw_arms_race(auditor_rewards: list, defender_evasion: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    if not auditor_rewards:
        ax.text(0.5, 0.5, "Train the model to see arms race curves",
                color="#94a3b8", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    eps = range(len(auditor_rewards))
    ax.plot(eps, auditor_rewards, color="#60a5fa", linewidth=2, label="Auditor Reward")
    if defender_evasion:
        ax.plot(eps, defender_evasion[:len(auditor_rewards)],
                color="#f87171", linewidth=2, label="Defender Evasion Rate", linestyle="--")

    # Annotate inflection points
    if len(auditor_rewards) > 100:
        ax.axvline(100, color="#fbbf24", linestyle=":", alpha=0.6)
        ax.text(102, max(auditor_rewards)*0.1, "Defender adapts", color="#fbbf24", fontsize=7)
    if len(auditor_rewards) > 200:
        ax.axvline(200, color="#4ade80", linestyle=":", alpha=0.6)
        ax.text(202, max(auditor_rewards)*0.1, "Auditor catches up", color="#4ade80", fontsize=7)

    ax.set_xlabel("Training Episode", color="white", fontsize=9)
    ax.set_ylabel("Score", color="white", fontsize=9)
    ax.set_title("Arms Race: Auditor vs Defender Co-Evolution", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


# ── Demo actions (used by Gradio buttons) ─────────────────────────────────────

def run_query(query_type: str, param1: str, param2: str):
    global _obs, _render
    if _env is None:
        return draw_graph({}), "<p>Start episode first.</p>", draw_reward_panel({})

    if query_type == "QUERY_RECORDS":
        action = {"type": "QUERY_RECORDS", "feature_filter": {}, "outcome_filter": param1 or None}
    elif query_type == "QUERY_FEATURE_DISTRIBUTION":
        action = {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": param1, "group_by": param2 or None}
    elif query_type == "QUERY_COUNTERFACTUAL":
        action = {"type": "QUERY_COUNTERFACTUAL", "record_id": param1,
                  "feature_id": "zip_code_cluster", "counterfactual_value": "cluster_3"}
    else:
        return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)

    _obs, _, done, _ = _env.step(action)
    _render = _env.render()
    return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)


def new_episode(level: int):
    global _env, _obs, _render
    _get_env(level=int(level))
    return draw_graph(_render), format_claim_chain(_render), draw_reward_panel(_render)


# ── Gradio layout ──────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        body { background: #0f172a; }
        .gradio-container { background: #0f172a !important; }
        h1, h2, h3, label { color: #e2e8f0 !important; }
        """,
        title="ARBITER — AI Oversight Training Environment",
    ) as demo:
        gr.Markdown(
            "# 🔍 ARBITER — AI Oversight Training Environment\n"
            "_Autonomous Reasoning-Based Inspector for Training Environments with Recursive Oversight_"
        )

        with gr.Tabs():
            with gr.Tab("🔬 Live Episode"):
                with gr.Row():
                    level_slider = gr.Slider(1, 7, value=3, step=1, label="Curriculum Level")
                    start_btn    = gr.Button("▶ New Episode", variant="primary")

                with gr.Row():
                    graph_plot   = gr.Plot(label="Causal Decision Graph")
                    with gr.Column():
                        claim_html   = gr.HTML(label="Claim Chain")
                        with gr.Row():
                            q_type  = gr.Dropdown(
                                ["QUERY_RECORDS", "QUERY_FEATURE_DISTRIBUTION", "QUERY_COUNTERFACTUAL"],
                                label="Query Type", value="QUERY_RECORDS")
                            p1      = gr.Textbox(label="Param 1 (outcome / feature_id / record_id)", value="")
                            p2      = gr.Textbox(label="Param 2 (group_by)", value="")
                            q_btn   = gr.Button("Run Query", variant="secondary")

                reward_plot  = gr.Plot(label="Reward Breakdown")

                start_btn.click(new_episode, inputs=[level_slider],
                                outputs=[graph_plot, claim_html, reward_plot])
                q_btn.click(run_query, inputs=[q_type, p1, p2],
                            outputs=[graph_plot, claim_html, reward_plot])

                # Initialize on load
                demo.load(lambda: new_episode(3), outputs=[graph_plot, claim_html, reward_plot])

            with gr.Tab("📈 Arms Race"):
                gr.Markdown("### Auditor Reward vs Defender Evasion Rate over Training")
                arms_plot = gr.Plot(label="Arms Race Co-Evolution")
                refresh_btn = gr.Button("Refresh")

                def refresh_arms():
                    return draw_arms_race(
                        _arms_race_data["auditor"],
                        _arms_race_data["defender"],
                    )

                refresh_btn.click(refresh_arms, outputs=[arms_plot])
                demo.load(refresh_arms, outputs=[arms_plot])

    return demo


def main():
    _get_env(level=3)
    demo = build_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
