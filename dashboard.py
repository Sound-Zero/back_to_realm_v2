"""Streamlit dashboard for PPO training metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


METRICS_FILE = Path("metrics.json")
CHECKPOINT_DIR = Path("ckpt/dump_model")


def load_metrics(metrics_file: Path = METRICS_FILE) -> pd.DataFrame:
    """Load the metrics JSON file produced by `PPO.algorithm.Algorithm`."""

    if not metrics_file.exists():
        return pd.DataFrame()
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()
    return pd.DataFrame(data)


def render_dashboard() -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Install dashboard dependencies with `pip install -e .[dashboard]`.") from exc

    st.set_page_config(page_title="PPO Training Dashboard", layout="wide")
    st.title("PPO Training Dashboard")

    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0)
    df = load_metrics()
    if df.empty:
        st.warning("No metrics found yet. Start training to generate metrics.json.")
        time.sleep(refresh_interval)
        st.rerun()
        return

    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learn steps", int(latest["learn_cnt"]))
    col2.metric("Total loss", f"{latest['total_loss']:.4f}")
    col3.metric("Mean reward", f"{latest['reward_mean']:.4f}")
    col4.metric("Learning rate", f"{latest['lr']:.6f}")

    metrics_data = {
        "TD return mean": latest["tdret_mean"],
        "Value loss": latest["value_loss"],
        "Policy loss": latest["policy_loss"],
        "Entropy loss": latest["entropy_loss"],
        "Advantage mean": latest["adv_mean"],
        "Advantage std": latest["adv_std"],
        "Clip fraction": latest["clip_frac"],
    }
    st.subheader("Latest Metrics")
    st.dataframe(pd.DataFrame(metrics_data.items(), columns=["Metric", "Value"]), hide_index=True)

    if len(df) > 1:
        st.subheader("Training Curves")
        fig_loss = make_subplots(specs=[[{"secondary_y": False}]])
        fig_loss.add_trace(go.Scatter(x=df["learn_cnt"], y=df["total_loss"], name="Total loss"))
        fig_loss.add_trace(go.Scatter(x=df["learn_cnt"], y=df["value_loss"], name="Value loss"))
        fig_loss.add_trace(go.Scatter(x=df["learn_cnt"], y=df["policy_loss"], name="Policy loss"))
        fig_loss.add_trace(go.Scatter(x=df["learn_cnt"], y=df["entropy_loss"], name="Entropy loss"))
        fig_loss.update_layout(title="Loss by Learn Step", xaxis_title="Learn step", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.line(df, x="learn_cnt", y="reward_mean", title="Mean Reward"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.line(df, x="learn_cnt", y="adv_mean", title="Mean Advantage"), use_container_width=True)

    st.subheader("Checkpoints")
    if CHECKPOINT_DIR.exists():
        checkpoints = sorted(path.name for path in CHECKPOINT_DIR.glob("*.pkl"))
        if checkpoints:
            st.dataframe(pd.DataFrame({"file": checkpoints}), hide_index=True)
        else:
            st.info("No checkpoint files found.")
    else:
        st.info("Checkpoint directory does not exist yet.")

    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    render_dashboard()
