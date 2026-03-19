"""
RAGRouter Eval Dashboard — Streamlit app for exploring evaluation results.

Usage:
    streamlit run eval/dashboard.py
    streamlit run eval/dashboard.py -- --results-dir eval/results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
DATASET_FILE = EVAL_DIR / "data" / "eval_dataset.jsonl"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAGRouter Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_results(results_dir: str) -> tuple[pd.DataFrame, dict, dict | None]:
    """Load eval results from a run directory."""
    rd = Path(results_dir)

    # Load individual results
    results_file = rd / "results.jsonl"
    if not results_file.exists():
        return pd.DataFrame(), {}, None

    rows = []
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            flat = {
                "id": r["id"],
                "question": r["question"],
                "ground_truth": r["ground_truth_answer"],
                "question_type": r["question_type"],
                "difficulty": r["difficulty"],
                "doc_types": ", ".join(r.get("doc_types", [])),
                # Arbiter
                "arbiter_answer": r.get("arbiter_answer", ""),
                "arbiter_engine": r.get("arbiter_engine", ""),
                "arbiter_latency_ms": r.get("arbiter_latency_ms", 0),
                "arbiter_routing_mode": r.get("arbiter_routing_mode", ""),
                "arbiter_routing_intent": r.get("arbiter_routing_intent", ""),
                "arbiter_error": r.get("arbiter_error", ""),
                # Retrieval
                "arbiter_hit": r.get("arbiter_retrieval", {}).get("hit", False),
                "arbiter_mrr": r.get("arbiter_retrieval", {}).get("reciprocal_rank", 0),
                "arbiter_precision": r.get("arbiter_retrieval", {}).get("precision_at_k", 0),
                "arbiter_recall": r.get("arbiter_retrieval", {}).get("recall", 0),
                # Answer quality
                "arbiter_faithfulness": r.get("arbiter_answer_metrics", {}).get("faithfulness", 0),
                "arbiter_relevance": r.get("arbiter_answer_metrics", {}).get("relevance", 0),
                "arbiter_correctness": r.get("arbiter_answer_metrics", {}).get("correctness", 0),
                "arbiter_completeness": r.get("arbiter_answer_metrics", {}).get("completeness", 0),
                "n_citations": len(r.get("arbiter_citations", [])),
                # Hybrid comparison
                "hybrid_answer": r.get("hybrid_answer", ""),
                "hybrid_latency_ms": r.get("hybrid_latency_ms", 0),
                "hybrid_correctness": r.get("hybrid_answer_metrics", {}).get("correctness", 0),
                "hybrid_relevance": r.get("hybrid_answer_metrics", {}).get("relevance", 0),
                "hybrid_error": r.get("hybrid_error", ""),
            }
            rows.append(flat)

    df = pd.DataFrame(rows)

    # Load aggregates
    agg_file = rd / "aggregates.json"
    agg = json.loads(agg_file.read_text()) if agg_file.exists() else {}

    # Load summary
    summary_file = rd / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else None

    return df, agg, summary


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the eval dataset for reference."""
    if not DATASET_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(DATASET_FILE) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["ground_truth_answer"],
                "type": r["question_type"],
                "difficulty": r["difficulty"],
                "doc_types": ", ".join(r.get("doc_types", [])),
                "source": r.get("metadata", {}).get("source", ""),
            })
    return pd.DataFrame(rows)


def find_runs() -> list[Path]:
    """Find all eval run directories."""
    if not RESULTS_DIR.exists():
        return []
    runs = sorted(RESULTS_DIR.iterdir(), reverse=True)
    return [r for r in runs if r.is_dir() and (r / "results.jsonl").exists()]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("RAGRouter Eval")

runs = find_runs()
tab_selection = st.sidebar.radio(
    "View",
    ["Dashboard", "Dataset Explorer", "Question Detail"],
    index=0,
)

selected_run = None
if runs:
    run_names = [r.name for r in runs]
    selected_idx = st.sidebar.selectbox("Eval Run", range(len(run_names)), format_func=lambda i: run_names[i])
    selected_run = runs[selected_idx]
else:
    st.sidebar.warning("No eval runs found. Run `python eval/run_eval.py` first.")


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

if tab_selection == "Dashboard":
    st.title("RAGRouter Evaluation Dashboard")

    if not selected_run:
        st.info("No evaluation results found. Run the eval first:\n\n"
                "```bash\npython eval/run_eval.py --arbiter-url http://localhost:8000\n```")
        st.stop()

    df, agg, summary = load_results(str(selected_run))

    if df.empty:
        st.error("No results in selected run.")
        st.stop()

    # --- Top-level metrics ---
    st.header("Overall Performance")

    arbiter_agg = agg.get("arbiter", {})
    retrieval = arbiter_agg.get("retrieval", {})
    answer = arbiter_agg.get("answer", {})
    latency = arbiter_agg.get("latency", {})

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Questions", agg.get("total_questions", 0))
    col2.metric("Success Rate", f"{agg.get('successful', 0)}/{agg.get('total_questions', 0)}")
    col3.metric("Hit Rate", f"{retrieval.get('hit_rate', 0):.1%}")
    col4.metric("MRR", f"{retrieval.get('mrr', 0):.3f}")
    col5.metric("Avg Correctness", f"{answer.get('avg_correctness', 0):.2f}")
    col6.metric("Avg Latency", f"{latency.get('avg_ms', 0):.0f}ms")

    st.divider()

    # --- Answer quality breakdown ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Answer Quality")
        quality_data = {
            "Metric": ["Faithfulness", "Relevance", "Correctness", "Completeness"],
            "Score": [
                answer.get("avg_faithfulness", 0),
                answer.get("avg_relevance", 0),
                answer.get("avg_correctness", 0),
                answer.get("avg_completeness", 0),
            ],
        }
        fig_quality = px.bar(
            quality_data,
            x="Metric",
            y="Score",
            color="Metric",
            range_y=[0, 1],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_quality.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_quality, use_container_width=True)

    with col_right:
        st.subheader("Engine Distribution")
        engine_dist = arbiter_agg.get("engine_distribution", {})
        if engine_dist:
            fig_engine = px.pie(
                names=list(engine_dist.keys()),
                values=list(engine_dist.values()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_engine.update_layout(height=350)
            st.plotly_chart(fig_engine, use_container_width=True)
        else:
            st.info("No engine distribution data")

    st.divider()

    # --- By question type ---
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Performance by Question Type")
        by_type = agg.get("by_question_type", {})
        if by_type:
            type_df = pd.DataFrame([
                {
                    "Type": qt,
                    "Correctness": v.get("avg_correctness", 0),
                    "Relevance": v.get("avg_relevance", 0),
                    "Hit Rate": v.get("hit_rate", 0),
                    "Count": v.get("count", 0),
                }
                for qt, v in by_type.items()
            ])
            fig_type = px.bar(
                type_df,
                x="Type",
                y=["Correctness", "Relevance", "Hit Rate"],
                barmode="group",
                range_y=[0, 1],
                text_auto=".2f",
            )
            fig_type.update_layout(height=400, legend_title="Metric")
            st.plotly_chart(fig_type, use_container_width=True)

    with col_right2:
        st.subheader("Performance by Difficulty")
        by_diff = agg.get("by_difficulty", {})
        if by_diff:
            diff_df = pd.DataFrame([
                {
                    "Difficulty": d,
                    "Correctness": v.get("avg_correctness", 0),
                    "Avg Latency (ms)": v.get("avg_latency_ms", 0),
                    "Count": v.get("count", 0),
                }
                for d, v in by_diff.items()
            ])
            # Reorder
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            diff_df["order"] = diff_df["Difficulty"].map(diff_order)
            diff_df = diff_df.sort_values("order").drop(columns=["order"])

            fig_diff = px.bar(
                diff_df,
                x="Difficulty",
                y="Correctness",
                color="Difficulty",
                range_y=[0, 1],
                text="Count",
                color_discrete_map={"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"},
            )
            fig_diff.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_diff, use_container_width=True)

    st.divider()

    # --- Latency distribution ---
    st.subheader("Latency Distribution")
    col_lat1, col_lat2 = st.columns(2)

    with col_lat1:
        valid_df = df[df["arbiter_error"] == ""]
        if not valid_df.empty:
            fig_lat = px.histogram(
                valid_df,
                x="arbiter_latency_ms",
                nbins=30,
                color="arbiter_engine",
                labels={"arbiter_latency_ms": "Latency (ms)"},
            )
            fig_lat.update_layout(height=350)
            st.plotly_chart(fig_lat, use_container_width=True)

    with col_lat2:
        if not valid_df.empty:
            fig_lat_box = px.box(
                valid_df,
                x="question_type",
                y="arbiter_latency_ms",
                color="question_type",
                labels={"arbiter_latency_ms": "Latency (ms)"},
            )
            fig_lat_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_lat_box, use_container_width=True)

    st.divider()

    # --- Routing intent distribution ---
    st.subheader("Routing Analysis")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        intent_dist = arbiter_agg.get("routing_intent_distribution", {})
        if intent_dist:
            fig_intent = px.bar(
                x=list(intent_dist.keys()),
                y=list(intent_dist.values()),
                labels={"x": "Intent", "y": "Count"},
                color=list(intent_dist.keys()),
            )
            fig_intent.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_intent, use_container_width=True)

    with col_r2:
        # Correctness vs latency scatter
        if not valid_df.empty:
            fig_scatter = px.scatter(
                valid_df,
                x="arbiter_latency_ms",
                y="arbiter_correctness",
                color="question_type",
                size="n_citations",
                hover_data=["question", "arbiter_engine"],
                labels={
                    "arbiter_latency_ms": "Latency (ms)",
                    "arbiter_correctness": "Correctness",
                },
            )
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Hybrid comparison (if available) ---
    hybrid_agg = agg.get("hybrid", {})
    if hybrid_agg:
        st.divider()
        st.subheader("Arbiter vs Direct Hybrid Comparison")

        comp_data = {
            "System": ["Arbiter", "Hybrid"],
            "Correctness": [
                answer.get("avg_correctness", 0),
                hybrid_agg.get("answer", {}).get("avg_correctness", 0),
            ],
            "Relevance": [
                answer.get("avg_relevance", 0),
                hybrid_agg.get("answer", {}).get("avg_relevance", 0),
            ],
            "Hit Rate": [
                retrieval.get("hit_rate", 0),
                hybrid_agg.get("retrieval", {}).get("hit_rate", 0),
            ],
            "Avg Latency (ms)": [
                latency.get("avg_ms", 0),
                hybrid_agg.get("latency", {}).get("avg_ms", 0),
            ],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Side by side bar chart
        fig_comp = px.bar(
            comp_df.melt(id_vars="System", value_vars=["Correctness", "Relevance", "Hit Rate"]),
            x="variable",
            y="value",
            color="System",
            barmode="group",
            range_y=[0, 1],
            labels={"variable": "Metric", "value": "Score"},
        )
        fig_comp.update_layout(height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Errors ---
    errors_df = df[df["arbiter_error"] != ""]
    if not errors_df.empty:
        st.divider()
        st.subheader(f"Errors ({len(errors_df)})")
        st.dataframe(
            errors_df[["id", "question", "arbiter_error"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Dataset Explorer tab
# ---------------------------------------------------------------------------

elif tab_selection == "Dataset Explorer":
    st.title("Eval Dataset Explorer")

    dataset_df = load_dataset()
    if dataset_df.empty:
        st.warning("No dataset found. Run `python eval/build_dataset.py generate` first.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        type_filter = st.multiselect("Question Type", dataset_df["type"].unique(), default=list(dataset_df["type"].unique()))
    with col_f2:
        diff_filter = st.multiselect("Difficulty", dataset_df["difficulty"].unique(), default=list(dataset_df["difficulty"].unique()))
    with col_f3:
        source_filter = st.multiselect("Source", dataset_df["source"].unique(), default=list(dataset_df["source"].unique()))

    filtered = dataset_df[
        (dataset_df["type"].isin(type_filter))
        & (dataset_df["difficulty"].isin(diff_filter))
        & (dataset_df["source"].isin(source_filter))
    ]

    st.metric("Questions", len(filtered))

    # Distribution charts
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.pie(filtered, names="type", title="By Type")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(filtered, names="difficulty", title="By Difficulty",
                     color_discrete_map={"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.pie(filtered, names="source", title="By Source")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(
        filtered[["id", "type", "difficulty", "question", "answer", "doc_types"]],
        use_container_width=True,
        hide_index=True,
        height=500,
    )


# ---------------------------------------------------------------------------
# Question Detail tab
# ---------------------------------------------------------------------------

elif tab_selection == "Question Detail":
    st.title("Question Detail View")

    if not selected_run:
        st.info("Select an eval run from the sidebar.")
        st.stop()

    df, agg, summary = load_results(str(selected_run))

    if df.empty:
        st.error("No results in selected run.")
        st.stop()

    # Question selector
    questions = df["question"].tolist()
    selected_q_idx = st.selectbox(
        "Select Question",
        range(len(questions)),
        format_func=lambda i: f"[{df.iloc[i]['question_type']}/{df.iloc[i]['difficulty']}] {questions[i][:100]}",
    )

    row = df.iloc[selected_q_idx]

    # Question info
    st.subheader("Question")
    st.markdown(f"**{row['question']}**")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Type:** `{row['question_type']}`")
    col2.markdown(f"**Difficulty:** `{row['difficulty']}`")
    col3.markdown(f"**Doc Types:** `{row['doc_types']}`")

    st.divider()

    # Ground truth vs system answer
    col_gt, col_sys = st.columns(2)

    with col_gt:
        st.subheader("Ground Truth")
        st.markdown(row["ground_truth"])

    with col_sys:
        st.subheader(f"Arbiter Answer ({row['arbiter_engine']})")
        if row["arbiter_error"]:
            st.error(f"Error: {row['arbiter_error']}")
        else:
            st.markdown(row["arbiter_answer"])

    st.divider()

    # Metrics
    st.subheader("Scores")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Correctness", f"{row['arbiter_correctness']:.2f}")
    col_m2.metric("Relevance", f"{row['arbiter_relevance']:.2f}")
    col_m3.metric("Faithfulness", f"{row['arbiter_faithfulness']:.2f}")
    col_m4.metric("Completeness", f"{row['arbiter_completeness']:.2f}")

    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
    col_m5.metric("Latency", f"{row['arbiter_latency_ms']:.0f}ms")
    col_m6.metric("Citations", int(row["n_citations"]))
    col_m7.metric("Routing Intent", row["arbiter_routing_intent"])
    col_m8.metric("Routing Mode", row["arbiter_routing_mode"])

    # Hybrid comparison
    if row.get("hybrid_answer"):
        st.divider()
        st.subheader("Hybrid Direct Comparison")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**Hybrid Answer:**")
            st.markdown(row["hybrid_answer"])
        with col_h2:
            st.metric("Hybrid Correctness", f"{row['hybrid_correctness']:.2f}",
                       delta=f"{row['hybrid_correctness'] - row['arbiter_correctness']:.2f}")
            st.metric("Hybrid Latency", f"{row['hybrid_latency_ms']:.0f}ms",
                       delta=f"{row['hybrid_latency_ms'] - row['arbiter_latency_ms']:.0f}ms",
                       delta_color="inverse")
