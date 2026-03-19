"""
RAGRouter Eval Dashboard — Streamlit app for exploring evaluation results.

Usage:
    streamlit run eval/dashboard.py --server.port 8701
    streamlit run eval/dashboard.py --server.port 8701 -- --results-dir eval/results
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
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


def correctness_reason(gt: str, answer: str, score: float) -> str:
    """Explain why the correctness score is what it is."""
    if not answer or not gt:
        return "empty answer or ground truth"

    gt_words = set(re.findall(r"\w+", gt.lower()))
    ans_words = set(re.findall(r"\w+", answer.lower()))
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "then", "than", "that", "this", "it", "its",
    }
    gt_words -= stopwords
    ans_words -= stopwords
    overlap = gt_words & ans_words
    missing = gt_words - ans_words

    parts = [f"F1={score:.2f}"]
    parts.append(f"overlap={len(overlap)}/{len(gt_words)} key terms")
    if missing:
        sample = sorted(missing)[:8]
        parts.append(f"missing: {', '.join(sample)}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("RAGRouter Eval")

tab_selection = st.sidebar.radio(
    "View",
    ["Dashboard", "Results Table", "Run Eval", "Dataset Explorer", "Question Detail"],
    index=0,
)

runs = find_runs()
selected_run = None
if runs:
    run_names = [r.name for r in runs]
    selected_idx = st.sidebar.selectbox(
        "Eval Run", range(len(run_names)), format_func=lambda i: run_names[i]
    )
    selected_run = runs[selected_idx]
else:
    st.sidebar.warning("No eval runs found yet.")


# ---------------------------------------------------------------------------
# Run Eval tab
# ---------------------------------------------------------------------------

if tab_selection == "Run Eval":
    st.title("Run Evaluation")

    st.markdown(
        "Trigger a batch evaluation against the arbiter. "
        "Results appear in the sidebar once complete."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        arbiter_url = st.text_input("Arbiter URL", value="http://localhost:8000")
        limit = st.number_input(
            "Question limit (0 = all)", min_value=0, max_value=500, value=0, step=5
        )
    with col_cfg2:
        run_name = st.text_input("Run name", value="eval")
        verbose = st.checkbox("Verbose (show answers in console)", value=False)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        hybrid_url = st.text_input("Hybrid URL (optional, for comparison)", value="")
    with col_opt2:
        judge_url = st.text_input("LLM Judge URL (optional)", value="")

    if st.button("Start Eval", type="primary", use_container_width=True):
        cmd = [
            sys.executable, str(EVAL_DIR / "run_eval.py"),
            "--arbiter-url", arbiter_url,
            "--run-name", run_name,
        ]
        if limit > 0:
            cmd += ["--limit", str(limit)]
        if verbose:
            cmd += ["--verbose"]
        if hybrid_url:
            cmd += ["--hybrid-url", hybrid_url]
        if judge_url:
            cmd += ["--judge-url", judge_url]

        st.info(f"Running: `{' '.join(cmd)}`")

        log_area = st.empty()
        progress_bar = st.progress(0)
        log_lines: list[str] = []

        # Count total questions
        total_q = 0
        if DATASET_FILE.exists():
            with open(DATASET_FILE) as f:
                total_q = sum(1 for _ in f)
        if limit > 0:
            total_q = min(total_q, limit)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        completed = 0
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            log_lines.append(line)
            # Parse progress from "[N/M]" pattern
            m = re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                completed = int(m.group(1))
                total = int(m.group(2))
                progress_bar.progress(completed / total if total > 0 else 0)
            # Show last 30 lines
            log_area.code("\n".join(log_lines[-30:]), language="text")

        proc.wait()
        progress_bar.progress(1.0)

        if proc.returncode == 0:
            st.success("Eval complete! Switch to Dashboard or Results Table to view.")
            # Clear cache so new run shows up
            load_results.clear()
            find_runs()
            st.rerun()
        else:
            st.error(f"Eval failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

elif tab_selection == "Dashboard":
    st.title("RAGRouter Evaluation Dashboard")

    if not selected_run:
        st.info(
            "No evaluation results found. Go to **Run Eval** to trigger one, or run:\n\n"
            "```bash\njust eval\n```"
        )
        st.stop()

    df, agg, summary = load_results(str(selected_run))

    if df.empty:
        st.error("No results in selected run.")
        st.stop()

    # --- Global accuracy score ---
    arbiter_agg = agg.get("arbiter", {})
    retrieval = arbiter_agg.get("retrieval", {})
    answer = arbiter_agg.get("answer", {})
    latency = arbiter_agg.get("latency", {})

    total_q = agg.get("total_questions", 0)
    successful = agg.get("successful", 0)
    avg_correctness = answer.get("avg_correctness", 0)
    avg_relevance = answer.get("avg_relevance", 0)
    avg_completeness = answer.get("avg_completeness", 0)
    hit_rate = retrieval.get("hit_rate", 0)

    # Composite accuracy: weighted blend of correctness, relevance, completeness
    composite_accuracy = (avg_correctness * 0.4 + avg_relevance * 0.3 + avg_completeness * 0.3)

    # --- Big accuracy gauge + key metrics ---
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=composite_accuracy * 100,
            number={"suffix": "%"},
            title={"text": "Overall Accuracy"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if composite_accuracy > 0.6 else "#f39c12" if composite_accuracy > 0.3 else "#e74c3c"},
                "steps": [
                    {"range": [0, 30], "color": "#fdecea"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#eafaf1"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": composite_accuracy * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("Weighted: 40% correctness + 30% relevance + 30% completeness")

    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Questions", f"{successful}/{total_q}")
        m2.metric("Hit Rate", f"{hit_rate:.0%}")
        m3.metric("MRR", f"{retrieval.get('mrr', 0):.3f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Correctness", f"{avg_correctness:.2f}")
        m5.metric("Relevance", f"{avg_relevance:.2f}")
        m6.metric("Completeness", f"{avg_completeness:.2f}")

        m7, m8, m9 = st.columns(3)
        m7.metric("Avg Latency", f"{latency.get('avg_ms', 0):.0f}ms")
        m8.metric("p95 Latency", f"{latency.get('p95_ms', 0):.0f}ms")
        m9.metric("Errors", agg.get("errors", 0))

    st.divider()

    # --- Score breakdown pie + bar ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Answer Quality")
        quality_data = {
            "Metric": ["Faithfulness", "Relevance", "Correctness", "Completeness"],
            "Score": [
                answer.get("avg_faithfulness", 0),
                avg_relevance,
                avg_correctness,
                avg_completeness,
            ],
        }
        fig_quality = px.bar(
            quality_data,
            x="Metric",
            y="Score",
            color="Metric",
            range_y=[0, 1],
            text_auto=".2f",
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

    # --- Correctness distribution histogram ---
    valid_df = df[df["arbiter_error"] == ""]
    if not valid_df.empty:
        st.subheader("Correctness Distribution")
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            fig_cdist = px.histogram(
                valid_df,
                x="arbiter_correctness",
                nbins=20,
                color="question_type",
                labels={"arbiter_correctness": "Correctness Score"},
            )
            fig_cdist.update_layout(height=300, bargap=0.05)
            st.plotly_chart(fig_cdist, use_container_width=True)

        with col_hist2:
            # Accuracy buckets pie chart
            def bucket(score):
                if score >= 0.7:
                    return "Good (>=0.7)"
                elif score >= 0.4:
                    return "Partial (0.4-0.7)"
                elif score > 0:
                    return "Low (>0-0.4)"
                else:
                    return "Zero (0)"

            valid_df = valid_df.copy()
            valid_df["accuracy_bucket"] = valid_df["arbiter_correctness"].apply(bucket)
            bucket_counts = valid_df["accuracy_bucket"].value_counts()
            fig_buckets = px.pie(
                names=bucket_counts.index,
                values=bucket_counts.values,
                title="Answer Quality Buckets",
                color=bucket_counts.index,
                color_discrete_map={
                    "Good (>=0.7)": "#2ecc71",
                    "Partial (0.4-0.7)": "#f39c12",
                    "Low (>0-0.4)": "#e67e22",
                    "Zero (0)": "#e74c3c",
                },
            )
            fig_buckets.update_layout(height=300)
            st.plotly_chart(fig_buckets, use_container_width=True)

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
                avg_correctness,
                hybrid_agg.get("answer", {}).get("avg_correctness", 0),
            ],
            "Relevance": [
                avg_relevance,
                hybrid_agg.get("answer", {}).get("avg_relevance", 0),
            ],
            "Hit Rate": [
                hit_rate,
                hybrid_agg.get("retrieval", {}).get("hit_rate", 0),
            ],
            "Avg Latency (ms)": [
                latency.get("avg_ms", 0),
                hybrid_agg.get("latency", {}).get("avg_ms", 0),
            ],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

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
# Results Table tab
# ---------------------------------------------------------------------------

elif tab_selection == "Results Table":
    st.title("Eval Results Table")

    if not selected_run:
        st.info("No evaluation results. Go to **Run Eval** to trigger one.")
        st.stop()

    df, agg, summary = load_results(str(selected_run))

    if df.empty:
        st.error("No results in selected run.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        type_filter = st.multiselect(
            "Question Type",
            df["question_type"].unique(),
            default=list(df["question_type"].unique()),
        )
    with col_f2:
        diff_filter = st.multiselect(
            "Difficulty",
            df["difficulty"].unique(),
            default=list(df["difficulty"].unique()),
        )
    with col_f3:
        score_range = st.slider("Correctness range", 0.0, 1.0, (0.0, 1.0), 0.05)

    filtered = df[
        (df["question_type"].isin(type_filter))
        & (df["difficulty"].isin(diff_filter))
        & (df["arbiter_correctness"] >= score_range[0])
        & (df["arbiter_correctness"] <= score_range[1])
    ].copy()

    st.metric("Showing", f"{len(filtered)}/{len(df)} questions")

    # Add reason column
    filtered["reason"] = filtered.apply(
        lambda r: correctness_reason(
            r["ground_truth"], r["arbiter_answer"], r["arbiter_correctness"]
        ),
        axis=1,
    )

    # Color-code correctness
    def color_correctness(val):
        if val >= 0.7:
            return "background-color: #d5f5e3"
        elif val >= 0.4:
            return "background-color: #fef9e7"
        elif val > 0:
            return "background-color: #fdebd0"
        else:
            return "background-color: #fadbd8"

    display_cols = [
        "question", "arbiter_answer", "ground_truth", "reason",
        "arbiter_correctness", "arbiter_relevance", "arbiter_completeness",
        "arbiter_engine", "arbiter_latency_ms",
    ]
    display_df = filtered[display_cols].rename(columns={
        "question": "Question",
        "arbiter_answer": "Answer",
        "ground_truth": "Expected",
        "reason": "Reason",
        "arbiter_correctness": "Correctness",
        "arbiter_relevance": "Relevance",
        "arbiter_completeness": "Completeness",
        "arbiter_engine": "Engine",
        "arbiter_latency_ms": "Latency (ms)",
    })

    st.dataframe(
        display_df.style.applymap(color_correctness, subset=["Correctness"]),
        use_container_width=True,
        hide_index=True,
        height=700,
        column_config={
            "Question": st.column_config.TextColumn(width="medium"),
            "Answer": st.column_config.TextColumn(width="large"),
            "Expected": st.column_config.TextColumn(width="medium"),
            "Reason": st.column_config.TextColumn(width="medium"),
            "Correctness": st.column_config.NumberColumn(format="%.2f"),
            "Relevance": st.column_config.NumberColumn(format="%.2f"),
            "Completeness": st.column_config.NumberColumn(format="%.2f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # Export
    if st.button("Download as CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "eval_results.csv", "text/csv")


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
        type_filter = st.multiselect(
            "Question Type",
            dataset_df["type"].unique(),
            default=list(dataset_df["type"].unique()),
        )
    with col_f2:
        diff_filter = st.multiselect(
            "Difficulty",
            dataset_df["difficulty"].unique(),
            default=list(dataset_df["difficulty"].unique()),
        )
    with col_f3:
        source_filter = st.multiselect(
            "Source",
            dataset_df["source"].unique(),
            default=list(dataset_df["source"].unique()),
        )

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
        fig = px.pie(
            filtered, names="difficulty", title="By Difficulty",
            color_discrete_map={"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"},
        )
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
        format_func=lambda i: (
            f"[{df.iloc[i]['question_type']}/{df.iloc[i]['difficulty']}] "
            f"{questions[i][:100]}"
        ),
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
        st.subheader("Expected Answer")
        st.markdown(row["ground_truth"])

    with col_sys:
        st.subheader(f"System Answer ({row['arbiter_engine']})")
        if row["arbiter_error"]:
            st.error(f"Error: {row['arbiter_error']}")
        else:
            st.markdown(row["arbiter_answer"])

    st.divider()

    # Correctness reason
    reason = correctness_reason(
        row["ground_truth"], row["arbiter_answer"], row["arbiter_correctness"]
    )
    st.info(f"**Correctness reason:** {reason}")

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
            st.metric(
                "Hybrid Correctness",
                f"{row['hybrid_correctness']:.2f}",
                delta=f"{row['hybrid_correctness'] - row['arbiter_correctness']:.2f}",
            )
            st.metric(
                "Hybrid Latency",
                f"{row['hybrid_latency_ms']:.0f}ms",
                delta=f"{row['hybrid_latency_ms'] - row['arbiter_latency_ms']:.0f}ms",
                delta_color="inverse",
            )
