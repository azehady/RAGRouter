"""
Build RAG evaluation dataset from Ciroos documentation.

Reads source markdown files, chunks them (matching the Qdrant index),
then generates Q&A pairs using an LLM for ground-truth evaluation.

Usage:
    # Generate chunks + metadata (no LLM needed)
    python eval/build_dataset.py chunks

    # Generate Q&A pairs from chunks (requires LiteLLM proxy)
    python eval/build_dataset.py generate --litellm-url http://localhost:4000

    # Generate from inside the cluster
    kubectl exec -n ciroos-rag deploy/arbiter -- python eval/build_dataset.py generate

    # Stats only
    python eval/build_dataset.py stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
SOURCE_DOCS_DIR = DATA_DIR / "source_docs"
CHUNKS_FILE = DATA_DIR / "chunks_with_text.jsonl"
DATASET_FILE = DATA_DIR / "eval_dataset.jsonl"
DATASET_SUMMARY_FILE = DATA_DIR / "eval_dataset_summary.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    title: str
    section: str
    subsection: str
    heading_hierarchy: list[str]
    doc_type: str
    chunk_index: int
    total_chunks: int
    word_count: int
    text: str


@dataclass
class QAPair:
    question: str
    answer: str
    chunk_ids: list[str]
    question_type: str  # factual, procedural, conceptual, multi-hop, comparison
    difficulty: str  # easy, medium, hard
    source_files: list[str]
    doc_types: list[str]


@dataclass
class EvalRecord:
    id: str
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str]
    chunk_ids: list[str]
    question_type: str
    difficulty: str
    source_files: list[str]
    doc_types: list[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunking (matches the ingest pipeline)
# ---------------------------------------------------------------------------

DOC_TYPE_MAP = {
    "api": ["api/"],
    "onboarding": ["onboarding", "CiroosOnboarding", "CiroosCloudOnboarding"],
    "integration": ["integration", "dynatrace", "webhook"],
    "feature": ["feature/", "alerts", "askciroos", "collab", "ubp", "usermgmt"],
    "domain": ["domain/"],
    "general": [],
}


def classify_doc_type(filepath: str) -> str:
    for doc_type, patterns in DOC_TYPE_MAP.items():
        for pattern in patterns:
            if pattern.lower() in filepath.lower():
                return doc_type
    return "general"


def extract_heading_hierarchy(text: str, up_to_line: int) -> list[str]:
    """Extract heading hierarchy from markdown up to a given line."""
    hierarchy = []
    current_level = 0
    for i, line in enumerate(text.split("\n")):
        if i >= up_to_line:
            break
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level <= current_level:
                # Pop back to parent level
                hierarchy = hierarchy[: level - 1]
            hierarchy.append(heading)
            current_level = level
    return hierarchy


def chunk_markdown(filepath: Path, max_words: int = 300, overlap_words: int = 30) -> list[Chunk]:
    """Chunk a markdown file by sections, respecting heading boundaries."""
    text = filepath.read_text(encoding="utf-8")
    if not text.strip():
        return []

    rel_path = str(filepath.relative_to(SOURCE_DOCS_DIR.parent.parent))
    # Normalize to match Qdrant paths: docs/ciroos-docs/docs/...
    if not rel_path.startswith("docs/"):
        rel_path = f"docs/{rel_path}"

    doc_type = classify_doc_type(rel_path)
    lines = text.split("\n")

    # Split into sections by headings
    sections: list[dict] = []
    current_section: dict = {"heading": "", "level": 0, "lines": [], "start_line": 0}

    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            # Save previous section
            if current_section["lines"] or current_section["heading"]:
                sections.append(current_section)
            current_section = {
                "heading": match.group(2).strip(),
                "level": len(match.group(1)),
                "lines": [],
                "start_line": i,
            }
        else:
            current_section["lines"].append(line)

    if current_section["lines"] or current_section["heading"]:
        sections.append(current_section)

    # Get document title
    title = sections[0]["heading"] if sections and sections[0]["heading"] else filepath.stem

    # Build chunks from sections
    chunks: list[Chunk] = []
    current_text: list[str] = []
    current_word_count = 0
    current_section_name = ""
    current_subsection = ""

    def flush_chunk():
        nonlocal current_text, current_word_count
        if not current_text:
            return
        chunk_text = "\n".join(current_text).strip()
        if not chunk_text or len(chunk_text.split()) < 10:
            return

        file_hash = hashlib.md5(rel_path.encode()).hexdigest()[:8]
        chunk_idx = len(chunks)
        chunk_id = f"{file_hash}_{chunk_idx:04d}"

        hierarchy = [title]
        if current_section_name and current_section_name != title:
            hierarchy.append(current_section_name)
        if current_subsection:
            hierarchy.append(current_subsection)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=rel_path,
                title=title,
                section=current_section_name or title,
                subsection=current_subsection,
                heading_hierarchy=hierarchy,
                doc_type=doc_type,
                chunk_index=chunk_idx,
                total_chunks=0,  # filled in later
                word_count=len(chunk_text.split()),
                text=chunk_text,
            )
        )
        current_text = []
        current_word_count = 0

    for section in sections:
        section_text = "\n".join(section["lines"]).strip()
        section_words = len(section_text.split()) if section_text else 0

        # Track section/subsection names
        if section["level"] <= 2:
            current_section_name = section["heading"]
            current_subsection = ""
        elif section["level"] == 3:
            current_subsection = section["heading"]

        # Add heading to chunk
        if section["heading"]:
            heading_line = f"{'#' * section['level']} {section['heading']}"
            current_text.append(heading_line)
            current_word_count += len(section["heading"].split())

        if current_word_count + section_words > max_words and current_word_count > 0:
            # Flush current chunk, start new one with this section
            flush_chunk()
            if section["heading"]:
                current_text.append(f"{'#' * section['level']} {section['heading']}")

        # Add section content, splitting long sections
        for line in section["lines"]:
            line_words = len(line.split())
            if current_word_count + line_words > max_words and current_word_count > 50:
                flush_chunk()
                # Carry heading context forward
                if current_section_name:
                    current_text.append(f"[continued] {current_section_name}")
            current_text.append(line)
            current_word_count += line_words

    flush_chunk()

    # Fill in total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Q&A generation prompts
# ---------------------------------------------------------------------------

GENERATE_QA_SYSTEM = """You are an expert at creating evaluation datasets for RAG (Retrieval-Augmented Generation) systems.

Given documentation chunks, generate diverse question-answer pairs that test different retrieval and reasoning capabilities.

For each chunk or group of related chunks, generate questions of these types:
- **factual**: Direct fact lookup ("What port does the arbiter run on?")
- **procedural**: How-to steps ("How do you onboard an AWS account?")
- **conceptual**: Understanding ("What is the purpose of the investigation agent?")
- **comparison**: Contrasting concepts ("What's the difference between alerts and investigations?")
- **multi-hop**: Requires combining info from multiple sections ("If I'm troubleshooting a K8s pod failure, which dashboard and which agent would I use?")

Rules:
1. Questions must be answerable from the provided context
2. Answers must be grounded in the text — include specific details, not vague summaries
3. Vary difficulty: easy (single fact), medium (requires understanding), hard (multi-step reasoning)
4. Make questions natural — how a real user would ask, not "According to the document..."
5. Include 1-3 sentence answers with key specifics
6. For multi-hop questions, reference which chunks contain the relevant info

Return JSON array of objects with these fields:
- question: the question text
- answer: ground truth answer (1-3 sentences, specific)
- question_type: factual|procedural|conceptual|comparison|multi-hop
- difficulty: easy|medium|hard
- relevant_chunk_indices: array of 0-based indices into the provided chunks
"""

GENERATE_QA_USER = """Here are {n_chunks} documentation chunks from the Ciroos platform docs.
Generate {n_questions} diverse Q&A pairs covering different question types and difficulties.

{chunks_text}

Return ONLY a JSON array, no other text."""


# ---------------------------------------------------------------------------
# Q&A generation with LLM
# ---------------------------------------------------------------------------


def generate_qa_pairs(
    chunks: list[Chunk],
    litellm_url: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 8,
    questions_per_batch: int = 6,
) -> list[EvalRecord]:
    """Generate Q&A pairs from chunks using LLM."""
    import httpx

    records: list[EvalRecord] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(chunks), batch_size):
        batch = chunks[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Format chunks for prompt
        chunks_text = ""
        for i, chunk in enumerate(batch):
            chunks_text += f"\n--- Chunk {i} [{chunk.doc_type}] {chunk.source_file} ---\n"
            chunks_text += f"Title: {chunk.title}\n"
            chunks_text += f"Section: {chunk.section}\n"
            chunks_text += f"Text:\n{chunk.text}\n"

        prompt = GENERATE_QA_USER.format(
            n_chunks=len(batch),
            n_questions=questions_per_batch,
            chunks_text=chunks_text,
        )

        print(f"  Batch {batch_num}/{total_batches}: generating {questions_per_batch} Q&A pairs...", flush=True)

        try:
            resp = httpx.post(
                f"{litellm_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": GENERATE_QA_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 3000,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON from response (handle ```json blocks)
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            qa_pairs = json.loads(content)

            for qa in qa_pairs:
                chunk_indices = qa.get("relevant_chunk_indices", [0])
                relevant_chunks = [batch[i] for i in chunk_indices if i < len(batch)]
                if not relevant_chunks:
                    relevant_chunks = [batch[0]]

                record_id = hashlib.md5(qa["question"].encode()).hexdigest()[:12]
                records.append(
                    EvalRecord(
                        id=f"eval_{record_id}",
                        question=qa["question"],
                        ground_truth_answer=qa["answer"],
                        ground_truth_contexts=[c.text for c in relevant_chunks],
                        chunk_ids=[c.chunk_id for c in relevant_chunks],
                        question_type=qa.get("question_type", "factual"),
                        difficulty=qa.get("difficulty", "medium"),
                        source_files=list({c.source_file for c in relevant_chunks}),
                        doc_types=list({c.doc_type for c in relevant_chunks}),
                    )
                )

        except Exception as e:
            print(f"  WARNING: Batch {batch_num} failed: {e}", file=sys.stderr)
            continue

        # Rate limit
        time.sleep(0.5)

    return records


# ---------------------------------------------------------------------------
# Manual seed questions (always included, no LLM needed)
# ---------------------------------------------------------------------------

SEED_QUESTIONS: list[dict] = [
    # Factual - easy
    {
        "question": "What port does the Ciroos arbiter service run on?",
        "answer": "The arbiter service runs on port 8000.",
        "question_type": "factual",
        "difficulty": "easy",
        "doc_types": ["api"],
    },
    {
        "question": "What cloud providers does Ciroos support for onboarding?",
        "answer": "Ciroos supports AWS, GCP, and Azure for cloud onboarding.",
        "question_type": "factual",
        "difficulty": "easy",
        "doc_types": ["onboarding"],
    },
    # Procedural - medium
    {
        "question": "How do I onboard an AWS account to Ciroos?",
        "answer": "To onboard an AWS account, navigate to Settings > Integrations > Cloud Accounts, click Add Account, select AWS, provide the account ID and IAM role ARN with the required permissions, and complete the connection wizard.",
        "question_type": "procedural",
        "difficulty": "medium",
        "doc_types": ["onboarding"],
    },
    {
        "question": "How do I onboard a Kubernetes cluster to Ciroos?",
        "answer": "To onboard a Kubernetes cluster, go to Settings > Integrations > Kubernetes, click Add Cluster, install the Ciroos agent using the provided Helm chart command, and verify the cluster appears in the Kubernetes dashboard.",
        "question_type": "procedural",
        "difficulty": "medium",
        "doc_types": ["onboarding"],
    },
    {
        "question": "How do I set up Dynatrace integration with Ciroos?",
        "answer": "To set up Dynatrace integration, navigate to Settings > Integrations > Observability Tools, select Dynatrace, provide the Dynatrace environment URL and API token with the required scopes, and test the connection.",
        "question_type": "procedural",
        "difficulty": "medium",
        "doc_types": ["integration"],
    },
    # Conceptual - medium
    {
        "question": "What is Ask Ciroos and what does it do?",
        "answer": "Ask Ciroos is a conversational AI assistant that allows users to ask questions about their infrastructure, investigations, and platform features using natural language. It routes queries to specialized agents that can query Kubernetes clusters, cloud accounts, and observability tools.",
        "question_type": "conceptual",
        "difficulty": "medium",
        "doc_types": ["feature"],
    },
    {
        "question": "What is the difference between an investigation and an alert in Ciroos?",
        "answer": "An alert is a notification triggered by a monitoring rule or threshold breach, while an investigation is a deeper analysis process that can be triggered by alerts or manually initiated. Investigations use AI-powered root cause analysis to analyze infrastructure state, correlate events, and produce detailed findings.",
        "question_type": "comparison",
        "difficulty": "medium",
        "doc_types": ["feature"],
    },
    {
        "question": "What is the purpose of the audit log feature?",
        "answer": "The audit log feature tracks and records all user actions and system events within the Ciroos platform, providing an audit trail for compliance, troubleshooting, and security review purposes.",
        "question_type": "conceptual",
        "difficulty": "easy",
        "doc_types": ["feature"],
    },
    # Hard / multi-hop
    {
        "question": "If I see a pod crash-looping in my Kubernetes cluster, which Ciroos features would help me investigate and what steps would I take?",
        "answer": "First, check the Kubernetes Dashboard for the cluster overview and pod status. Then either start an investigation manually or let an alert trigger one automatically. The investigation will use AI-powered root cause analysis with the K8s agent to examine pod events, logs, and resource metrics. You can also use Ask Ciroos to query the cluster directly. The investigation details pane will show the AI's thinking process and findings.",
        "question_type": "multi-hop",
        "difficulty": "hard",
        "doc_types": ["feature", "domain"],
    },
    {
        "question": "What API endpoints are available for automating cloud onboarding, and what authentication is required?",
        "answer": "Ciroos provides REST API endpoints for automating cloud account onboarding, Kubernetes cluster onboarding, and observability tool onboarding. All API calls require authentication via an API key or bearer token passed in the Authorization header, along with the x-organization-id header for tenant context.",
        "question_type": "multi-hop",
        "difficulty": "hard",
        "doc_types": ["api"],
    },
    {
        "question": "How does Ciroos's AI-powered incident analysis work end-to-end, from alert to resolution?",
        "answer": "When an alert fires, Ciroos can automatically trigger an investigation. The investigation orchestrator dispatches specialized agents (K8s, cloud, observability) to gather relevant data. Each agent uses MCP tools to query infrastructure. The findings are correlated and analyzed by the RCA engine, which produces a root cause summary with confidence levels and recommended remediation steps. The full analysis is available in the Investigation Details view.",
        "question_type": "conceptual",
        "difficulty": "hard",
        "doc_types": ["feature"],
    },
    # Comparison
    {
        "question": "What is the difference between the Kubernetes dashboard and the investigation details view?",
        "answer": "The Kubernetes Dashboard provides a real-time operational view of cluster health, pod statuses, resource utilization, and node metrics. The Investigation Details view shows the AI-driven root cause analysis process, including the agent's reasoning steps, collected evidence, correlated findings, and the final RCA summary with confidence scores.",
        "question_type": "comparison",
        "difficulty": "medium",
        "doc_types": ["feature", "domain"],
    },
    {
        "question": "How does webhook integration differ from the Slack and Teams integrations?",
        "answer": "Webhooks provide a generic HTTP callback mechanism for sending notifications to any endpoint, while Slack and Teams integrations are purpose-built connectors that support bidirectional communication — they can send notifications and receive commands. The messaging integrations also support Ask Ciroos interactive conversations.",
        "question_type": "comparison",
        "difficulty": "medium",
        "doc_types": ["integration"],
    },
    # User behavior
    {
        "question": "What are User Behavior Patterns in Ciroos?",
        "answer": "User Behavior Patterns (UBP) is a feature that analyzes how users interact with the Ciroos platform to identify usage patterns, optimize workflows, and provide recommendations for improving operational efficiency.",
        "question_type": "conceptual",
        "difficulty": "medium",
        "doc_types": ["feature"],
    },
    {
        "question": "How do I manage users and roles in Ciroos?",
        "answer": "User management is done through Settings > User Management. You can invite users, assign roles (admin, member, viewer), and manage permissions. Role-based access control (RBAC) determines what each user can view and modify across the platform.",
        "question_type": "procedural",
        "difficulty": "easy",
        "doc_types": ["feature"],
    },
]


def build_seed_records() -> list[EvalRecord]:
    """Build eval records from seed questions."""
    records = []
    for i, q in enumerate(SEED_QUESTIONS):
        record_id = hashlib.md5(q["question"].encode()).hexdigest()[:12]
        records.append(
            EvalRecord(
                id=f"seed_{record_id}",
                question=q["question"],
                ground_truth_answer=q["answer"],
                ground_truth_contexts=[],  # will be filled by retrieval
                chunk_ids=[],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                source_files=[],
                doc_types=q["doc_types"],
                metadata={"source": "seed"},
            )
        )
    return records


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_chunks(args: argparse.Namespace) -> None:
    """Extract and save all chunks with text from source docs."""
    if not SOURCE_DOCS_DIR.exists():
        print(f"ERROR: Source docs not found at {SOURCE_DOCS_DIR}", file=sys.stderr)
        print("Copy docs first: cp -r /path/to/docs/ciroos-docs/docs/ eval/data/source_docs/")
        sys.exit(1)

    md_files = sorted(SOURCE_DOCS_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {SOURCE_DOCS_DIR}")

    all_chunks: list[Chunk] = []
    for md_file in md_files:
        chunks = chunk_markdown(md_file)
        all_chunks.extend(chunks)
        print(f"  {md_file.relative_to(SOURCE_DOCS_DIR)}: {len(chunks)} chunks")

    # Write JSONL
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), default=str) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks written to {CHUNKS_FILE}")

    # Stats by doc_type
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
    print("\nBy doc_type:")
    for dt, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate full eval dataset (seed + LLM-generated Q&A)."""
    # Load chunks
    if not CHUNKS_FILE.exists():
        print("Chunks file not found. Running chunking first...")
        cmd_chunks(args)

    chunks: list[Chunk] = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))

    print(f"Loaded {len(chunks)} chunks")

    # Seed questions (always included)
    records = build_seed_records()
    print(f"Added {len(records)} seed questions")

    # LLM-generated questions
    litellm_url = args.litellm_url
    if litellm_url:
        print(f"Generating Q&A pairs via {litellm_url} using {args.model}...")
        llm_records = generate_qa_pairs(
            chunks=chunks,
            litellm_url=litellm_url,
            model=args.model,
            batch_size=args.batch_size,
            questions_per_batch=args.questions_per_batch,
        )
        records.extend(llm_records)
        print(f"Generated {len(llm_records)} LLM Q&A pairs")
    else:
        print("No --litellm-url provided, skipping LLM generation (seed questions only)")

    # Write dataset
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    # Write summary
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_doc_type: dict[str, int] = {}
    for r in records:
        by_type[r.question_type] = by_type.get(r.question_type, 0) + 1
        by_difficulty[r.difficulty] = by_difficulty.get(r.difficulty, 0) + 1
        for dt in r.doc_types:
            by_doc_type[dt] = by_doc_type.get(dt, 0) + 1

    summary = {
        "total_questions": len(records),
        "seed_questions": len([r for r in records if r.metadata.get("source") == "seed"]),
        "generated_questions": len([r for r in records if r.metadata.get("source") != "seed"]),
        "by_question_type": by_type,
        "by_difficulty": by_difficulty,
        "by_doc_type": by_doc_type,
        "source_chunks": len(chunks),
        "source_files": len(set(c.source_file for c in chunks)),
    }

    with open(DATASET_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset: {len(records)} questions written to {DATASET_FILE}")
    print(f"Summary: {DATASET_SUMMARY_FILE}")
    print(json.dumps(summary, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    """Show dataset statistics."""
    if DATASET_SUMMARY_FILE.exists():
        summary = json.loads(DATASET_SUMMARY_FILE.read_text())
        print(json.dumps(summary, indent=2))
    else:
        print("No dataset summary found. Run 'generate' first.")

    if CHUNKS_FILE.exists():
        count = sum(1 for _ in open(CHUNKS_FILE))
        print(f"\nChunks file: {count} chunks")

    if DATASET_FILE.exists():
        count = sum(1 for _ in open(DATASET_FILE))
        print(f"Dataset file: {count} questions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG evaluation dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("chunks", help="Extract chunks with text from source docs")

    gen = sub.add_parser("generate", help="Generate eval dataset (seed + LLM Q&A)")
    gen.add_argument("--litellm-url", default=None, help="LiteLLM proxy URL")
    gen.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    gen.add_argument("--batch-size", type=int, default=8, help="Chunks per LLM batch")
    gen.add_argument("--questions-per-batch", type=int, default=6, help="Questions per batch")

    sub.add_parser("stats", help="Show dataset statistics")

    args = parser.parse_args()

    if args.command == "chunks":
        cmd_chunks(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
