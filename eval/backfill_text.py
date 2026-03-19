"""
Backfill text content into Qdrant ciroos-docs collection.

The original ingestion stored metadata but not the chunk text. This script
reads source markdown files, splits each into the exact number of chunks that
Qdrant has for that file (preserving chunk_index alignment), and upserts the
text field into each point's payload.

Usage:
    # From inside the cluster
    python eval/backfill_text.py \
        --qdrant-url http://qdrant.ciroos.svc.cluster.local:6333 \
        --source-dir /path/to/source_docs

    # Dry run
    python eval/backfill_text.py --dry-run \
        --qdrant-url http://qdrant.ciroos.svc.cluster.local:6333 \
        --source-dir /path/to/source_docs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx


def split_markdown_into_n_chunks(text: str, n: int) -> list[str]:
    """Split markdown text into exactly n chunks, breaking at section boundaries.

    Strategy:
    1. Split by headings into sections
    2. If more sections than n, merge small adjacent sections
    3. If fewer sections than n, split largest sections by paragraphs
    """
    if n <= 0:
        return []
    if n == 1:
        return [text.strip()]

    # Split into sections by headings
    sections: list[str] = []
    current_lines: list[str] = []

    for line in text.split("\n"):
        if re.match(r"^#{1,6}\s+", line) and current_lines:
            sections.append("\n".join(current_lines).strip())
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append("\n".join(current_lines).strip())

    # Remove empty sections
    sections = [s for s in sections if s.strip()]

    if not sections:
        return [text.strip()] + [""] * (n - 1)

    # If we have exactly n, done
    if len(sections) == n:
        return sections

    # If we have more sections than n, merge smallest adjacent pairs
    while len(sections) > n:
        # Find the smallest adjacent pair
        min_size = float("inf")
        min_idx = 0
        for i in range(len(sections) - 1):
            combined = len(sections[i].split()) + len(sections[i + 1].split())
            if combined < min_size:
                min_size = combined
                min_idx = i
        sections[min_idx] = sections[min_idx] + "\n\n" + sections[min_idx + 1]
        sections.pop(min_idx + 1)

    # If we have fewer sections than n, split the largest by paragraphs
    while len(sections) < n:
        # Find largest section
        max_idx = max(range(len(sections)), key=lambda i: len(sections[i].split()))
        section = sections[max_idx]

        # Split by double newline (paragraphs)
        paragraphs = re.split(r"\n\n+", section)
        if len(paragraphs) < 2:
            # Split by single newline as fallback
            lines = section.split("\n")
            mid = len(lines) // 2
            first = "\n".join(lines[:mid]).strip()
            second = "\n".join(lines[mid:]).strip()
            if first and second:
                sections[max_idx] = first
                sections.insert(max_idx + 1, second)
            else:
                break  # Can't split further
        else:
            mid = len(paragraphs) // 2
            first = "\n\n".join(paragraphs[:mid]).strip()
            second = "\n\n".join(paragraphs[mid:]).strip()
            sections[max_idx] = first
            sections.insert(max_idx + 1, second)

    return sections[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill text into Qdrant ciroos-docs")
    parser.add_argument("--qdrant-url", required=True, help="Qdrant REST URL")
    parser.add_argument("--source-dir", required=True, help="Path to source markdown docs root")
    parser.add_argument("--collection", default="ciroos-docs", help="Qdrant collection name")
    parser.add_argument("--dry-run", action="store_true", help="Print matches without writing")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    qdrant_url = args.qdrant_url.rstrip("/")
    collection = args.collection

    # 1. Read all points from Qdrant
    print(f"Reading points from {qdrant_url}/collections/{collection}...")
    all_points = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        resp = httpx.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json=body,
            timeout=30,
        )
        result = resp.json()["result"]
        if not result["points"]:
            break
        all_points.extend(result["points"])
        offset = result.get("next_page_offset")
        if not offset:
            break

    print(f"  {len(all_points)} points")

    # 2. Group by source_file
    by_file: dict[str, list[dict]] = {}
    for pt in all_points:
        sf = pt["payload"].get("source_file", "")
        by_file.setdefault(sf, []).append(pt)

    # Sort each file's points by chunk_index
    for sf in by_file:
        by_file[sf].sort(key=lambda p: p["payload"].get("chunk_index", 0))

    print(f"  {len(by_file)} source files")

    # 3. Process each file
    total_updated = 0
    total_missing = 0

    for sf, points in sorted(by_file.items()):
        n_chunks = len(points)

        # Resolve source file path
        # sf looks like "docs/ciroos-docs/docs/architecture.md"
        # source_dir might be the docs root, try various resolutions
        file_path = None
        candidates = [
            source_dir / sf,
            source_dir / sf.removeprefix("docs/"),
            source_dir / sf.removeprefix("docs/ciroos-docs/docs/"),
            source_dir / sf.removeprefix("docs/ciroos-docs/"),
        ]
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                break

        if not file_path:
            print(f"  SKIP {sf}: source file not found (tried {len(candidates)} paths)")
            total_missing += n_chunks
            continue

        # Read and split into n chunks
        text = file_path.read_text(encoding="utf-8")
        chunks = split_markdown_into_n_chunks(text, n_chunks)

        if len(chunks) != n_chunks:
            print(f"  WARN {sf}: got {len(chunks)} chunks, expected {n_chunks}")
            # Pad or truncate
            while len(chunks) < n_chunks:
                chunks.append("")
            chunks = chunks[:n_chunks]

        # 4. Upsert text into each point
        for pt, chunk_text in zip(points, chunks):
            point_id = pt["id"]
            ci = pt["payload"].get("chunk_index", "?")

            if not chunk_text.strip():
                continue

            if args.dry_run:
                preview = chunk_text[:100].replace("\n", " ")
                print(f"  [{sf}][{ci}] {point_id}: {preview}...")
                total_updated += 1
                continue

            resp = httpx.post(
                f"{qdrant_url}/collections/{collection}/points/payload",
                json={
                    "points": [point_id],
                    "payload": {"text": chunk_text.strip()},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                total_updated += 1
            else:
                print(f"  ERROR {point_id}: {resp.status_code} {resp.text[:100]}")

        print(f"  {sf}: {n_chunks} chunks {'(dry-run)' if args.dry_run else 'updated'}")

    print(f"\nTotal: {total_updated} points updated, {total_missing} skipped (file not found)")

    if not args.dry_run and total_updated > 0:
        # Verify
        resp = httpx.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json={"limit": 1, "with_payload": True, "with_vector": False},
            timeout=10,
        )
        pt = resp.json()["result"]["points"][0]
        text_val = pt["payload"].get("text", "")
        print(f"\nVerification: text={'yes' if text_val else 'no'}")
        if text_val:
            print(f"  Preview: {text_val[:150]}...")


if __name__ == "__main__":
    main()
