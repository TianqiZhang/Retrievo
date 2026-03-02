#!/usr/bin/env python3
"""
Generate embeddings for BEIR NFCorpus dataset and save in HybridSearch binary cache format.

Produces a .bin file compatible with HybridSearch.Benchmarks EmbeddingCache.

Usage:
    pip install sentence-transformers
    python tools/generate_embeddings.py --data-dir benchmarks/data/nfcorpus --output embeddings.bin

The binary format matches C# BinaryWriter conventions:
    [count:int32]
    For each entry:
        [id: 7-bit-encoded-length-prefixed UTF-8 string]
        [dim:int32]
        [floats:float32_le * dim]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path


def write_7bit_encoded_int(f, value: int) -> None:
    """Write a non-negative integer in .NET 7-bit encoded format."""
    while value >= 0x80:
        f.write(struct.pack("B", (value & 0x7F) | 0x80))
        value >>= 7
    f.write(struct.pack("B", value & 0x7F))


def write_dotnet_string(f, s: str) -> None:
    """Write a string in C# BinaryWriter.Write(string) format: 7-bit length prefix + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    write_7bit_encoded_int(f, len(encoded))
    f.write(encoded)


def load_corpus(path: Path) -> dict[str, tuple[str, str]]:
    """Load BEIR corpus.jsonl → {id: (title, text)}."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            corpus[obj["_id"]] = (obj.get("title", ""), obj.get("text", ""))
    print(f"Loaded {len(corpus)} corpus documents", file=sys.stderr)
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Load BEIR queries.jsonl → {id: text}."""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[obj["_id"]] = obj.get("text", "")
    print(f"Loaded {len(queries)} queries", file=sys.stderr)
    return queries


def save_embedding_cache(
    path: Path, embeddings: dict[str, list[float]]
) -> None:
    """Save embeddings in HybridSearch EmbeddingCache binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(embeddings)))
        for doc_id, vector in embeddings.items():
            write_dotnet_string(f, doc_id)
            f.write(struct.pack("<i", len(vector)))
            f.write(struct.pack(f"<{len(vector)}f", *vector))
    print(f"Saved {len(embeddings)} embeddings to {path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for BEIR NFCorpus in HybridSearch binary format."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to extracted NFCorpus directory (containing corpus.jsonl, queries.jsonl).",
    )
    parser.add_argument(
        "--output",
        default="embeddings.bin",
        help="Output binary cache path (default: embeddings.bin).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2, 384 dims).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        print(
            f"Error: expected corpus.jsonl and queries.jsonl in {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here so --help works without sentence-transformers installed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Error: sentence-transformers not installed.\n"
            "  pip install sentence-transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: {dim} dimensions", file=sys.stderr)

    # Embed corpus: title + " " + text
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        f"{title} {text}".strip() for title, text in corpus.values()
    ]

    print(f"Encoding {len(corpus_texts)} corpus documents...", file=sys.stderr)
    t0 = time.time()
    corpus_vectors = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t1 = time.time()
    print(f"Corpus encoded in {t1 - t0:.1f}s", file=sys.stderr)

    # Embed queries
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    print(f"Encoding {len(query_texts)} queries...", file=sys.stderr)
    t0 = time.time()
    query_vectors = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t1 = time.time()
    print(f"Queries encoded in {t1 - t0:.1f}s", file=sys.stderr)

    # Merge into single dict
    all_embeddings: dict[str, list[float]] = {}
    for doc_id, vec in zip(corpus_ids, corpus_vectors):
        all_embeddings[doc_id] = vec.tolist()
    for query_id, vec in zip(query_ids, query_vectors):
        all_embeddings[query_id] = vec.tolist()

    print(
        f"Total: {len(all_embeddings)} embeddings "
        f"({len(corpus_ids)} docs + {len(query_ids)} queries)",
        file=sys.stderr,
    )

    save_embedding_cache(Path(args.output), all_embeddings)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
