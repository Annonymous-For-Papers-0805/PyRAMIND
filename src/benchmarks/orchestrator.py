"""Benchmark orchestrator CLI.

Single entrypoint that ties baselines × benchmarks × seeds together. Writes
a per-run JSON artifact to <output-dir>/<benchmark>/<system>/<seed>.json
with all per-question records, summary metrics, cost, and the config hash
needed for the reproducibility bundle.

Usage:
    python -m benchmarks.orchestrator \\
        --system pyramind --benchmark lme \\
        --data-path data/longmemeval_s.json \\
        --seed 42 \\
        --output-dir results/

Run with --help for the full flag list. Pass --max-items N to subsample the
benchmark; omit to evaluate on the full split.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


# "naive-rag" is the legacy alias for the DENSE (Azure embedding + cosine)
# RAG baseline. "dense-rag" is the preferred name; both route to the same
# implementation. "bm25" is the sparse Naive-RAG baseline from paper Table 2.
SUPPORTED_SYSTEMS = ("full-context", "naive-rag", "dense-rag", "bm25", "pyramind")
SUPPORTED_BENCHMARKS = ("lme", "lme-oracle", "locomo", "hotpot")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _build_azure_client(args: argparse.Namespace, cache_dir: Path) -> Any:
    """Construct AzureClient from CLI args / env vars."""
    from pyramind_lib import AzureClient

    api_key = args.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "AZURE_OPENAI_API_KEY is required (set env var or pass --azure-api-key)."
        )
    return AzureClient(
        api_key=api_key,
        endpoint=args.azure_endpoint or os.environ.get(
            "AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"
        ),
        api_version=args.azure_api_version,
        deployment=args.chat_deployment,
        embedding_deployment=args.embed_deployment,
        cache_dir=cache_dir,
    )


def _build_baseline(args: argparse.Namespace, azure_client: Any) -> Any:
    """Construct the requested baseline."""
    if args.system == "full-context":
        from benchmarks.baselines.full_context import FullContextBaseline
        return FullContextBaseline(
            azure_client=azure_client, deployment=args.chat_deployment,
        )

    if args.system in ("naive-rag", "dense-rag"):
        # Dense RAG (Azure embedding + cosine). "naive-rag" kept as legacy
        # alias so existing artifact paths under runs/<bench>/naive-rag/...
        # still resolve unchanged.
        from benchmarks.baselines.naive_rag import NaiveRAGBaseline
        return NaiveRAGBaseline(
            azure_client=azure_client, deployment=args.chat_deployment,
            top_k=args.top_k,
        )

    if args.system == "bm25":
        from benchmarks.baselines.bm25 import BM25Baseline
        return BM25Baseline(
            azure_client=azure_client, deployment=args.chat_deployment,
            top_k=args.top_k,
        )

    if args.system == "pyramind":
        from pyramind_lib import PyramindConfig

        from benchmarks.baselines.pyramind import PyramindBaseline
        if args.config_path:
            cfg = PyramindConfig.load(args.config_path)
        else:
            cfg = PyramindConfig(agent_role="benchmark agent", tau=40.0)
        return PyramindBaseline(
            azure_client=azure_client, config=cfg,
            formula_name=args.formula, top_k=args.top_k,
            deployment=args.chat_deployment,
        )

    raise ValueError(f"Unknown system: {args.system}")


def _load_items(args: argparse.Namespace) -> list:
    if args.benchmark in ("lme", "lme-oracle"):
        from benchmarks.datasets.lme_loader import load_lme
        return load_lme(args.data_path, max_items=args.max_items)
    if args.benchmark == "locomo":
        from benchmarks.datasets.locomo_loader import load_locomo
        # paper §4 split convention: dev=first 3 convs (~586 Qs),
        # test=remaining 7 convs (~1400 Qs), all=both.
        return load_locomo(
            args.data_path,
            max_conversations=args.max_items,
            split=getattr(args, "split", "all"),
        )
    if args.benchmark == "hotpot":
        from benchmarks.datasets.hotpot_loader import load_hotpot
        return load_hotpot(args.data_path, max_items=args.max_items)
    raise ValueError(f"Unknown benchmark: {args.benchmark}")


def _run_benchmark(
    args: argparse.Namespace, baseline: Any, items: list, judge_client: Any,
) -> Any:
    if args.benchmark in ("lme", "lme-oracle"):
        from benchmarks.runners.lme import run_lme
        return run_lme(baseline, items, judge_client, seed=args.seed,
                       benchmark_name=args.benchmark)
    if args.benchmark == "locomo":
        from benchmarks.runners.locomo import run_locomo
        return run_locomo(baseline, items, judge_client, seed=args.seed)
    if args.benchmark == "hotpot":
        from benchmarks.runners.hotpot import run_hotpot
        return run_hotpot(baseline, items, judge_client, seed=args.seed)
    raise ValueError(f"Unknown benchmark: {args.benchmark}")


def _config_hash(args: argparse.Namespace) -> str:
    """SHA-256 hash of the run configuration for the reproducibility bundle."""
    payload = {
        "system": args.system,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "top_k": args.top_k,
        "formula": args.formula,
        "chat_deployment": args.chat_deployment,
        "embed_deployment": args.embed_deployment,
        "azure_api_version": args.azure_api_version,
        "max_items": args.max_items,
        "split": getattr(args, "split", "all"),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, or empty string if missing / unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _git_commit() -> str:
    """Best-effort current git commit; empty string if unavailable."""
    try:
        # Search upward for a .git dir so this works from any cwd.
        repo_root = Path(__file__).resolve().parent
        for _ in range(8):
            if (repo_root / ".git").exists():
                break
            if repo_root.parent == repo_root:
                break
            repo_root = repo_root.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return ""


def _prompt_templates_sha256() -> str:
    """SHA-256 over the answerer + judge prompt templates.

    Captures the exact prompt-string surface the run used; if either prompt
    changes, this hash changes and the run is no longer comparable to prior
    runs even at the same ``config_hash``.
    """
    try:
        # Answer prompt — shared across baselines, defined inline in the
        # baseline classes. We hash the literal template by re-importing it
        # from the module that actually emits it.
        from benchmarks.baselines import pyramind as _p
        from benchmarks import judge as _j
        blob = "\n---\n".join([
            getattr(_p, "_SYSTEM", ""),
            getattr(_j, "_JUDGE_SYSTEM", ""),
            getattr(_j, "_JUDGE_PROMPT_TEMPLATE", ""),
        ]).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return ""


def _build_manifest(args: argparse.Namespace, baseline: Any) -> dict:
    """Extended run manifest for paper §4 reproducibility.

    Captures the environment + input fingerprints alongside ``config_hash``
    so an artifact can be replayed against the same data, code, and prompts.
    The fields are best-effort — any one of them missing (e.g. running
    outside git, no requirements.txt next to cwd) degrades to "".
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/benchmarks/.. = repo
    data_path = Path(args.data_path)
    requirements_path = repo_root / "requirements.txt"

    return {
        "dataset_sha256": _sha256_file(data_path) if data_path.exists() else "",
        "git_commit": _git_commit(),
        "requirements_sha256": (
            _sha256_file(requirements_path) if requirements_path.exists() else ""
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "prompt_template_sha256": _prompt_templates_sha256(),
        "split": getattr(args, "split", "all"),
    }


def _serialize_result(
    result: Any, args: argparse.Namespace, elapsed: float, baseline: Any = None,
) -> dict:
    return {
        "system": result.system,
        "benchmark": result.benchmark,
        "seed": result.seed,
        "config_hash": _config_hash(args),
        "manifest": _build_manifest(args, baseline),
        "wall_time_seconds": round(elapsed, 2),
        "summary": {
            "n": len(result.records),
            "accuracy": round(result.accuracy(), 4),
            "avg_f1": round(result.avg_f1(), 4),
            "avg_latency_ms": int(
                sum(r.elapsed_ms for r in result.records) / max(1, len(result.records))
            ),
        },
        "cost": result.cost,
        "records": [asdict(r) if dataclasses.is_dataclass(r) else r for r in result.records],
    }


def _write_artifact(payload: dict, output_dir: Path, args: argparse.Namespace) -> Path:
    out = output_dir / args.benchmark / args.system / f"seed-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarks.orchestrator",
        description="Run a memory baseline over a benchmark and write a per-run artifact.",
    )
    p.add_argument("--system", required=True, choices=SUPPORTED_SYSTEMS)
    p.add_argument("--benchmark", required=True, choices=SUPPORTED_BENCHMARKS)
    p.add_argument("--data-path", required=True, type=Path,
                   help="Path to the benchmark data file (json or jsonl).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-items", type=int, default=None,
                   help="Optional truncation for smoke / dev runs.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Retrieval depth for RAG-style baselines.")
    p.add_argument("--formula", type=str, default="ensemble",
                   help="Decay formula name (only used when system=pyramind).")
    p.add_argument("--config-path", type=Path, default=None,
                   help="Optional PyramindConfig JSON path (defaults to standard config).")
    p.add_argument("--output-dir", type=Path, default=Path("runs/"),
                   help="Where to write the per-run artifact.")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/azure"),
                   help="Azure response cache (shared across runs).")
    p.add_argument("--chat-deployment", type=str, default="gpt-4.1")
    p.add_argument("--embed-deployment", type=str, default="text-embedding-3-large")
    p.add_argument("--azure-api-version", type=str, default="2024-10-21")
    p.add_argument("--azure-endpoint", type=str, default=None)
    p.add_argument("--azure-api-key", type=str, default=None)
    p.add_argument("--split", choices=("all", "dev", "test"), default="all",
                   help="LoCoMo dev/test split (paper §4): "
                        "dev=first 3 convs (~586 Qs), test=remaining 7 convs "
                        "(~1400 Qs). Ignored by other benchmarks.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything but don't execute the benchmark.")
    return p


def run(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    _seed_everything(args.seed)
    print(f"[orchestrator] system={args.system} benchmark={args.benchmark} seed={args.seed}")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    azure_client = _build_azure_client(args, cache_dir)
    baseline = _build_baseline(args, azure_client)
    items = _load_items(args)
    print(f"[orchestrator] loaded {len(items)} items from {args.data_path}")

    if args.dry_run:
        print("[orchestrator] dry-run: skipping execution")
        return 0

    t0 = time.time()
    result = _run_benchmark(args, baseline, items, judge_client=azure_client)
    elapsed = time.time() - t0

    payload = _serialize_result(result, args, elapsed, baseline=baseline)
    out = _write_artifact(payload, Path(args.output_dir), args)

    s = payload["summary"]
    print(
        f"[orchestrator] done in {elapsed:.1f}s — "
        f"acc={s['accuracy']:.3f} f1={s['avg_f1']:.3f} n={s['n']}"
    )
    print(f"[orchestrator] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
