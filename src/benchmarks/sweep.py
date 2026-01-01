"""Sweep CLI — runs the orchestrator over a Cartesian product of configs.

Wraps benchmarks.orchestrator to drive (system x benchmark x seed) sweeps.
Idempotent by default: artifacts at results/<benchmark>/<system>/seed-<N>.json
are skipped on rerun unless --force.

Usage:
    python -m benchmarks.sweep \\
        --systems pyramind \\
        --benchmarks locomo \\
        --seeds 42 \\
        --data-paths locomo=data/locomo10.json \\
        --formula ensemble \\
        --top-k 5 \\
        --output-dir results/

Pass --max-items N to subsample a benchmark; omit to run the full split.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class SweepCell:
    """One (system, benchmark, seed) cell of the sweep."""
    system: str
    benchmark: str
    seed: int
    data_path: Path

    @property
    def artifact_path(self) -> Path:
        return Path(self.benchmark) / self.system / f"seed-{self.seed}.json"


def expand_cells(
    systems: List[str],
    benchmarks: List[str],
    seeds: List[int],
    data_paths: Dict[str, Path],
) -> List[SweepCell]:
    """Cartesian product, skipping benchmarks without a data path."""
    cells: List[SweepCell] = []
    for benchmark in benchmarks:
        if benchmark not in data_paths:
            raise ValueError(
                f"--data-paths is missing an entry for benchmark={benchmark!r}. "
                f"Provide via --data-paths {benchmark}=path/to/file"
            )
        for system in systems:
            for seed in seeds:
                cells.append(SweepCell(
                    system=system, benchmark=benchmark, seed=seed,
                    data_path=data_paths[benchmark],
                ))
    return cells


def parse_data_paths(raw: List[str]) -> Dict[str, Path]:
    """Parse `--data-paths benchmark=path` pairs into a dict."""
    out: Dict[str, Path] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"--data-paths entry must be 'name=path', got {entry!r}")
        name, path = entry.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def run_sweep(
    cells: List[SweepCell],
    output_dir: Path,
    extra_args: List[str],
    force: bool = False,
    runner: Optional[Callable[[List[str]], int]] = None,
) -> Dict[str, int]:
    """Run all cells. Returns {ran, skipped, failed} counts.

    Each cell is dispatched as a fresh orchestrator invocation. Failures don't
    abort the sweep — they're counted and reported at the end so a single bad
    cell doesn't waste the rest.
    """
    if runner is None:
        from benchmarks import orchestrator
        runner = orchestrator.run

    output_dir = Path(output_dir).expanduser().resolve()
    counts = {"ran": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []

    print(f"[sweep] {len(cells)} cells → {output_dir}", flush=True)
    for i, cell in enumerate(cells, start=1):
        artifact = output_dir / cell.artifact_path
        cell_id = f"{cell.benchmark}/{cell.system}/seed-{cell.seed}"

        if artifact.exists() and not force:
            print(f"[sweep] [{i}/{len(cells)}] SKIP {cell_id} (exists)", flush=True)
            counts["skipped"] += 1
            continue

        argv = [
            "--system", cell.system,
            "--benchmark", cell.benchmark,
            "--data-path", str(cell.data_path),
            "--seed", str(cell.seed),
            "--output-dir", str(output_dir),
            *extra_args,
        ]
        t0 = time.time()
        print(f"[sweep] [{i}/{len(cells)}] RUN  {cell_id}", flush=True)
        try:
            rc = runner(argv)
        except SystemExit as e:
            rc = int(getattr(e, "code", 1) or 1)
        except Exception as e:
            rc = 1
            failures.append(f"{cell_id}: {type(e).__name__}: {e}")
        elapsed = time.time() - t0

        if rc == 0:
            counts["ran"] += 1
            print(f"[sweep] [{i}/{len(cells)}] DONE {cell_id} ({elapsed:.1f}s)", flush=True)
        else:
            counts["failed"] += 1
            failures.append(f"{cell_id}: orchestrator exit={rc}")
            print(f"[sweep] [{i}/{len(cells)}] FAIL {cell_id} ({elapsed:.1f}s, rc={rc})",
                  flush=True)

    print(
        f"[sweep] summary: ran={counts['ran']} skipped={counts['skipped']} "
        f"failed={counts['failed']}",
        flush=True,
    )
    if failures:
        print(f"[sweep] failures:")
        for f in failures:
            print(f"  - {f}")
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarks.sweep",
        description="Run the orchestrator over a (systems × benchmarks × seeds) grid.",
    )
    p.add_argument("--systems", nargs="+", required=True,
                   help="Baseline names: full-context, naive-rag (alias dense-rag), "
                        "bm25, pyramind.")
    p.add_argument("--benchmarks", nargs="+", required=True,
                   choices=("lme", "lme-oracle", "locomo", "hotpot"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    p.add_argument("--data-paths", nargs="+", required=True,
                   help="One or more 'benchmark=path' pairs.")
    p.add_argument("--output-dir", type=Path, default=Path("runs/main/"))
    p.add_argument("--force", action="store_true",
                   help="Re-run even if artifact exists.")
    p.add_argument("--max-items", type=int, default=None,
                   help="Truncate each benchmark for smoke runs.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--formula", type=str, default="ensemble")
    p.add_argument("--config-path", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/azure"))
    p.add_argument("--chat-deployment", type=str, default="gpt-4.1")
    p.add_argument("--embed-deployment", type=str, default="text-embedding-3-large")
    p.add_argument("--azure-api-version", type=str, default="2024-10-21")
    p.add_argument("--azure-endpoint", type=str, default=None)
    p.add_argument("--azure-api-key", type=str, default=None)
    p.add_argument("--split", choices=("all", "dev", "test"), default="all",
                   help="LoCoMo dev/test split (paper §4); plumbed to the "
                        "orchestrator. Ignored by other benchmarks.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build cells but don't actually invoke the orchestrator.")
    return p


def _build_extra_args(args: argparse.Namespace) -> List[str]:
    """Pass-through args that go straight to every orchestrator invocation."""
    extra: List[str] = []
    if args.max_items is not None:
        extra += ["--max-items", str(args.max_items)]
    extra += ["--top-k", str(args.top_k), "--formula", args.formula,
              "--cache-dir", str(args.cache_dir),
              "--chat-deployment", args.chat_deployment,
              "--embed-deployment", args.embed_deployment,
              "--azure-api-version", args.azure_api_version,
              "--split", args.split]
    if args.config_path:
        extra += ["--config-path", str(args.config_path)]
    if args.azure_endpoint:
        extra += ["--azure-endpoint", args.azure_endpoint]
    if args.azure_api_key:
        extra += ["--azure-api-key", args.azure_api_key]
    if args.dry_run:
        extra += ["--dry-run"]
    return extra


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_paths = parse_data_paths(args.data_paths)
    cells = expand_cells(args.systems, args.benchmarks, args.seeds, data_paths)
    counts = run_sweep(
        cells, output_dir=args.output_dir,
        extra_args=_build_extra_args(args), force=args.force,
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
