"""Run the current assignment experiments from the repository root.

Examples:
    python src/assignment2_runner.py --part b
    python src/assignment2_runner.py --out outputs/results

Task completion and known experiment corrections are tracked in README.md.
"""

import argparse
from pathlib import Path

import numpy as np

from part_a import load_all, part_a, prepare, splits
from part_b import part_b
from part_c import part_c, library_comparisons as logistic_comparisons
from part_d import part_d, task_22, library_comparisons as softmax_comparisons
from part_e import part_e, task_25, task_28, library_comparisons as svm_comparisons
from plot_utils import savefig


def task22_and_25(prepared, output_dir):
    """Keep the original combined call available for existing callers."""
    rng = np.random.default_rng(42)
    task_22(prepared, output_dir, rng=rng)
    task_25(prepared, output_dir, rng=rng)


def off_the_shelf(prepared, output_dir):
    """Run the library comparisons in their original order."""
    logistic_comparisons(prepared, output_dir)
    svm_comparisons(prepared, output_dir)
    softmax_comparisons(prepared, output_dir)
    task_28(prepared, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Assignment 2 experiments.")
    parser.add_argument(
        "--out", default="outputs/results", help="Directory for figures."
    )
    parser.add_argument(
        "--part",
        choices=["all", "a", "b", "c", "d", "e"],
        default="all",
        help="Run one Part independently, or run all currently implemented experiments.",
    )
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare(load_all())

    if args.part == "all":
        part_a(prepared, output_dir)
        part_b(prepared["diabetes"], output_dir)
        part_c(prepared["breast_cancer"], output_dir)
        part_d(prepared["digits"], output_dir)
        part_e(prepared["breast_cancer"], output_dir)
        task22_and_25(prepared, output_dir)
        off_the_shelf(prepared, output_dir)
    elif args.part == "a":
        part_a(prepared, output_dir)
    elif args.part == "b":
        part_b(prepared["diabetes"], output_dir)
    elif args.part == "c":
        part_c(prepared["breast_cancer"], output_dir)
        logistic_comparisons(prepared, output_dir)
    elif args.part == "d":
        part_d(prepared["digits"], output_dir)
        task_22(prepared, output_dir)
        softmax_comparisons(prepared, output_dir)
    elif args.part == "e":
        part_e(prepared["breast_cancer"], output_dir)
        task_25(prepared, output_dir)
        svm_comparisons(prepared, output_dir)
        task_28(prepared, output_dir)


if __name__ == "__main__":
    main()
