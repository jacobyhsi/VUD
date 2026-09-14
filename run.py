#!/usr/bin/env python3
"""Run a VUD experiment.

Examples:
  python run.py toy-classification --dataset_name logistic_regression
  python run.py toy-regression --dataset_name gaps --va_method parallel_pp
  python run.py qa --id boolqa --ood hotpotqa
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "toy-classification": "src.experiments.toy_classification",
    "toy-regression": "src.experiments.toy_regression",
    "qa": "src.experiments.qa",
    "qa-abstention": "src.experiments.qa_abstention",
    "bandit-classification": "src.experiments.bandit_classification",
    "bandit-classification-benchmark": "src.experiments.bandit_classification_benchmark",
    "ood-benchmark": "src.experiments.ood_benchmark",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run a VUD experiment.",
    )
    parser.add_argument(
        "command",
        choices=list(COMMANDS),
        help="Experiment to run. Pass -h after the command for that experiment's options.",
    )
    return parser


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        parser.exit(0 if argv else 2)
    command, *rest = argv
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}")
    module = importlib.import_module(COMMANDS[command])
    _set_parser_prog(module, f"run.py {command}")
    module.main(rest)


def _set_parser_prog(module, prog: str) -> None:
    parser = getattr(module, "parser", None)
    if isinstance(parser, argparse.ArgumentParser):
        parser.prog = prog
    original_build = getattr(module, "build_parser", None)
    if not callable(original_build):
        return

    def build_parser_with_prog(*args, **kwargs):
        built = original_build(*args, **kwargs)
        if isinstance(built, argparse.ArgumentParser):
            built.prog = prog
        return built

    module.build_parser = build_parser_with_prog


if __name__ == "__main__":
    main()
