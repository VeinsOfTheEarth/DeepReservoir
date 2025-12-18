# deepreservoir/cli.py
"""
Command-line entry points for DeepReservoir.

This module is responsible only for:
- defining CLI arguments
- parsing them
- calling into model.train(...)
"""

from __future__ import annotations
import argparse

from deepreservoir.drl import model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DeepReservoir agent for Navajo.",
    )

    parser.add_argument(
        "--n-years-test",
        type=int,
        default=10,
        help="Number of (water) years to reserve for testing (most recent).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=400_000,
        help="Total training timesteps for PPO.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to SB3.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo"],
        help="RL algorithm to use (only PPO wired up for now).",
    )
    parser.add_argument(
        "--reward-spec",
        type=str,
        default="dam_safety:storage_band",
        help=(
            "Comma-separated objective:variant pairs, e.g. "
            "'dam_safety:storage_band,esa_min_flow:baseline,hydropower:baseline,niip:baseline'"
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/navajo",
        help="Directory to store models and eval logs.",
    )

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    drlm = model.DRLModel(
        n_years_test=args.n_years_test,
        reward_spec=args.reward_spec,
        algo=args.algo,
        logdir=args.logdir,
        seed=args.seed,
    )
    drlm.train(total_timesteps=args.total_timesteps,)

if __name__ == "__main__":
    main()
