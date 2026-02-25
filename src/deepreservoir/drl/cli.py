"""deepreservoir.drl.cli

Command-line interface for DeepReservoir training/evaluation.

This is intentionally thin glue:
- define CLI args
- instantiate :class:`deepreservoir.drl.model.DRLModel`
- call ``train()`` and (optionally) ``evaluate_test()``

Reward spec syntax
------------------
Comma-separated objective specs.

  - ``objective``                      -> variant="baseline", alpha=1
  - ``objective:variant``              -> alpha=1
  - ``objective:variant@alpha``        -> alpha parsed as float

Example:
  ``dam_safety:storage_band@8.0,esa_min_flow:baseline@3.0,physics:baseline``
"""

from __future__ import annotations

import argparse
import math

from deepreservoir.drl import model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate DeepReservoir agent for Navajo.",
    )

    parser.add_argument(
        "--n-years-test",
        type=int,
        default=10,
        help="Number of (water) years to reserve for testing (most recent).",
    )

    # --- training length ---
    parser.add_argument(
        "--episode-length-train",
        type=int,
        default=3600,
        help="Training episode length (timesteps) used to map n_episodes -> total_timesteps.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help=(
            "Total training timesteps for PPO. If --n-episodes is not provided, the CLI "
            "computes n_episodes = ceil(total_timesteps / episode_length_train)."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help=(
            "Training-length knob used by DRLModel.train(). If provided, overrides --total-timesteps. "
            "Effective total_timesteps = n_episodes * episode_length_train."
        ),
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

    # --- SB3 / PPO knobs ---
    parser.add_argument("--device", type=str, default="auto", help="SB3 device (cpu, cuda, auto).")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor (PPO gamma).")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument(
        "--use-subproc-vec",
        action="store_true",
        help="Use SubprocVecEnv for parallelism (recommended only for script usage).",
    )
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps.")
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO batch_size.")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs.")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="EvalCallback eval_freq (timesteps).")
    parser.add_argument(
        "--no-track-reward-components",
        action="store_true",
        help="Disable tracking mean reward components per PPO update.",
    )

    parser.add_argument(
        "--reward-spec",
        type=str,
        default="dam_safety:storage_band",
        help=(
            "Comma-separated objective specs. Examples: "
            "'dam_safety:storage_band,esa_min_flow:baseline,hydropower:baseline' or with per-objective alpha "
            "multipliers: 'dam_safety:storage_band@8.0,esa_min_flow:baseline@3.0,physics:baseline'"
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/navajo",
        help="Directory to store models and eval logs.",
    )

    # --- evaluation after training ---
    parser.add_argument(
        "--no-evaluate-test",
        action="store_true",
        help="Do not run evaluate_test() after training.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation during evaluate_test().",
    )
    parser.add_argument(
        "--no-rollout",
        action="store_true",
        help="Skip saving eval_test_rollout parquet/csv during evaluate_test().",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip saving eval_metrics.csv/json during evaluate_test().",
    )

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_size > (args.n_steps * args.n_envs):
        raise ValueError(
            f"batch_size ({args.batch_size}) must be <= n_steps*n_envs ({args.n_steps}*{args.n_envs}={args.n_steps*args.n_envs})."
        )

    # Determine n_episodes from total_timesteps if user didn't provide it.
    if args.n_episodes is None:
        args.n_episodes = int(math.ceil(float(args.total_timesteps) / float(args.episode_length_train)))

    eff_total = int(args.n_episodes * args.episode_length_train)
    print(
        "[cli] training config: "
        f"n_episodes={args.n_episodes} episode_length_train={args.episode_length_train} "
        f"-> total_timesteps≈{eff_total} (requested {args.total_timesteps})"
    )

    drlm = model.DRLModel(
        n_years_test=args.n_years_test,
        reward_spec=args.reward_spec,
        algo=args.algo,
        logdir=args.logdir,
        seed=args.seed,
        device=args.device,
        gamma=args.gamma,
        n_envs=args.n_envs,
        use_subproc_vec=args.use_subproc_vec,
        episode_length_train=args.episode_length_train,
    )

    drlm.train(
        n_episodes=args.n_episodes,
        eval_freq=args.eval_freq,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        track_reward_components=(not args.no_track_reward_components),
    )

    if not args.no_evaluate_test:
        _, df_metrics = drlm.evaluate_test(
            save_rollout=(not args.no_rollout),
            save_plots=(not args.no_plots),
            save_metrics=(not args.no_metrics),
        )
        try:
            print("\n[cli] eval metrics:\n" + df_metrics.to_string(index=False))
        except Exception:
            pass


if __name__ == "__main__":
    main()
