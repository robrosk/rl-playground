"""Run any registered algorithm on any env.

    python scripts/run.py value_iteration --env frozen_lake
    python scripts/run.py q_learning --env frozen_lake --budget 50000 --set alpha=0.2
    python scripts/run.py ppo --env gym:Acrobot-v1 --budget 500000
"""

from __future__ import annotations

import argparse
import ast

from rl_playground.algorithms import ALGORITHMS
from rl_playground.build import build_runner
from rl_playground.core.config import RunConfig, config_for


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            overrides[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[key] = raw
    return overrides


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("algo", choices=ALGORITHMS.names())
    parser.add_argument("--env", default="frozen_lake")
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="algorithm config overrides")
    args = parser.parse_args(argv)

    run = RunConfig(algo=args.algo, env=args.env, budget=args.budget, seed=args.seed, log_every=args.log_every)
    runner = build_runner(run, config_for(args.algo, **parse_overrides(args.set)))
    runner.run(run.budget)


if __name__ == "__main__":
    main()
