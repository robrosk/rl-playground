"""Train several agents in one game, choosing each agent's algorithm.

    python scripts/run_multi.py --env tictactoe --seat player_1=ppo --seat player_2=random
    python scripts/run_multi.py --env connect_four --seat player_0=ppo --seat player_1=dqn \\
        --set player_0.lr=3e-4 --set player_1.warmup=500
    python scripts/run_multi.py --env tictactoe --seat "*=ppo:shared"          # self-play
    python scripts/run_multi.py --env simple_tag --seat "adversary_*=ppo:shared" --seat "*=dqn"

A seat is PATTERN=ALGO, or PATTERN=ALGO:shared for one learner across every matched agent.
PATTERN is an agent name or a glob; the first seat that matches an agent wins. ALGO is any
algorithm in ALGORITHMS or a fixed policy (random, first_legal). ``--set SEAT.KEY=VALUE``
changes one seat's config, where SEAT is the seat's pattern as written in ``--seat``.
"""

from __future__ import annotations

import argparse

from rl_playground.build import build_multi_agent_runner
from rl_playground.multi_agent import MultiAgentRunConfig, SeatSpec

try:
    from run import parse_overrides
except ImportError:  # run as a module
    from scripts.run import parse_overrides


def parse_seats(seats, sets):
    overrides = {}
    for pair in sets:
        key, _, value = pair.partition("=")
        pattern, _, field = key.rpartition(".")
        if not pattern:
            raise SystemExit(f"--set {pair!r}: write it as SEAT.KEY=VALUE")
        overrides.setdefault(pattern, []).append(f"{field}={value}")
    specs = []
    for seat in seats:
        pattern, _, algo = seat.partition("=")
        algo, _, flag = algo.partition(":")
        if not pattern or not algo or flag not in ("", "shared"):
            raise SystemExit(f"--seat {seat!r}: write it as PATTERN=ALGO or PATTERN=ALGO:shared")
        specs.append(SeatSpec(pattern, algo, parse_overrides(overrides.pop(pattern, [])), shared=flag == "shared"))
    if overrides:
        raise SystemExit(f"--set names seats that no --seat defines: {sorted(overrides)}")
    return tuple(specs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="a name in MULTI_AGENT_ENVS (scripts/list_envs.py)")
    parser.add_argument("--seat", action="append", required=True, metavar="PATTERN=ALGO[:shared]")
    parser.add_argument("--set", action="append", default=[], metavar="SEAT.KEY=VALUE")
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args(argv)

    run = MultiAgentRunConfig(args.env, parse_seats(args.seat, args.set), args.budget, args.seed, args.log_every)
    runner = build_multi_agent_runner(run).run(run.budget)
    print(f"{runner.driver.episodes} episodes")
    for agent, returns in runner.driver.episode_returns.items():
        if returns:
            recent = returns[-run.returns_window:]
            print(f"  {agent:<16} mean return {sum(recent) / len(recent):+.3f} over the last {len(recent)}")


if __name__ == "__main__":
    main()
