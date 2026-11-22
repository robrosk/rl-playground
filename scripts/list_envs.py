"""Print every supported environment by suite, and whether its packages are installed.

    python scripts/list_envs.py
    python scripts/list_envs.py --installed
"""

from __future__ import annotations

import argparse
from itertools import groupby

from rl_playground.envs.catalog import GYM_SPECS, missing
from rl_playground.envs.multi_agent.catalog import MULTI_AGENT_SPECS

HAND_BUILT = ["frozen_lake", "frozen_lake_det", "frozen_lake_8x8", "gridworld"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--installed", action="store_true", help="only list envs that can be built now")
    args = parser.parse_args(argv)

    print("hand_built (tabular, known model)")
    print("  " + ", ".join(HAND_BUILT))
    for title, specs in (("single-agent", GYM_SPECS), ("multi-agent (PettingZoo)", MULTI_AGENT_SPECS)):
        print(f"\n== {title} ==")
        for suite, group in groupby(specs, key=lambda s: s.suite):
            rows = [s for s in group if s.available() or not args.installed]
            if not rows:
                continue
            print(suite)
            for spec in rows:
                absent = missing(spec.extra)
                status = "ok" if not absent else "needs " + ",".join(absent)
                note = f"  ({spec.notes})" if spec.notes else ""
                print(f"  {spec.name:<28} {status}{note}")


if __name__ == "__main__":
    main()
