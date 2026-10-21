#!/usr/bin/env python
"""Run (or resume) an SYC Monte Carlo simulation.

Usage
-----
    python scripts/run_syc.py --config config/example_trust_on_ru.json
    python scripts/run_syc.py --config my.json --max-cycles 20 --seed 1
    python scripts/run_syc.py --config my.json --no-relax      # MC only, no Matlantis

With relaxation enabled, start ``scripts/matlantis_relax.py --stage pre`` and
``--stage post`` on Matlantis first (see README).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from syc import SYCConfig, run_simulation  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="SYC Monte Carlo driver")
    p.add_argument("--config", required=True, help="JSON configuration file")
    p.add_argument("--max-cycles", type=int, help="override max_cycles")
    p.add_argument("--seed", type=int, help="override RNG seed")
    p.add_argument("--no-relax", action="store_true", help="disable both relaxation stages")
    args = p.parse_args()

    cfg = SYCConfig.from_json(args.config)
    if args.max_cycles is not None:
        cfg.max_cycles = args.max_cycles
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_relax:
        cfg.use_relaxation_pre = False
        cfg.use_relaxation_post = False

    results = run_simulation(cfg)
    if results["stats"]:
        last = results["stats"][-1]
        print(f"\n[DONE] last cycle {last['cycle']}: chemisorbed = {last['chemisorbed']}, "
              f"thickness = {last['thickness_A']:.2f} Å")
        print(f"[DONE] per-cycle stats: {os.path.join(cfg.output_dir, 'cycle_stats.csv')}")


if __name__ == "__main__":
    main()
