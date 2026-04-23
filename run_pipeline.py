"""
run_pipeline.py
================
Runs the full WNBA draft model pipeline in one command.

Usage:
    python run_pipeline.py            # model + scores + webapp (skips slow scraping)
    python run_pipeline.py --scrape   # also re-scrapes NCAA player data (slow, ~1 hr)
    python run_pipeline.py --all      # everything including measurables scrape

Steps:
    [scrape]  01_scrape_multi_year.py        — scrape NCAA stats (slow, skip if data exists)
    [scrape]  01b_scrape_draft_class_only.py — scrape historical draft picks' college stats
                                               (fast after first run — skips already-scraped years)
    [all]     02_scrape_measurables.py       — scrape height/weight (slow, optional)
    [scrape]  03_opp_strength.py             — compute opponent strength metrics
              04_build_features.py           — build feature matrix
    [all]     05_scrape_wnba_archetypes.py   — scrape WNBA player stats + cluster into archetypes
    [all]     05b_wnba_team_needs.py         — scrape 2025 WNBA team stats → wnba_top5_needs.csv
              06_archetype_classifier.py     — train college+height → WNBA archetype classifier
              07_xgboost_model.py            — train readiness model + score prospects
              08_fit_scores.py               — compute team fit scores
              generate_team_needs.py         — generate webapp/public/team_needs.json
              prepare_data.py                — generate webapp/public/players.json
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

PIPELINE_DIR = ROOT / "pipeline"

STEPS = [
    # (script_path, description, requires_flag)
    (PIPELINE_DIR / "01_scrape_multi_year.py",        "Scraping NCAA player data (slow)",                        "scrape"),
    (PIPELINE_DIR / "01b_scrape_draft_class_only.py", "Scraping historical draft picks' college stats (targeted)", "scrape"),
    (PIPELINE_DIR / "02_scrape_measurables.py",       "Scraping player measurables (slow)",                      "all"),
    (PIPELINE_DIR / "03_opp_strength.py",             "Computing opponent strength metrics",                     "scrape"),
    (PIPELINE_DIR / "04_build_features.py",         "Building feature matrix",                      None),
    (PIPELINE_DIR / "05_scrape_wnba_archetypes.py",  "Scraping WNBA player stats + clustering archetypes", "all"),
    (PIPELINE_DIR / "05b_wnba_team_needs.py",        "Scraping 2025 WNBA team stats → team needs CSV",    "all"),
    (PIPELINE_DIR / "06_archetype_classifier.py",    "Training archetype classifier (college+height→WNBA)", None),
    (PIPELINE_DIR / "07_xgboost_model.py",           "Training readiness model + scoring prospects", None),
    (PIPELINE_DIR / "08_fit_scores.py",              "Computing team fit scores",                    None),
    (ROOT         / "generate_team_needs.py",        "Generating webapp/public/team_needs.json",     None),
    (ROOT         / "prepare_data.py",               "Generating webapp/public/players.json",        None),
]


def run_step(script: Path, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {script.name}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )

    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAIL] {script.name} exited with code {result.returncode} ({elapsed:.0f}s)")
        return False

    print(f"\n[OK] {script.name} completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the WNBA draft model pipeline.")
    parser.add_argument("--scrape", action="store_true",
                        help="Also re-scrape NCAA multi-year player data (slow, ~1 hour)")
    parser.add_argument("--all", action="store_true",
                        help="Run everything including measurables scrape")
    parser.add_argument("--from-step", type=int, default=1, metavar="N",
                        help="Start from step N (1=scrape, 2=historical, 3=measurables, 4=opp-strength, 5=features, 6=wnba-players, 7=wnba-teams, 8=classifier, 9=model, 10=fit, 11=team-needs-json, 12=webapp)")
    args = parser.parse_args()

    if args.all:
        args.scrape = True

    print("WNBA Draft Model Pipeline")
    print(f"  --scrape: {args.scrape}")
    print(f"  --all:    {args.all}")
    print(f"  --from-step: {args.from_step}")

    failed = []
    for i, (script, description, flag) in enumerate(STEPS, start=1):
        if i < args.from_step:
            print(f"\n[SKIP] Step {i}: {script.name} (--from-step={args.from_step})")
            continue

        # Skip scrape-only steps unless the right flag is set
        if flag == "scrape" and not args.scrape:
            print(f"\n[SKIP] Step {i}: {script.name} (pass --scrape to run)")
            continue
        if flag == "all" and not args.all:
            print(f"\n[SKIP] Step {i}: {script.name} (pass --all to run)")
            continue

        if not script.exists():
            print(f"\n[SKIP] Step {i}: {script.name} not found")
            continue

        ok = run_step(script, f"Step {i}: {description}")
        if not ok:
            failed.append(script.name)
            ans = input(f"\nStep {i} failed. Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("Pipeline aborted.")
                sys.exit(1)

    print(f"\n{'='*60}")
    if failed:
        print(f"Pipeline finished with failures: {', '.join(failed)}")
    else:
        print("Pipeline complete.")
        print("\nNext steps:")
        print("  Streamlit app: streamlit run app/app.py")
        print("  React webapp:  cd webapp && npm run dev")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
