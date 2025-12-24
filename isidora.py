import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "task1-1": "task1_1_make_onsite_submission.py",
    "task1-2": "task1_2_frozen_backbone_train_head.py",
    "task1-3": "task1_3_full_finetune.py",
    "task2-1": "task2_1_focal_loss.py",
    "task2-2": "task2_2_class_balanced_full_finetune.py",
    "task3-1": "task3_1_se_full_finetune.py",
    "task3-2": "task3_2_mha_full_finetune.py",
    "task4":   "task4_vae_augment_and_finetune.py",
}


def run_script(script_name: str) -> int:
    script_path = ROOT / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    print(f"\n[INFO] Running: {script_name}\n")
   
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning Final Project - Isidora Erakovic (entry point)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(SCRIPTS.keys()) + ["all"],
        help="Which task to run (e.g., task3-2). Use 'all' to run everything sequentially.",
    )
    args = parser.parse_args()

    if args.task == "all":
        for k in ["task1-1", "task1-2", "task1-3", "task2-1", "task2-2", "task3-1", "task3-2", "task4"]:
            code = run_script(SCRIPTS[k])
            if code != 0:
                print(f"[ERROR] Task {k} failed with code {code}. Stopping.")
                sys.exit(code)
        print("\n[INFO] All tasks finished successfully.")
        return

    script = SCRIPTS[args.task]
    code = run_script(script)
    if code != 0:
        print(f"[ERROR] {args.task} failed with code {code}.")
    sys.exit(code)


if __name__ == "__main__":
    main()
