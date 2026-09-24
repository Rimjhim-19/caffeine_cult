"""
Run All EDA Sections sequentially for Amazon ML Challenge 2026: Business Entity Resolution.
Executes scripts 01 through 06 and checks execution status.
"""

import subprocess
import sys
import time

SCRIPTS = [
    ("Section A: Basic Structure", "eda/01_basic_structure.py"),
    ("Section B: Ground Truth Structure", "eda/02_ground_truth.py"),
    ("Section C: Noise Pattern Study", "eda/03_noise_patterns.py"),
    ("Section D: Similarity Separability", "eda/04_similarity_separability.py"),
    ("Section E: Blocking Baseline", "eda/05_blocking_baseline.py"),
    ("Section F: Train-vs-Test Shift", "eda/06_train_test_shift.py"),
]

def main():
    print("=" * 70)
    print("STARTING FULL EDA PIPELINE (SECTIONS A TO F)")
    print("=" * 70)
    
    total_start = time.time()
    
    for title, script_path in SCRIPTS:
        print(f"\n>>> Running {title} ({script_path})...")
        start = time.time()
        res = subprocess.run([sys.executable, script_path])
        elapsed = time.time() - start
        if res.returncode != 0:
            print(f"FAILED: {title} exited with returncode {res.returncode}")
            sys.exit(res.returncode)
        print(f"COMPLETED: {title} in {elapsed:.1f}s")
        
    print("\n" + "=" * 70)
    print(f"ALL EDA SECTIONS COMPLETED SUCCESSFULLY IN {time.time() - total_start:.1f}s!")
    print("Outputs available in eda/output/ and report in eda/EDA_REPORT.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
