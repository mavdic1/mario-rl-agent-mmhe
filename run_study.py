# --- NEW FILE: run_study.py ---
import subprocess
import time

# CONFIGURATION
VERSIONS = ["v1", "v2"]
SEEDS = range(10)  # 0 to 9
STEPS_PER_RUN = 5000000

print("Starting Comparative Study...")
print(f"Total Runs: {len(VERSIONS) * len(SEEDS)}")
start_study_time = time.time()

for version in VERSIONS:
    for seed in SEEDS:
        print("\n" + "="*50)
        print(f"RUNNING: {version.upper()} | SEED: {seed}")
        print("="*50)
        
        # Build the command
        cmd = [
            "python", "train.py",
            "--version", version,
            "--seed", str(seed),
            "--total_steps", str(STEPS_PER_RUN)
        ]
        
        # subprocess.run waits for the training to finish before starting the next one
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nStudy stopped by user.")
            exit()
        except Exception as e:
            print(f"\nRun failed with error: {e}")
            continue

end_study_time = time.time()
total_duration = (end_study_time - start_study_time) / 3600
print(f"\nStudy Complete! Total time: {total_duration:.2f} hours.")