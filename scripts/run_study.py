import subprocess
import time
import os

from src.agent.config import TOTAL_TIMESTEPS

# CONFIGURATION
VERSIONS = ["v1", "v2"]
SEEDS = range(2)  # 0 to 9

def check_if_done(version, seed):
    """
    Checks if a specific run is already finished.
    We look for a hidden '.done' file created by this script 
    or check if the latest model exists (optional).
    """
    # This path must match the STUDY_DIR logic in train.py
    done_marker = f"study/{version}/seed_{seed}/.completed"
    return os.path.exists(done_marker)

def mark_as_done(version, seed):
    done_marker = f"study/{version}/seed_{seed}/.completed"
    os.makedirs(os.path.dirname(done_marker), exist_ok=True)
    with open(done_marker, "w") as f:
        f.write(f"Completed at {time.ctime()}")

print("Starting/Resuming Comparative Study...")
start_study_time = time.time()

for version in VERSIONS:
    for seed in SEEDS:
        if check_if_done(version, seed):
            print(f"SKIPPING: {version.upper()} | SEED: {seed} (Already Finished)")
            continue

        print("\n" + "="*50)
        print(f"RUNNING: {version.upper()} | SEED: {seed}")
        print("="*50)
        
        # We use 'python -m main.train' to ensure imports work 
        # correctly from the root directory
        cmd = [
            "python", "-m", "scripts.train",
            "--version", version,
            "--seed", str(seed),
            "--total_steps", str(TOTAL_TIMESTEPS)
        ]
        
        try:
            # check=True will raise an error if the training script crashes
            subprocess.run(cmd, check=True)
            
            # If we reach here, the script finished successfully
            mark_as_done(version, seed)
            
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Run {version} Seed {seed} crashed with exit code {e.returncode}.")
            print("Moving to next run. You can restart the study script to try this one again later.")
            continue
        except KeyboardInterrupt:
            print("\nStudy stopped by user.")
            exit()

total_duration = (time.time() - start_study_time) / 3600
print(f"\nStudy Process Complete! Total time: {total_duration:.2f} hours.")