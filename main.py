import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os

class MarioInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario RL Agent Manager")
        self.root.geometry("500x600")
        self.data_path = os.path.join(os.getcwd(), "data", "study")
        
        # --- TOP SECTION: GLOBAL PARAMETERS ---
        params_frame = ttk.LabelFrame(root, text=" Global Settings ", padding=15)
        params_frame.pack(fill="x", padx=20, pady=10)

        # Version Selection
        ttk.Label(params_frame, text="Current Version:").grid(row=0, column=0, sticky="w", pady=5)
        self.version_var = tk.StringVar(value="v2")
        self.version_combo = ttk.Combobox(params_frame, textvariable=self.version_var, values=["v1", "v2"], width=10)
        self.version_combo.grid(row=0, column=1, sticky="w", padx=10)
        self.version_combo.bind("<<ComboboxSelected>>", self.refresh_seeds)

        # Seed Selection
        ttk.Label(params_frame, text="Current Seed:").grid(row=1, column=0, sticky="w", pady=5)
        self.seed_var = tk.StringVar()
        self.seed_combo = ttk.Combobox(params_frame, textvariable=self.seed_var, width=10)
        self.seed_combo.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Button(params_frame, text="Scan Folders", command=self.refresh_seeds).grid(row=1, column=2, padx=5)

        # --- TRAINING SECTION ---
        train_frame = ttk.LabelFrame(root, text=" 1. Training ", padding=15)
        train_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(train_frame, text="Steps:").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.StringVar(value="5000000")
        ttk.Entry(train_frame, textvariable=self.steps_var, width=15).grid(row=0, column=1, sticky="w", padx=10)

        self.dash_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(train_frame, text="Show Live Dashboard", variable=self.dash_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Button(train_frame, text="START / RESUME TRAINING", command=self.run_train).grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # --- PLAY SECTION ---
        play_frame = ttk.LabelFrame(root, text=" 2. Play / Watch ", padding=15)
        play_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(play_frame, text="Load Model:").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="best")
        ttk.Combobox(play_frame, textvariable=self.mode_var, values=["best", "latest"], width=10).grid(row=0, column=1, sticky="w", padx=10)

        ttk.Button(play_frame, text="LAUNCH WATCHER", command=self.run_play).grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # --- STUDY SECTION ---
        study_frame = ttk.LabelFrame(root, text=" 3. Research Study ", padding=15)
        study_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(study_frame, text="RUN 10-SEED COMPARATIVE STUDY", command=self.run_study, style="Accent.TButton").pack(fill="x")

        # Status Bar
        self.status_label = ttk.Label(root, text="Ready", foreground="gray")
        self.status_label.pack(side="bottom", pady=10)

        # Initial Refresh
        self.refresh_seeds()

    def refresh_seeds(self, event=None):
        """Scans the data directory to see which seeds exist for the chosen version."""
        version = self.version_var.get()
        version_path = os.path.join(self.data_path, version)
        
        seeds = []
        if os.path.exists(version_path):
            # Find all folders starting with 'seed_'
            seeds = [f.replace("seed_", "") for f in os.listdir(version_path) if f.startswith("seed_")]
            seeds.sort(key=int)
        
        if seeds:
            self.seed_combo['values'] = seeds
            if self.seed_var.get() not in seeds:
                self.seed_var.set(seeds[0])
            self.status_label.config(text=f"Found {len(seeds)} existing seeds for {version}", foreground="blue")
        else:
            self.seed_combo['values'] = ["0"]
            self.seed_var.set("0")
            self.status_label.config(text=f"No existing data for {version}. Defaulting to seed 0.", foreground="orange")

    def run_command(self, cmd):
        """Runs the command in a new gnome-terminal."""
        try:
            # We use python3 -m to ensure imports work from root
            cmd_str = f"python3 -m {' '.join(cmd)}"
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f"{cmd_str}; exec bash"])
            self.status_label.config(text=f"Running: {' '.join(cmd[:2])}", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch: {e}")

    def run_train(self):
        cmd = [
            "scripts.train",
            f"--version={self.version_var.get()}",
            f"--seed={self.seed_var.get()}",
            f"--total_steps={self.steps_var.get()}"
        ]
        if self.dash_var.get(): cmd.append("--dashboard")
        self.run_command(cmd)

    def run_play(self):
        # Double check if model exists before launching
        v = self.version_var.get()
        s = self.seed_var.get()
        m = self.mode_var.get()
        
        # Construct path to check
        model_name = "mario_ppo_best.zip" if m == "best" else "mario_ppo_latest.zip"
        check_path = os.path.join(self.data_path, v, f"seed_{s}", "models", model_name)
        
        if not os.path.exists(check_path):
            messagebox.showwarning("Missing Model", f"No {m} model found for {v} Seed {s}.\nPath: {check_path}")
            return

        cmd = [
            "scripts.play",
            f"--version={v}",
            f"--seed={s}",
            f"--mode={m}"
        ]
        self.run_command(cmd)

    def run_study(self):
        msg = "Start 10-seed study? This runs V1 and V2 across seeds 0-9 (100M steps total)."
        if messagebox.askyesno("Research Study", msg):
            self.run_command(["scripts.run_study"])

if __name__ == "__main__":
    root = tk.Tk()
    # If you are on Fedora, the default theme is fine, but you can add 'clam' for a cleaner look
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    app = MarioInterface(root)
    root.mainloop()