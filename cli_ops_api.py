#!/usr/bin/env python3
import sys
from pathlib import Path
from icodemix_lang_ident_classifier.language.utils.property_utils import PropertyUtils
from icodemix_lang_ident_classifier.language.utils.log_utils import LogUtils

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

import os
import subprocess
from datetime import datetime
import socket
import torch
import shutil
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter, WordCompleter

# ----------------------
# --- UTILITY FUNCTIONS ---
# ----------------------
def prompt_choice(prompt, choices, default=None):
    choices_str = "/".join(choices)
    while True:
        resp = input(f"{prompt} [{choices_str}]{' (default='+default+')' if default else ''}: ").strip()
        if not resp and default:
            return default
        if resp in choices:
            return resp
        print(f"Invalid choice. Please choose one of: {choices_str}")

def prompt_input(prompt, default=None):
    resp = input(f"{prompt}{' (default='+default+')' if default else ''}: ").strip()
    return resp or default

def find_free_port(start=8000, end=35000):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No free port found")


# ----------------------
# --- JOB RUNNER CLASS ---
# ----------------------
class LangIdentJobRunner:
    def __init__(self):
        self.env_type = ""
        self.env_value = ""
        self.config_file = ""
        self.resume_trial = ""
        self.backend = "nccl"
        self.cpu_cores = ""
        self.ppn = 1
        self.master_port = None
        self.run_timestamp = None
        self.job_name = ""
        self.workdir = Path.cwd()
        self.job_log_dir = None
        self.log_file = None

        #  NEW
        self.operation = ""

    # ----------------------
    # --- USER INTERACTION ---
    # ----------------------
    def prompt_choice_pt(self, prompt_text, choices, default=None):
        completer = WordCompleter(choices, ignore_case=True)
        while True:
            resp = prompt(
                f"{prompt_text}{f' (default={default})' if default else ''}: ",
                completer=completer
            ).strip()
            if not resp and default:
                return default
            if resp in choices:
                return resp
            print(f"Invalid choice. Choose from: {', '.join(choices)}")


    def prompt_input_pt(self, prompt_text, default=None, path=False):
        completer = PathCompleter(expanduser=True) if path else None
        resp = prompt(
            f"{prompt_text}{f' (default={default})' if default else ''}: ",
            completer=completer
        ).strip()
        return resp or default


    # ----------------------
    # DROP-IN REPLACEMENT
    # ----------------------
    def gather_user_inputs(self):
        print("Welcome to the Lang Ident Classifier job runner!")

        # Operation selection
        self.operation = self.prompt_choice_pt(
            "Choose operation (Press Tab Key to View Options)",
            ["process_bhasha_dataset", "model_hyperparameter_optimization"],
            default="process_bhasha_dataset"
        )

        # Environment
        self.env_type = self.prompt_choice_pt(
            "Choose environment type",
            ["conda", "docker", "none"],
            default="none"
        )

        if self.env_type in ("conda", "docker"):
            self.env_value = self.prompt_input_pt(
                f"Enter the {self.env_type} environment name or image"
            )

        # Config file (WITH TAB COMPLETION)
        self.config_file = self.prompt_input_pt(
            "Enter path to config YAML",
            default="config/embedding_based_classifiers_config.yaml",
            path=True
        )
        self.config_file = str(Path(self.config_file).resolve())
        pu = PropertyUtils()
        lu = LogUtils()
        props = pu.get_yaml_config_properties(self.config_file)
        log = lu.get_time_rotated_log(props)
        self.log_file = log.handlers[0].baseFilename

        # Optional arguments
        if "process" not in self.operation:
            self.resume_trial = self.prompt_input_pt(
                "Enter trial number to resume from (optional)",
                default=""
            )
            self.backend = self.prompt_choice_pt(
                "Choose backend",
                ["nccl", "gloo"],
                default="nccl"
            )
            self.cpu_cores = self.prompt_input_pt(
                "Specify number of CPU cores (optional)",
                default=""
            )

    # ----------------------
    # --- SETUP JOB ---
    # ----------------------
    def setup_job(self):
        PROJECT_ROOT = Path(__file__).parent.resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.ppn = len(cuda_visible.split(",")) if cuda_visible else 1

        gpu_available = torch.cuda.is_available() and self.ppn > 0

        if self.env_type == "none" and not gpu_available:
            print("[INFO] No GPUs detected. Switching backend to 'gloo'.")
            self.backend = "gloo"
        else:
            print(f"[INFO] Backend set to '{self.backend}'")

        print(f"[INFO] Detected CUDA_VISIBLE_DEVICES={cuda_visible}, PPN={self.ppn}")

        self.job_log_dir = self.workdir / "logs" / self.job_name
        self.job_log_dir.mkdir(parents=True, exist_ok=True)
        self.master_port = find_free_port()
        self.run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if self.cpu_cores:
            for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                os.environ[var] = self.cpu_cores
            print(f"[INFO] Setting CPU thread vars to {self.cpu_cores}")

    # ----------------------
    # --- BUILD COMMAND ---
    # ----------------------
    def build_python_cmd(self, config_path):

        if self.operation == "process_bhasha_dataset":
            return [
                "python", "-u",
                "-m", "icodemix_lang_ident_classifier_app.cli.main",
                f"--config_file_path={config_path}",
                f"--operation_mode={self.operation}",
            ]
        else:
            cmd = [
                "python", "-u", "-m", "torch.distributed.run",
                f"--nproc-per-node={self.ppn}",
                f"--master-port={self.master_port}",
                "-m", "icodemix_lang_ident_classifier_app.cli.main",
                "--config_file_path", config_path,
                "--operation_mode", self.operation,
                "--backend", self.backend,
                "--run_timestamp", self.run_timestamp
            ]

            if self.cpu_cores:
                cmd += ["--cpu_cores", self.cpu_cores]

            if self.resume_trial:
                cmd += ["--resume_study_from_trial_number", self.resume_trial]

            return cmd

    # ----------------------
    # --- RUN JOB IN BACKGROUND ---
    # ----------------------
    def run(self):
        ROOT = Path(__file__).resolve().parent
        SRC = ROOT / "src"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        print(f"\n[INFO] Ready to run job in background. Logs will be saved to {self.log_file}")
        print(f"[INFO] Operation: {self.operation}\n")

        cmd = []
        if self.env_type == "conda":
            cmd = f"conda run -n {self.env_value} " + " ".join(self.build_python_cmd(self.config_file))
            process = subprocess.Popen(cmd, shell=True, env=env ,stdout=open(self.log_file, "w"), stderr=subprocess.STDOUT, start_new_session=True)

        elif self.env_type == "docker":
            uid, gid, uname = os.getuid(), os.getgid(), os.getenv("USER", "user")
            gpu_flag = ""
            runtime_flag = ""
            if shutil.which("nvidia-smi"):
                runtime_flag = "--runtime=nvidia"
                cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                gpu_flag = f"--gpus device={cuda_env}" if cuda_env else ""
            docker_cmd = [
                "docker", "run", "--rm", runtime_flag, gpu_flag,
                "-v", f"{self.workdir}:/app",
                "--ipc=host",
                "-w", "/app",
                "-e", f"UID={uid}", "-e", f"GID={gid}", "-e", f"USERNAME={uname}", "-e", "HF_CACHE=/app/.cache", "-e", "PYTHONPATH=/app/src",
                self.env_value
            ] + self.build_python_cmd(f"/app/{self.config_file}")
            process = subprocess.Popen(" ".join(docker_cmd), shell=True, stdout=open(self.log_file, "w"), stderr=subprocess.STDOUT, start_new_session=True)

        elif self.env_type == "none":
            process = subprocess.Popen(self.build_python_cmd(self.config_file), env=env ,stdout=open(self.log_file, "w"), stderr=subprocess.STDOUT, start_new_session=True)

        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")

        print(f"[INFO] Job started in background with PID {process.pid}")
        print(f"[INFO] You can monitor logs using:\n  tail -f {self.log_file}")


# ----------------------
# --- MAIN ENTRY POINT ---
# ----------------------
def main():
    runner = LangIdentJobRunner()
    runner.gather_user_inputs()
    runner.setup_job()
    runner.run()


if __name__ == "__main__":
    main()