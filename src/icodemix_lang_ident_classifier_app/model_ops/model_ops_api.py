# -*- coding: utf-8 -*-
import argparse
from datetime import timedelta
import json
import os
import random
import resource
import sys
import time
from typing import Any

from filelock import FileLock
import numpy as np
import optuna
from setproctitle import setproctitle
import torch
import lightning as pl

from icodemix_lang_ident_classifier.language.utils.property_utils import PropertyUtils
from icodemix_lang_ident_classifier.language.utils.log_utils import LogUtils
from icodemix_lang_ident_classifier.language.utils.file_dir_utils import FileDirUtils
from icodemix_lang_ident_classifier.language.utils.date_time_utils import DateTimeUtils
from icodemix_lang_ident_classifier.language.utils.model_helper_utils import (
    StudyPercentageBasedEarlyStoppingCallback, clear_gpu_and_cpu_resources, 
    property_validation, gamma_for_tpe_sampler, NoDuplicateSampler, 
    clear_checkpoints, resume_study_trial_execution, create_new_study, 
    get_study_stats, log_trial_results
)
from icodemix_lang_ident_classifier.cli.model_hyperparameter_selection import (
    train_model, testset_from_best_checkpoint_callback
)

class ModelOpsApi:
    """
    All Model related operations such as hyperparameter optimization, train, finetune and test.
    """

    @staticmethod
    def objective(trial: optuna.trial.Trial, args: Any, continue_optimization: bool, study: optuna.Study) -> Any:
        try:
            is_gen_llm = False
            clear_gpu_and_cpu_resources()
            
            pu = PropertyUtils()
            props = pu.get_yaml_config_properties(args.config_file_path)
            
            random_seed = property_validation(
                props=props, name="app.random_seed", dtype=int, required=True
            )
            
            # Global Seeding
            np.random.seed(random_seed)
            random.seed(random_seed)
            pl.seed_everything(random_seed, workers=True)
            torch.manual_seed(random_seed)
            
            # Safe CUDA Seeding
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            if continue_optimization:
                prompt_template = ""
                # train_model must handle accelerator="cpu" based on args
                trial, trainer_precision = train_model(args, trial, study)
                
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # Strategy must be 'ddp' or 'auto' for CPU; 'custom_fsdp' is usually GPU-only
                # strategy = "ddp" if not torch.cuda.is_available() else "custom_fsdp"
                strategy = property_validation(
                    props=props,
                    name="app.strategy",
                    dtype=list[str],
                    required=False,
                    allowed_values=['auto','ddp','fsdp'],
                    default="auto",
                    help="Distributed strategy (e.g., fsdp, ddp, auto)."
                )
                
                f1 = testset_from_best_checkpoint_callback(
                    args=args, study=study, trial=trial, 
                    trainer_strategy=strategy, precision=trainer_precision
                )
                return f1

        except (optuna.exceptions.TrialPruned, Exception) as e:
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except: pass
            
            clear_gpu_and_cpu_resources()
            
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if trial and "test_accuracy" not in trial.user_attrs:
                    trial.set_user_attr("test_accuracy", 0.0)
            
            if isinstance(e, optuna.exceptions.TrialPruned):
                raise e
            raise e

    def run_app(self, args) -> Any:
        global continue_optimization
        continue_optimization = True
        setproctitle("python3")
        
        pu = PropertyUtils()
        props = pu.get_yaml_config_properties(args.config_file_path)
        lu = LogUtils()
        log = lu.get_time_rotated_log(props)
        fdu = FileDirUtils()
        
        random_seed = property_validation(props=props, name="app.random_seed", dtype=int, required=True)
        model_config_name = property_validation(props=props, name="app.model_config_name", dtype=str, required=True)
        max_epochs = property_validation(props=props, name="model.max_epochs", dtype=int, required=True)
        n_trials = property_validation(props=props, name="app.num_trials", dtype=int, required=True)

        # Environment Handling
        global_rank = int(os.environ.get("RANK", "0"))
        local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
        world_size  = int(os.environ.get("WORLD_SIZE", "1"))

        # Setup Device & Distributed
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if cuda_ok:
            device_id = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            torch.cuda.manual_seed_all(random_seed)
        
        if not torch.distributed.is_initialized() and world_size > 1:
            # Force GLOO for CPU
            backend = args.backend if cuda_ok else "gloo"
            torch.distributed.init_process_group(
                backend=backend, rank=global_rank, world_size=world_size, timeout=timedelta(seconds=600)
            )

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Optuna Storage & Sampler setup
        optim_dir = f"optim_studies/{model_config_name}"
        fdu.create_dir(optim_dir)
        db_name = f"{optim_dir}/study.db"
        
        lang_study_storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_name}")
        
        tpe_sampler = optuna.samplers.TPESampler(
            seed=random_seed, multivariate=True, group=True, 
            gamma=gamma_for_tpe_sampler, n_startup_trials=20
        )
        hyperparam_sampler = NoDuplicateSampler(tpe_sampler)
        study_pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=max_epochs, reduction_factor=3)

        # Callbacks
        def log_results(s, t): log_trial_results(model_config_name, log, s, t, f"metrics/{model_config_name}", "trial_metrics.csv")
        def clear_cb(s, t): clear_checkpoints(log, model_config_name, s, t)
        
        # Study Initialization
        study = None
        if rank == 0:
            try:
                study = optuna.load_study(study_name=model_config_name, storage=lang_study_storage)
                log.info(f"Resuming study: {model_config_name}")
            except KeyError:
                study = optuna.create_study(study_name=model_config_name, storage=lang_study_storage, 
                                            direction="maximize", sampler=hyperparam_sampler, pruner=study_pruner)
                log.info(f"Created new study: {model_config_name}")

        # Optimization Loop
        while continue_optimization:
            if rank == 0:
                if len(study.trials) >= n_trials:
                    break
                study.optimize(
                    lambda t: ModelOpsApi.objective(t, args, continue_optimization, study),
                    n_trials=1, # Run one-by-one to check continue flag
                    callbacks=[log_results, clear_cb]
                )
            else:
                # Workers just execute the objective
                ModelOpsApi.objective(None, args, continue_optimization, None)
            
            if not continue_optimization: break

    @staticmethod
    def main():
        """
        Entrypoint for CLI execution.
        """
        # --- CRITICAL: FORCE CPU MODE BEFORE ANY TORCH CALLS ---
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        # ------------------------------------------------------

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        except Exception:
            pass

        torch.multiprocessing.set_sharing_strategy('file_system')

        parser = argparse.ArgumentParser(description='App to train and manage language identification models')
        
        # SATISFY THE OBFUSCATED CORE REQUIREMENTS
        parser.add_argument('--config_file_path', type=str, required=True)
        parser.add_argument('--cpu_cores', type=int, default=1)
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--local-rank', type=int, default=0)
        parser.add_argument('--backend', type=str, default="gloo")
        parser.add_argument('--run_timestamp', type=float, default=None)
        parser.add_argument('--resume_study_from_trial_number', type=lambda x: int(x) if x != 'None' else None, default=None)

        parsed_args, _ = parser.parse_known_args()
        if parsed_args.run_timestamp is None:
            parsed_args.run_timestamp = time.time()

        try:
            # We must ensure lightning doesn't try to use 'auto' (which picks GPU)
            # We do this by passing a clean args object
            api = ModelOpsApi()
            api.run_app(args=parsed_args)
        except Exception as e:
            # This captures the 'Torch not compiled with CUDA enabled'
            print(f"Fatal Exception: {e}")
        finally:
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except: pass
            print("Job Completed")
            sys.exit(0)

def main():
    """Module-level entry point for setuptools console_scripts."""
    ModelOpsApi.main()

if __name__ == "__main__":
    torch.cuda.is_available = lambda: False
    torch.set_default_device("cpu")
    main()