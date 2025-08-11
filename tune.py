import argparse
import os
import torch
import gc
import optuna
import wandb
from pathlib import Path

# Import project components
from darija_llm_lab.utils.common import ConfigurationManager
from darija_llm_lab.components.model_loader import ModelLoader
from darija_llm_lab.components.data_ingestion import DataIngestion
from darija_llm_lab.components.model_trainer import ModelTrainer
from darija_llm_lab.entity.config_entity import SFTPEFTConfig, DAPTTrainerConfig, SFTTrainerConfig

def objective_dapt(trial: optuna.Trial, base_model_config, dapt_base_config, new_tokens) -> float:
    """
    Objective function for Domain-Adapted Pre-training (DAPT) tuning.
    A single trial represents one fast DAPT training run.
    """
    try:
        # --- 1. Define Hyperparameter Search Space ---
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 2e-5, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 10, 50)
        
        run_name = f"dapt_trial_{trial.number}_lr_{learning_rate:.2e}_warmup_{warmup_steps}"
        wandb.init(name=run_name, reinit=True, config=trial.params)

        # --- 2. Create Trial-Specific Configurations ---
        trial_trainer_config = DAPTTrainerConfig(
            output_dir=Path(f"artifacts/tuning/dapt/trial_{trial.number}"),
            per_device_train_batch_size=dapt_base_config.trainer.per_device_train_batch_size,
            gradient_accumulation_steps=dapt_base_config.trainer.gradient_accumulation_steps,
            max_steps=100,  # Use a small number of steps for fast tuning
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=50, # Save infrequently during tuning
            warmup_steps=warmup_steps,
            seed=dapt_base_config.trainer.seed
        )
        
        # --- 3. Run the Miniature DAPT Pipeline ---
        model_loader = ModelLoader(base_model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()
        model, tokenizer = model_loader.perform_embedding_surgery(new_tokens)
        model = model_loader.prepare_model_for_dapt()

        data_ingestion = DataIngestion()
        train_dataset = data_ingestion.load_dapt_dataset(dapt_base_config.data, tokenizer)

        trainer_component = ModelTrainer()
        trainer = trainer_component.get_dapt_trainer(model, tokenizer, train_dataset, trial_trainer_config)
        
        result = trainer.train()
        
        # DAPT has no eval set, so return the final training loss
        final_training_loss = result.training_loss
        wandb.log({"final_training_loss": final_training_loss})
        
        return final_training_loss

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Prune the trial if it fails (e.g., OOM error)
        raise optuna.exceptions.TrialPruned()
    finally:
        # --- 4. Clean up memory ---
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()


def objective_sft(trial: optuna.Trial, base_model_config, sft_base_config) -> float:
    """
    Objective function for Supervised Fine-Tuning (SFT) tuning.
    A single trial represents one fast SFT training run.
    """
    try:
        # --- 1. Define Hyperparameter Search Space ---
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        lora_r = trial.suggest_categorical("r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [lora_r, 2 * lora_r])
        
        run_name = f"sft_trial_{trial.number}_lr_{learning_rate:.2e}_r_{lora_r}_alpha_{lora_alpha}"
        wandb.init(name=run_name, reinit=True, config=trial.params)

        # --- 2. Create Trial-Specific Configurations ---
        trial_peft_config = SFTPEFTConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=sft_base_config.peft.lora_dropout,
            bias=sft_base_config.peft.bias,
            random_state=sft_base_config.peft.random_state,
            target_modules=sft_base_config.peft.target_modules,
        )
        
        trial_trainer_config = SFTTrainerConfig(
            output_dir=Path(f"artifacts/tuning/sft/trial_{trial.number}"),
            per_device_train_batch_size=sft_base_config.trainer.per_device_train_batch_size,
            gradient_accumulation_steps=sft_base_config.trainer.gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=50,
            warmup_steps=10,
            max_steps=100, # Use a small number of steps for fast tuning
            learning_rate=learning_rate,
            logging_steps=5,
            seed=sft_base_config.trainer.seed,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True, # Crucial for getting the best metric
            greater_is_better=False,
            early_stopping_patience=sft_base_config.trainer.early_stopping_patience,
        )

        # --- 3. Run the Miniature SFT Pipeline ---
        model_loader = ModelLoader(base_model_config)
        # Load the DAPT-adapted model
        model, tokenizer = model_loader.load_model_and_tokenizer(model_path=str(sft_base_config.input_model_path))
        model = model_loader.apply_sft_peft(trial_peft_config)

        data_ingestion = DataIngestion()
        train_ds, eval_ds = data_ingestion.load_sft_dataset(sft_base_config.data, tokenizer)
        
        trainer_component = ModelTrainer()
        trainer = trainer_component.get_sft_trainer(model, tokenizer, train_ds, eval_ds, trial_trainer_config)
        
        trainer.train()

        # Return the best evaluation loss achieved during the trial
        best_eval_loss = trainer.state.best_metric
        wandb.log({"best_eval_loss": best_eval_loss})
        
        return best_eval_loss

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        # --- 4. Clean up memory ---
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for DAPT or SFT.")
    parser.add_argument("--stage", type=str, required=True, choices=["dapt", "sft"],
                        help="The training stage to tune: 'dapt' or 'sft'.")
    args = parser.parse_args()

    # --- Setup Configurations and Wandb ---
    config_manager = ConfigurationManager()
    optuna_cfg = config_manager.get_optuna_config()
    wandb_cfg = config_manager.get_wandb_config()

    # Set up a dedicated Wandb project for tuning
    os.environ["WANDB_PROJECT"] = f"{wandb_cfg.project}-tuning"
    if wandb_cfg.entity:
        os.environ["WANDB_ENTITY"] = wandb_cfg.entity

    # --- Select Stage and Run Study ---
    if args.stage == "dapt":
        study_name = "gemma-dapt-tuning"
        n_trials = optuna_cfg.dapt_n_trials
        
        # Pass base configs to objective function
        objective_func = lambda trial: objective_dapt(
            trial,
            config_manager.get_model_config(),
            config_manager.get_dapt_config(),
            config_manager.new_tokens,
        )
        direction = "minimize" # Minimize training loss

    elif args.stage == "sft":
        study_name = "gemma-sft-tuning"
        n_trials = optuna_cfg.sft_n_trials
        
        objective_func = lambda trial: objective_sft(
            trial,
            config_manager.get_model_config(),
            config_manager.get_sft_config(),
        )
        direction = "minimize" # Minimize evaluation loss
        
    else:
        raise ValueError("Invalid stage selected.")

    print(f"--- Starting Optuna study '{study_name}' for stage '{args.stage}' ---")
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=optuna_cfg.storage,
        load_if_exists=True, # Allows you to resume a study
    )
    
    study.optimize(objective_func, n_trials=n_trials)

    print("\n--- Tuning Complete ---")
    print(f"Study '{study_name}' statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value ({'eval_loss' if args.stage == 'sft' else 'train_loss'}): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")