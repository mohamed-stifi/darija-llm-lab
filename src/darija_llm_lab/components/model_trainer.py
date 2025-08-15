import torch
import transformers
from trl import SFTTrainer, SFTConfig, Trainer, TrainingArguments
from unsloth.chat_templates import train_on_responses_only

class ModelTrainer:
    def get_dapt_trainer(self, model, tokenizer, train_dataset, config):
        """Initializes a basic Trainer for DAPT."""
        training_args = SFTConfig(
            dataset_text_field = "text",
            output_dir=str(config.output_dir),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            ## add for eval
            # eval_strategy = "steps",
            # eval_steps= 10,
            # save_strategy="steps",
            save_steps=config.save_steps,
            
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,

            optim = config.optim,
            weight_decay= config.weight_decay, 
            warmup_steps=config.warmup_steps,

            lr_scheduler_type = "linear",
            seed=config.seed,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="wandb",
            # metric_for_best_model = "eval_loss",   # or any metric you're computing
            load_best_model_at_end = True,
            # greater_is_better = False, 
        )

        trainer = SFTTrainer(
                    model = model,
                    tokenizer = tokenizer,
                    train_dataset = train_dataset,
                    args=training_args
                    # eval_dataset = eval_dataset, # Can set up evaluation!
                    #callbacks=[early_stopping_callback],
                    )
        # Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)


        return trainer

    def get_sft_trainer(self, model, tokenizer, train_ds, eval_ds, config):
        """Initializes the SFTTrainer with evaluation and response masking."""
        early_stopping = transformers.EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=0.0,)
        
        sft_args = SFTConfig(
                dataset_text_field = "text",
                output_dir=str(config.output_dir),
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            

                ## add for eval
                eval_strategy = "steps",
                eval_steps= 10,
                save_strategy="steps",
                save_steps=config.save_steps,
                #load_best_model_at_end=True, 
                optim = config.optim,
                weight_decay= config.weight_decay, 
                warmup_steps=config.warmup_steps,

                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                logging_steps=config.logging_steps,
                
                
                lr_scheduler_type = "linear",
                seed = 3407,
                

                ### For EarlyStopping :
                bf16 = torch.cuda.is_bf16_supported(),
                fp16 = not torch.cuda.is_bf16_supported(),
                
                metric_for_best_model = "eval_loss",   # or any metric you're computing
                load_best_model_at_end = True,
                greater_is_better = False,
                report_to="wandb", 
            )
        trainer = SFTTrainer(model=model,
                            tokenizer=tokenizer,
                            train_dataset=train_ds,
                            eval_dataset=eval_ds,
                            callbacks=[early_stopping],
                            args=sft_args)
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )
        return trainer

    def train(self, trainer):
        print("--- Starting Training ---")
        result = trainer.train()
        print("--- Training Complete ---")
        return result