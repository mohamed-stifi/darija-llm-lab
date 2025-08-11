# src/pipeline/sft_pipeline.py
from darija_llm_lab.utils.common import ConfigurationManager
from darija_llm_lab.components.model_loader import ModelLoader
from darija_llm_lab.components.data_ingestion import DataIngestion
from darija_llm_lab.components.model_trainer import ModelTrainer

class SFTPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        # 1. Get configs
        model_cfg = self.config_manager.get_model_config()
        sft_cfg = self.config_manager.get_sft_config()
        
        # 2. Load DAPT model and apply SFT PEFT
        model_loader = ModelLoader(model_cfg)
        model, tokenizer = model_loader.load_model_and_tokenizer(model_path=str(sft_cfg.input_model_path))
        model = model_loader.apply_sft_peft(sft_cfg.peft)

        # 3. Load SFT data
        data_ingestion = DataIngestion()
        train_ds, eval_ds = data_ingestion.load_sft_dataset(sft_cfg.data, tokenizer)
        
        # 4. Train
        trainer_component = ModelTrainer()
        trainer = trainer_component.get_sft_trainer(model, tokenizer, train_ds, eval_ds, sft_cfg.trainer)
        trainer_component.train(trainer)

        # 5. Save final adapters
        print(f"--- Saving SFT LoRA adapters to {sft_cfg.output_adapters_path} ---")
        model.save_pretrained(str(sft_cfg.output_adapters_path))
        print("✅ SFT pipeline finished successfully.")