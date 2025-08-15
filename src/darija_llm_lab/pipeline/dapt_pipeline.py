# src/pipeline/dapt_pipeline.py
from darija_llm_lab.utils.common import ConfigurationManager
from darija_llm_lab.components.model_loader import ModelLoader
from darija_llm_lab.components.data_ingestion import DataIngestion
from darija_llm_lab.components.model_trainer import ModelTrainer

class DAPTPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        # 1. Get configs
        model_cfg = self.config_manager.get_model_config()
        dapt_cfg = self.config_manager.get_dapt_config()
        new_tokens = self.config_manager.new_tokens

        # 2. Load and prepare model
        model_loader = ModelLoader(model_cfg, new_tokens)
        model, tokenizer = model_loader.load_model_and_tokenizer()
        model, tokenizer = model_loader.perform_embedding_surgery()
        model = model_loader.prepare_model_for_dapt()

        # 3. Load data
        data_ingestion = DataIngestion()
        train_dataset = data_ingestion.load_dapt_dataset(dapt_cfg.data, tokenizer)

        # 4. Train
        trainer_component = ModelTrainer()
        trainer = trainer_component.get_dapt_trainer(model, tokenizer, train_dataset, dapt_cfg.trainer)
        trainer_component.train(trainer)

        # 5. Save final model
        print(f"--- Saving DAPT-adapted model to {dapt_cfg.output_model_path} ---")
        model.save_pretrained(str(dapt_cfg.output_model_path))
        tokenizer.save_pretrained(str(dapt_cfg.output_model_path))
        print("✅ DAPT pipeline finished successfully.")