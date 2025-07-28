from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
from darija_llm_lab.constants import CONFIG_PATH
from darija_llm_lab.components.data_ingestion.dataset_factory import DatasetStrategyFactory
from darija_llm_lab.entity.entity import DataIngestionConfig
from darija_llm_lab.utils.utils import create_directories, read_yaml


class ConfigurationManager:
    """Manager class for handling configuration loading and validation."""
    
    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialize ConfigurationManager with config file path.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.config = read_yaml(config_path)

        create_directories([
            Path(self.config.artifacts_root)
        ])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Load data ingestion configuration from YAML file.
        
        Returns:
            DataIngestionConfig: Configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            KeyError: If required configuration keys are missing
        """
        data_ingestion_config = self.config.data_ingestion

        create_directories([
            Path(data_ingestion_config.output_path)
        ])
        
        
        
        return DataIngestionConfig(
            datasets=data_ingestion_config.get('datasets', []),
            output_path=data_ingestion_config.get('output_path', "data/processed"),
            combined_filename=data_ingestion_config.get('combined_filename', "combined_dataset.json"),
            metadata_filename=data_ingestion_config.get('metadata_filename', "dataset_metadata.json"),
            save_individual_datasets=data_ingestion_config.get('save_individual_datasets', True),
            text_column_name=data_ingestion_config.get('text_column_name', "text"),
            source_column_name=data_ingestion_config.get('source_column_name', "source"),
            remove_duplicates=data_ingestion_config.get('remove_duplicates', True),
            min_text_length=data_ingestion_config.get('min_text_length', 1),
            max_text_length=data_ingestion_config.get('max_text_length', None),

            # Hugging Face configuration
            push_to_hub=data_ingestion_config.get('push_to_hub', False),
            hub_dataset_name=data_ingestion_config.get('hub_dataset_name'),
            hub_organization=data_ingestion_config.get('hub_organization'),
            hub_private=data_ingestion_config.get('hub_private', False),
            hub_token=data_ingestion_config.get('hub_token')
        )
            
    