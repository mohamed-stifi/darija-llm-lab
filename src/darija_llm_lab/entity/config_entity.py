from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datasets import Dataset

@dataclass
class DatasetInfo:
    """Information about a single dataset."""
    name: str
    strategy_type: str
    path: str
    custom_name: Optional[str] = None
    text_count: int = 0
    total_characters: int = 0
    avg_text_length: float = 0.0
    unique_texts: int = 0
    processed_at: Optional[str] = None


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion process."""
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    output_path: str = "artifacts/data_ingestion"
    combined_filename: str = "combined_dataset.json"
    metadata_filename: str = "dataset_metadata.json"
    save_individual_datasets: bool = False
    text_column_name: str = "text"
    source_column_name: str = "source"
    remove_duplicates: bool = True
    min_text_length: int = 1
    max_text_length: Optional[int] = None

    # Hugging Face configuration
    push_to_hub: bool = False
    hub_dataset_name: Optional[str] = None
    hub_organization: Optional[str] = None
    hub_private: bool = False
    hub_token: Optional[str] = None



@dataclass
class DataIngestionEntity:
    """Entity representing the result of data ingestion."""
    combined_texts: List[str] = field(default_factory=list)
    dataset_info: List[DatasetInfo] = field(default_factory=list)
    total_texts: int = 0
    total_characters: int = 0
    unique_datasets: int = 0
    processing_time: Optional[float] = None
    config: Optional[DataIngestionConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    huggingface_dataset: Optional[Dataset] = None
    hub_url: Optional[str] = None



@dataclass(frozen=True)
class ModelConfig:
    base_model_id: str
    max_seq_length: int
    load_in_4bit: bool

@dataclass(frozen=True)
class DAPTDataConfig:
    dataset_id: str
    text_column: str
    debug_mode: bool
    debug_split_size: int

@dataclass(frozen=True)
class DAPTTrainerConfig:
    output_dir: Path
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    warmup_steps: int
    seed: int
    optim: str
    weight_decay: float

@dataclass(frozen=True)
class DAPTConfig:
    output_model_path: Path
    data: DAPTDataConfig
    trainer: DAPTTrainerConfig

@dataclass(frozen=True)
class SFTDataConfig:
    dataset_id: str
    test_size: float
    seed: int
    debug_mode: bool
    debug_split_size: int

@dataclass(frozen=True)
class SFTPEFTConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    random_state: int
    target_modules: List[str]

@dataclass(frozen=True)
class SFTTrainerConfig:
    output_dir: Path
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    warmup_steps: int
    max_steps: int
    learning_rate: float
    logging_steps: int
    seed: int
    metric_for_best_model: str
    load_best_model_at_end: bool
    greater_is_better: bool
    early_stopping_patience: int
    optim: str
    weight_decay: float

@dataclass(frozen=True)
class SFTConfig:
    input_model_path: Path
    output_adapters_path: Path
    data: SFTDataConfig
    peft: SFTPEFTConfig
    trainer: SFTTrainerConfig

@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: Optional[str]

@dataclass(frozen=True)
class OptunaConfig:
    dapt_n_trials: int
    sft_n_trials: int
    storage: str