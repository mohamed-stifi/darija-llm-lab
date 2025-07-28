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