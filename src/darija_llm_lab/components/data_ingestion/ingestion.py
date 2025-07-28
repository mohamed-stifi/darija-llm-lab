from darija_llm_lab.entity.entity import DataIngestionConfig, DatasetInfo, DataIngestionEntity
from darija_llm_lab.components.data_ingestion.dataset_factory import DatasetStrategyFactory
import logging
from typing import List, Any, Dict, Optional, Tuple
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter
from datasets import Dataset
from huggingface_hub import HfApi, login

class DataIngestion:
    """Main class for data ingestion process."""
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion with configuration.
        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config
        self.logger = self._setup_logging()
        self.metadata: Dict[str, Any] = {}
        self.dataset_info: List[DatasetInfo] = []
        self.processing_time: float = 0.0

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the ingestion process."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _filter_texts(self, texts: List[str]) -> List[str]:
        """
        Filter texts based on configuration criteria.
        Args:
            texts (List[str]): List of texts to filter
        Returns:
            List[str]: Filtered texts
        """
        filtered_texts = []
        for text in texts:
            if not isinstance(text, str):
                continue
            text = text.strip()
            # Check minimum length
            if len(text) < self.config.min_text_length:
                continue
            # Check maximum length if specified
            if self.config.max_text_length and len(text) > self.config.max_text_length:
                continue
            filtered_texts.append(text)
        return filtered_texts

    def _process_dataset(self, dataset_config: Dict[str, Any]) -> Tuple[DatasetInfo, List[str]]:
        """
        Process a single dataset and return its information.
        Args:
            dataset_config (Dict[str, Any]): Configuration for the dataset
        Returns:
            Tuple[DatasetInfo, List[str]]: Information about the processed dataset and filtered texts
        """
        strategy_type = dataset_config['strategy_type']
        path = dataset_config['path']
        custom_name = dataset_config.get('name')
        self.logger.info(f"Processing dataset: {strategy_type} from {path}")
        
        # Create strategy using factory
        strategy = DatasetStrategyFactory.create(
            strategy_type=strategy_type,
            path=path,
            name=custom_name
        )
        
        # Load texts
        texts = strategy.load_dataset_for_train_tokenizer()
        
        # Filter texts
        filtered_texts = self._filter_texts(texts)
        
        # Calculate statistics
        text_count = len(filtered_texts)
        total_characters = sum(len(text) for text in filtered_texts)
        avg_text_length = total_characters / text_count if text_count > 0 else 0
        unique_texts = len(set(filtered_texts))
        
        dataset_info = DatasetInfo(
            name=strategy.get_strategy_name(),
            strategy_type=strategy_type,
            path=path,
            custom_name=custom_name,
            text_count=text_count,
            total_characters=total_characters,
            avg_text_length=avg_text_length,
            unique_texts=unique_texts,
            processed_at=datetime.now().isoformat()
        )
        
        self.logger.info(f"Dataset {dataset_info.name}: {text_count} texts, {total_characters} characters")
        return dataset_info, filtered_texts

    def _save_individual_dataset(self, dataset_info: DatasetInfo, texts: List[str]):
        """Save individual dataset to file."""
        if not self.config.save_individual_datasets:
            return
            
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from dataset name
        safe_name = "".join(c for c in dataset_info.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = f"{safe_name}_dataset.json"
        
        dataset_data = {
            "metadata": {
                "name": dataset_info.name,
                "strategy_type": dataset_info.strategy_type,
                "path": dataset_info.path,
                "text_count": dataset_info.text_count,
                "total_characters": dataset_info.total_characters,
                "processed_at": dataset_info.processed_at
            },
            "texts": texts
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved individual dataset to: {filepath}")

    def _create_huggingface_dataset(self, combined_data: List[Dict[str, str]]) -> Dataset:
        """
        Create a Hugging Face Dataset from the combined data.
        Args:
            combined_data (List[Dict[str, str]]): Combined dataset with text and source
        Returns:
            Dataset: Hugging Face Dataset object
        """
        self.logger.info("Creating Hugging Face Dataset")
        try:
            # Convert to pandas DataFrame first for easier manipulation
            df = pd.DataFrame(combined_data)
            # Create Hugging Face Dataset
            dataset = Dataset.from_pandas(df)
            self.logger.info(f"Created Hugging Face Dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            self.logger.error(f"Error creating Hugging Face Dataset: {e}")
            raise

    def _push_to_huggingface_hub(self, dataset: Dataset) -> str:
        """
        Push the dataset to Hugging Face Hub.
        Args:
            dataset (Dataset): The dataset to push
        Returns:
            str: URL of the uploaded dataset
        Raises:
            ValueError: If required configuration is missing
            Exception: If upload fails
        """
        if not self.config.hub_dataset_name:
            raise ValueError("hub_dataset_name is required to push to Hugging Face Hub")
            
        self.logger.info("Preparing to push dataset to Hugging Face Hub")
        try:
            # Login to Hugging Face if token is provided
            if self.config.hub_token:
                login(token=self.config.hub_token)
                self.logger.info("Logged in to Hugging Face Hub with provided token")
            
            # Construct the dataset repository name
            if self.config.hub_organization:
                repo_id = f"{self.config.hub_organization}/{self.config.hub_dataset_name}"
            else:
                repo_id = self.config.hub_dataset_name
            
            # Create dataset card content
            dataset_card_content = self._create_dataset_card()
            
            # Push dataset to hub
            self.logger.info(f"Pushing dataset to {repo_id}")
            dataset.push_to_hub(
                repo_id=repo_id,
                private=self.config.hub_private,
                commit_message=f"Add Darija dataset - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Create and push dataset card
            api = HfApi()
            try:
                api.upload_file(
                    path_or_fileobj=dataset_card_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add dataset card"
                )
                self.logger.info("Dataset card uploaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to upload dataset card: {e}")
            
            hub_url = f"https://huggingface.co/datasets/{repo_id}"
            self.logger.info(f"Dataset successfully pushed to: {hub_url}")
            return hub_url
        except Exception as e:
            self.logger.error(f"Error pushing to Hugging Face Hub: {e}")
            raise

    def _create_dataset_card(self) -> str:
        """
        Create a dataset card (README.md) content for the Hugging Face dataset.
        Returns:
            str: Dataset card content in markdown format
        """
        # Get source distribution from metadata
        source_dist = self.metadata.get('source_distribution', {})
        source_info = "\n".join([f"- **{source}**: {count:,} texts" for source, count in source_dist.items()])
        
        # Get dataset info for detailed breakdown
        dataset_details = "\n".join([
            f"- **{info.name}**: {info.text_count:,} texts, {info.total_characters:,} characters (avg: {info.avg_text_length:.1f})"
            for info in self.dataset_info
        ])
        
        total_texts = self.metadata.get('total_texts', 0)
        total_characters = self.metadata.get('total_characters', 0)
        avg_length = total_characters / total_texts if total_texts > 0 else 0
        
        card_content = f"""---
language:
- ar
- ary
multilinguality:
- multilingual
size_categories:
- {self._get_size_category()}
task_categories:
- text-generation
- text-classification
pretty_name: {self.config.hub_dataset_name or 'Darija Dataset'}
tags:
- darija
- moroccan-arabic
- arabic
- north-african
---
# {self.config.hub_dataset_name or 'Darija Dataset'}
## Dataset Description
This dataset contains Darija (Moroccan Arabic) text data collected from multiple sources and processed for machine learning tasks.

### Dataset Summary
- **Total texts**: {total_texts:,}
- **Total characters**: {total_characters:,}
- **Average text length**: {avg_length:.1f} characters
- **Number of unique datasets**: {len(self.dataset_info)}
- **Created**: {self.metadata.get('created_at', 'Unknown')}
- **Processing time**: {self.processing_time:.2f} seconds

### Source Distribution
{source_info}

### Dataset Composition
{dataset_details}

### Data Fields
- `{self.config.text_column_name}`: The text content in Darija
- `{self.config.source_column_name}`: The source dataset name

### Data Processing
The dataset was processed with the following configuration:
- **Minimum text length**: {self.config.min_text_length} characters
- **Maximum text length**: {self.config.max_text_length or 'No limit'}
- **Duplicates removed**: {'Yes' if self.config.remove_duplicates else 'No'}

### Usage
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self.config.hub_organization + '/' if self.config.hub_organization else ''}{self.config.hub_dataset_name}")

# Access the data
for example in dataset['train']:
    print(f"Text: {{example['{self.config.text_column_name}']}}")
    print(f"Source: {{example['{self.config.source_column_name}']}}")
```

### Languages
This dataset primarily contains:
- **Darija** (Moroccan Arabic) - ary
- **Modern Standard Arabic** - ar

### Citation
If you use this dataset, please cite:
```bibtex
@dataset{{darija_dataset,
    title={{Darija Dataset}},
    author={{Darija LLM Lab}},
    year={{2024}},
    url={{https://huggingface.co/datasets/{self.config.hub_organization + '/' if self.config.hub_organization else ''}{self.config.hub_dataset_name}}}
}}
```

### License
Please refer to the individual source datasets for their respective licenses.
"""
        return card_content

    def _get_size_category(self) -> str:
        """Get the appropriate size category for the dataset."""
        total_texts = self.metadata.get('total_texts', 0)
        if total_texts < 1000:
            return "n<1K"
        elif total_texts < 10000:
            return "1K<n<10K"
        elif total_texts < 100000:
            return "10K<n<100K"
        elif total_texts < 1000000:
            return "100K<n<1M"
        else:
            return "1M<n<10M"

    def _combine_datasets(self, all_texts: List[List[str]], dataset_infos: List[DatasetInfo]) -> List[Dict[str, str]]:
        """
        Combine all datasets into a single list with source information.
        Args:
            all_texts (List[List[str]]): List of text lists from each dataset
            dataset_infos (List[DatasetInfo]): Information about each dataset
        Returns:
            List[Dict[str, str]]: Combined dataset with source information
        """
        combined_data = []
        for texts, dataset_info in zip(all_texts, dataset_infos):
            for text in texts:
                combined_data.append({
                    self.config.text_column_name: text,
                    self.config.source_column_name: dataset_info.name
                })
        
        # Remove duplicates if configured
        if self.config.remove_duplicates:
            seen_texts = set()
            unique_data = []
            for item in combined_data:
                text = item[self.config.text_column_name]
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_data.append(item)
            self.logger.info(f"Removed {len(combined_data) - len(unique_data)} duplicate texts")
            combined_data = unique_data
        
        return combined_data

    def _save_combined_data(self, combined_data: List[Dict[str, str]], metadata: Dict[str, Any]):
        """Save combined dataset and metadata to files."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined dataset
        combined_filepath = output_dir / self.config.combined_filename
        with open(combined_filepath, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata_filepath = output_dir / self.config.metadata_filename
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved combined dataset to: {combined_filepath}")
        self.logger.info(f"Saved metadata to: {metadata_filepath}")

    def ingest_data(self) -> DataIngestionEntity:
        """
        Main method to ingest all configured datasets.
        Returns:
            DataIngestionEntity: Result of the ingestion process
        """
        start_time = datetime.now()
        self.logger.info("Starting data ingestion process")
        
        all_texts = []
        dataset_infos = []
        
        # Process each configured dataset
        for dataset_config in self.config.datasets:
            try:
                dataset_info, texts = self._process_dataset(dataset_config)
                dataset_infos.append(dataset_info)
                all_texts.append(texts)
                
                # Save individual dataset if configured
                self._save_individual_dataset(dataset_info, texts)
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_config}: {e}")
                continue
        
        # Combine all datasets
        combined_data = self._combine_datasets(all_texts, dataset_infos)
        combined_texts = [item[self.config.text_column_name] for item in combined_data]
        
        # Calculate overall statistics
        total_texts = len(combined_texts)
        total_characters = sum(len(text) for text in combined_texts)
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_time = processing_time
        
        # Create metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "total_datasets": len(dataset_infos),
            "total_texts": total_texts,
            "total_characters": total_characters,
            "average_text_length": total_characters / total_texts if total_texts > 0 else 0,
            "config": {
                "remove_duplicates": self.config.remove_duplicates,
                "min_text_length": self.config.min_text_length,
                "max_text_length": self.config.max_text_length
            },
            "source_distribution": dict(Counter(item[self.config.source_column_name] for item in combined_data))
        }
        self.metadata = metadata
        self.dataset_info = dataset_infos
        
        # Save combined dataset and metadata
        self._save_combined_data(combined_data, metadata)
        
        # Create Hugging Face Dataset
        hf_dataset = self._create_huggingface_dataset(combined_data)
        hub_url = None
        
        # Push to Hugging Face Hub if configured
        if self.config.push_to_hub:
            try:
                hub_url = self._push_to_huggingface_hub(hf_dataset)
                self.logger.info(f"Dataset successfully uploaded to Hugging Face Hub: {hub_url}")
            except Exception as e:
                self.logger.error(f"Failed to push to Hugging Face Hub: {e}")
                # Continue execution even if hub upload fails
        
        # Create result entity
        entity = DataIngestionEntity(
            combined_texts=combined_texts,
            dataset_info=dataset_infos,
            total_texts=total_texts,
            total_characters=total_characters,
            unique_datasets=len(dataset_infos),
            processing_time=processing_time,
            config=self.config,
            metadata=metadata,
            huggingface_dataset=hf_dataset,
            hub_url=hub_url
        )
        
        self.logger.info(f"Data ingestion completed in {processing_time:.2f} seconds")
        self.logger.info(f"Total: {total_texts} texts from {len(dataset_infos)} datasets")
        return entity
