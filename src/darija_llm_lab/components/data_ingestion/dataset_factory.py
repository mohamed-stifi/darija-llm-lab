from typing import Dict, Type, Optional
from darija_llm_lab.components.data_ingestion.dataset_strategy import (
    DatasetStrategy,
    DarijaPattern,
    AudioData, 
    DODaAudio, 
    FilteredSamples,
    WikipediaDarija, 
    IADDDarija, 
    OpenaiGsm8kDarija,
    DarijaBanking
)


class DatasetStrategyFactory:
    """Factory class for creating dataset strategy instances."""
    
    # Registry of available strategies
    _strategies: Dict[str, Type[DatasetStrategy]] = {
        "darija_pattern": DarijaPattern,
        "audio_data": AudioData,
        "doda_audio": DODaAudio,
        "filtered_samples": FilteredSamples,
        "wikipedia_darija": WikipediaDarija,
        "iadd_darija": IADDDarija,
        "openai_gsm8k_darija": OpenaiGsm8kDarija,
        "darija_banking": DarijaBanking,
    }
    
    @classmethod
    def create(cls, strategy_type: str, path: str, name: Optional[str] = None) -> DatasetStrategy:
        """
        Create a dataset strategy instance.
        
        Args:
            strategy_type (str): The type of strategy to create
            path (str): Path to the dataset
            name (str, optional): Custom name for the strategy. If None, uses default.
            
        Returns:
            DatasetStrategy: An instance of the requested strategy
            
        Raises:
            ValueError: If strategy_type is not supported
        """
        if strategy_type not in cls._strategies:
            available_strategies = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy type: '{strategy_type}'. "
                f"Available strategies: {available_strategies}"
            )
        
        strategy_class = cls._strategies[strategy_type]
        
        # Create instance with custom name if provided, otherwise use default
        if name is not None:
            return strategy_class(path=path, name=name)
        else:
            return strategy_class(path=path)
    
    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available strategy types.
        
        Returns:
            list[str]: List of available strategy type names
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[DatasetStrategy]) -> None:
        """
        Register a new strategy type.
        
        Args:
            name (str): Name for the strategy type
            strategy_class (Type[DatasetStrategy]): The strategy class to register
        """
        if not issubclass(strategy_class, DatasetStrategy):
            raise ValueError("Strategy class must inherit from DatasetStrategy")
        
        cls._strategies[name] = strategy_class
    
    @classmethod
    def unregister_strategy(cls, name: str) -> None:
        """
        Unregister a strategy type.
        
        Args:
            name (str): Name of the strategy type to remove
        """
        if name in cls._strategies:
            del cls._strategies[name]


# Example usage:
if __name__ == "__main__":
    # Create different dataset strategies using the factory
    
    # Create a Darija pattern strategy
    darija_strategy = DatasetStrategyFactory.create(
        strategy_type="darija_pattern",
        path="/path/to/darija/data.txt",
        name="My Darija Dataset"
    )
    
    # Create an audio data strategy with default name
    audio_strategy = DatasetStrategyFactory.create(
        strategy_type="audio_data",
        path="/path/to/audio/data.csv"
    )
    
    # Create a DODa audio strategy
    doda_strategy = DatasetStrategyFactory.create(
        strategy_type="doda_audio",
        path="/path/to/doda/dataset"
    )
    
    # Get available strategies
    available = DatasetStrategyFactory.get_available_strategies()
    print(f"Available strategies: {available}")
    
    # Load datasets for tokenizer training
    darija_texts = darija_strategy.load_dataset_for_train_tokenizer()
    audio_texts = audio_strategy.load_dataset_for_train_tokenizer()
    
    print(f"Darija strategy name: {darija_strategy.get_strategy_name()}")
    print(f"Audio strategy name: {audio_strategy.get_strategy_name()}")