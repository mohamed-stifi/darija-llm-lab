from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class DatasetStrategy(ABC):
    """Abstract base class for datasets strategies."""
    
    @abstractmethod
    def load_dataset_for_train_tokenizer(self) -> List[str]:
        """
        load dataset for train the tokenizer
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the strategy."""
        pass