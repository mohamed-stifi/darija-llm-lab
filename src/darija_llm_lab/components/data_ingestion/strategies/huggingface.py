from darija_llm_lab.components.data_ingestion.strategies.base import DatasetStrategy
from datasets import load_dataset

class DODaAudio(DatasetStrategy):
    def __init__(self, path, name = "DODa-audio"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')


    def load_dataset_for_train_tokenizer(self):
        text_lines = list(self.dataset['darija_Arab_new'])
        return text_lines
    
    def get_strategy_name(self):
        return self.name
    
class FilteredSamples(DatasetStrategy):
    def __init__(self, path, name = "filtered samples"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')

    def load_dataset_for_train_tokenizer(self):
        return list(self.dataset['transcription'])
    
    def get_strategy_name(self):
        return self.name

class WikipediaDarija(DatasetStrategy):
    def __init__(self, path, name = "wekipedia darija"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')
    def load_dataset_for_train_tokenizer(self):
        return list(self.dataset['text'])
    
    def get_strategy_name(self):
        return self.name
    

class IADDDarija(DatasetStrategy):
    def __init__(self, path, name = "IADD darija"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')
    def load_dataset_for_train_tokenizer(self):
        return list(self.dataset['text'])
    
    def get_strategy_name(self):
        return self.name
    
class OpenaiGsm8kDarija(DatasetStrategy):
    def __init__(self, path, name = "openai gsm8k test darija"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')
    def load_dataset_for_train_tokenizer(self):
        return list(self.dataset['question'])
    
    def get_strategy_name(self):
        return self.name


class DarijaBanking(DatasetStrategy):
    def __init__(self, path, name = "openai gsm8k test darija"):
        super().__init__()
        self.path = path
        self.name = name

        self.dataset = load_dataset(self.path, split= 'train')
    def load_dataset_for_train_tokenizer(self):
        return list(self.dataset['text'])
    
    def get_strategy_name(self):
        return self.name