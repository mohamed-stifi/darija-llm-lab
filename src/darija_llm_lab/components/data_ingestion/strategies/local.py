from darija_llm_lab.components.data_ingestion.strategies.base import DatasetStrategy
from typing import List
import pandas as pd

class DarijaPattern(DatasetStrategy):
    def __init__(self, path, name = "darija pattern"):
        self.text_lines = []
        self.wav_files = []
        self.name = name

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                # Each line is structured as: "filename" "text"
                parts = line.strip().split('" "')
                if len(parts) == 2:
                    # Remove the surrounding quotes
                    text = parts[1].rstrip('"')  # right quote only
                    self.text_lines.append(text)

                    wav = parts[0].rstrip('"')
                    self.wav_files.append(wav)

        

    def load_dataset_for_train_tokenizer(self) -> List[str]:
        return self.text_lines
    
    def get_strategy_name(self):
        return self.name

class AudioData(DatasetStrategy):
    def __init__(self, path, name = "audion data"):
        super().__init__()
        self.path = path
        self.name = name

        data = pd.read_csv(self.path)

        self.text_lines = data["transcription"].to_list()
        self.wav_files = data["audio"].to_list()

    def load_dataset_for_train_tokenizer(self) -> List[str]:
        return self.text_lines
    
    def get_strategy_name(self):
        return self.name 
