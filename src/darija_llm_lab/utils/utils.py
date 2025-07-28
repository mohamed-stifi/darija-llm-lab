import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as file:
            content = yaml.safe_load(file)
            logger.info(
                f"yaml file: {path_to_yaml} loaded successfully"
            )
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def create_directories(path_list: List[Path]) -> None:
    """
    Create directories from a list of paths.
    
    Args:
        path_list: List of paths to create
    """
    for path in path_list:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    """
    Get human readable size of a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        Human readable size string
    """
    if not path.exists():
        return "0 B"
    
    size_bytes = path.stat().st_size if path.is_file() else sum(
        f.stat().st_size for f in path.rglob('*') if f.is_file()
    )
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"



def save_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        path: Path to save file
        data: List of dictionaries to save
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"JSONL file saved: {path} with {len(data)} records")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        path: Path to load file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    logger.info(f"JSONL file loaded: {path} with {len(data)} records")
    return data


def get_current_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_text_content(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validate text content.
    
    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        True if text is valid
    """
    if not isinstance(text, str):
        return False
    
    text = text.strip()
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Check for suspicious patterns
    if text.count(' ') == 0 and len(text) > 50:  # Very long text without spaces
        return False
        
    return True


