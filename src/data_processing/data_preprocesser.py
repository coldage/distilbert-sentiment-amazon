"""
===========================================
PART 1: TEXT PREPROCESSING
===========================================
This module handles text preprocessing for Amazon reviews:
- Remove HTML tags
- Remove URL links
- Convert to lowercase
- Remove special symbols (keep English punctuation)
"""

import re
from typing import List


def preprocess_text(text: str) -> str:
    """
    Preprocess a single text according to task requirements:
    1. Remove HTML tags (e.g., <br>)
    2. Remove URL links (e.g., https://amzn.to/xxx)
    3. Convert to lowercase (required for DistilBERT-base-uncased)
    4. Remove special symbols, keep English punctuation (.,?!)
    
    Args:
        text: Raw review text
        
    Returns:
        Preprocessed text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URL links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special symbols but keep English punctuation and alphanumeric
    # Keep: letters, numbers, spaces, and basic punctuation (.,?!)
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Preprocess a list of texts.
    
    Args:
        texts: List of raw review texts
        
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(text) for text in texts]