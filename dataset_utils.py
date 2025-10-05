import json
import os
import re
from typing import List, Dict

AMAZON_TRN_PATH = "dataset/LF-Amazon-1.3M/trn.json"
MAX_CONTENT_LENGTH = 1000   # Truncar descrições muito longas
OUTPUT_DIR = "finetune_chunks"

def load_amazon_trn(path=AMAZON_TRN_PATH):
    """
    Load the Amazon Titles dataset from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file if line.strip()]
    return data

def clean_content(content: str) -> str:
    """Limpa, remove espaços extras e corta conteúdos muito longos."""
    content = sanitize_text(content)
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH].rsplit(" ", 1)[0] + "..."
    return content

def save_chunk(chunk_data: List[Dict], chunk_index: int) -> None:
    """Salva um bloco de produtos em um arquivo JSONL."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index:03d}.jsonl")

    with open(chunk_path, "w", encoding="utf-8") as f:
        for example in chunk_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunk_data)} products -> {chunk_path}")

def sanitize_text(text: str) -> str:
    """Remove ou escapa caracteres problemáticos."""
    if not text:
        return ""
    
    # Remove caracteres de controle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)    
    
    # Remove quebras de linha problemáticas dentro de strings
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    
    # Remove espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Escapa aspas
    # text = text.replace('"', '\\"').replace("'", "\\'")
    
    return text.strip()