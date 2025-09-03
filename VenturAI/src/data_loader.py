import os
import glob
import config

def load_text_files(directory=config.RAW_DATA_DIR):
    """
    Load all .txt files from the raw data directory.
    Returns a list of raw text strings.
    """
    files = glob.glob(os.path.join(directory, "*.txt"))
    docs = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def clean_text(text: str) -> str:
    """
    Basic cleaning: strip whitespace and normalize newlines.
    """
    return text.strip().replace("\n", " ")

def preprocess_docs():
    """
    Load and clean txt files, then save to processed folder.
    """
    docs = load_text_files()
    cleaned_docs = [clean_text(doc) for doc in docs]

    # Make sure processed folder exists
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Save cleaned files
    for i, doc in enumerate(cleaned_docs):
        out_path = os.path.join(config.PROCESSED_DATA_DIR, f"doc_{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc)

    return cleaned_docs