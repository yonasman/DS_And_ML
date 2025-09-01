# src/embedding.py
import os
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = Client()

# Load HuggingFace embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



# Free, local Hugging Face embeddings
embeddings_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create or get Chroma collection
collection = client.get_or_create_collection(
    name="venturai_docs",
    embedding_function=embeddings_fn
)

def add_documents_to_chroma(docs):
    """
    Add documents to ChromaDB using HuggingFace embeddings
    """

    ids = [f"doc_{i}" for i in range(len(docs))]
    collection.add(
        documents=docs,
        ids=ids
    )
    print(f"Added {len(docs)} documents to ChromaDB using HuggingFace embeddings")



