import os
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = Client()

# Load HuggingFace embedding model
hf_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Free, local Hugging Face embeddings
embeddings_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Create or get Chroma collection
collection = client.get_or_create_collection(
    name="foundrai_docs",
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


def search_documents(query,threshold=0.6, top_k=3):

    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents','distances']
    )

    results = []

    for doc,dist in zip(res["documents"][0],res["distances"][0]):
        similarity = 1 - dist

        if similarity >= threshold:
            results.append((doc,similarity))
    
    if results:
        print(f"🔎 Query: {query}")
        for doc,score in results:
            print(f"{doc} (similarity {score:.2f})")
    else:
        print(f"❌ No relevant results found for: {query}")
