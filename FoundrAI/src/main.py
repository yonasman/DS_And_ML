from data_loader import preprocess_docs
from embedding import add_documents_to_chroma,collection
from sklearn.metrics.pairwise import cosine_similarity
from embedding import hf_model


def run():
    print("FoundrAI starting...")
    docs = preprocess_docs()
    print(f"Loaded & processed {len(docs)} documents.")

    if docs:
        print("Adding documents to ChromaDB...")
        print("Documents embedded!")
    res = collection.get(include=['embeddings','documents'])



if __name__== '__main__':
    run()