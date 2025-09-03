from data_loader import preprocess_docs
from embedding import add_documents_to_chroma, collection, search_documents
from groq_client import ask_groq

def run():
    print("VenturAI starting...")
    docs = preprocess_docs()
    print(f"Loaded & processed {len(docs)} documents.")

    if docs:
        print("Adding documents to ChromaDB...")
        add_documents_to_chroma(docs)
        print("Documents embedded!")

    res = collection.get(include=['embeddings', 'documents'])
    print("ChromaDB ready.")

# Universal query function for any language
def query_rag(user_query, top_k=1):
    results = collection.query(
        query_texts=[user_query],
        n_results=top_k
    )

    retrieved_docs = results['documents'][0]
    context = "\n".join(retrieved_docs) if retrieved_docs else None

    # Ask Groq using the retrieved context
    answer = ask_groq(user_query, context)
    return answer, context

if __name__ == "__main__":
    run()
    # Example English query
    query_en = "What's VenturAI"
    answer_en, context_en = query_rag(query_en)
    print("Answer (EN):", answer_en.content)
    # print("Context (EN):", context_en)

    print('\n\n*****************************************\n\n')
    
    # Example Amharic query
    query_am = "ፈንደርኤይ ምንድነው?"
    answer_am, context_am = query_rag(query_am)
    print("Answer (AM):", answer_am.content)

