from langchain_groq import ChatGroq
import os

# api_key
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=GROQ_API_KEY)

def ask_groq(user_query, context=""):
    """
    Simple Groq call. Optionally add context.
    """
    prompt = f"{context}\n\nUser: {user_query}" if context else user_query
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Request failed: {e}"
