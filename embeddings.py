from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_and_chunk
import sys

load_dotenv()

def build_vector_store(pdf_path: str):
    # Step 1: Load and chunk the PDF
    chunks = load_and_chunk(pdf_path)

    # Step 2: Create embeddings (free, runs locally)
    print("\nGenerating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 3: Store in FAISS
    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 4: Save locally
    vector_store.save_local("faiss_index")
    print("Vector store saved to faiss_index/")

    return vector_store

def search(query: str, vector_store):
    print(f"\nQuery: {query}")
    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py path/to/your.pdf")
    else:
        vs = build_vector_store(sys.argv[1])
        search("What is this document about?", vs)
