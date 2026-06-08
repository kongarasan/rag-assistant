from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loader import load_and_chunk
import sys

load_dotenv()

def build_rag_chain(pdf_path: str):
    # Step 1: Load and chunk PDF
    chunks = load_and_chunk(pdf_path)

    # Step 2: Embeddings + FAISS
    print("\nBuilding vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")

    # Step 3: LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Step 4: Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
""")

    # Step 5: Build LCEL chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask(chain, question: str):
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Setting up RAG pipeline for: {pdf_path}")
    chain = build_rag_chain(pdf_path)

    print("\nRAG pipeline ready! Type your questions (or 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            ask(chain, question)
