from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    # Load PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Total pages loaded: {len(pages)}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}")

    # Preview first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content)

    return chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loader.py path/to/your.pdf")
    else:
        load_and_chunk(sys.argv[1])
