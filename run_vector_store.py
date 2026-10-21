
from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

# 1. Extract PDF text
pages = extract_pages("data/sample.pdf")

# 2. Split text into chunks
chunks = chunk_pages(pages, chunk_size=500, overlap=50)

if not chunks:
    raise ValueError("No text chunks found in the PDF.")

# 3. Create embeddings
embedding_model = EmbeddingModel()

# 4. Build FAISS vector index
store = VectorStore(embedding_model)
store.build(chunks)

# 5. Ask a question
question = input("Ask a question about your PDF: ")

# 6. Retrieve relevant chunks
results = store.search(question, top_k=3)

print("\nRelevant results:")
for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Similarity score:", round(result["score"], 4))
    print("Source:", result["filename"])
    print("Page:", result["page_number"])
    print("Text:", result["text"])
