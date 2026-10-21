
from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_search import BM25Search
from src.hybrid_search import HybridSearch

pages = extract_pages("data/sample.pdf")
chunks = chunk_pages(pages, chunk_size=500, overlap=50)

if not chunks:
    raise ValueError("No text chunks found in the PDF.")

embedding_model = EmbeddingModel()

vector_store = VectorStore(embedding_model)
vector_store.build(chunks)

bm25_search = BM25Search()
bm25_search.build(chunks)

hybrid_search = HybridSearch(bm25_search, vector_store)

question = input("Ask a question about your PDF: ")
results = hybrid_search.search(question, top_k=3)

print("\nHybrid Search Results:")
for rank, result in enumerate(results, start=1):
    print(f"\nResult {rank}")
    print("RRF score:", round(result["rrf_score"], 4))
    print("Source:", result["filename"])
    print("Page:", result["page_number"])
    print("Text:", result["text"])
