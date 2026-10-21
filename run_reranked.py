
from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_search import BM25Search
from src.hybrid_search import HybridSearch
from src.reranker import Reranker

pages = extract_pages("data/sample.pdf")
chunks = chunk_pages(pages, chunk_size=500, overlap=50)

if not chunks:
    raise ValueError("No text chunks found in the PDF.")

embedding_model = EmbeddingModel()

vector_store = VectorStore(embedding_model)
vector_store.build(chunks)

bm25 = BM25Search()
bm25.build(chunks)

hybrid = HybridSearch(bm25, vector_store)
reranker = Reranker()

question = input("Ask a question about your PDF: ")

# Retrieve more candidates before reranking.
results = hybrid.search(question, top_k=10)

# Reorder candidates by question-to-text relevance.
final_results = reranker.rerank(question, results, top_k=3)

print("\nReranked Results:")
for rank, result in enumerate(final_results, start=1):
    print(f"\nResult {rank}")
    print("Rerank score:", round(result["rerank_score"], 4))
    print("Source:", result["filename"])
    print("Page:", result["page_number"])
    print("Text:", result["text"])
