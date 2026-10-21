from src.pdf_extraction import extract_pages
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.bm25_search import BM25Search
from src.hybrid_search import HybridSearch
from src.reranker import Reranker
from src.generator import AnswerGenerator

# 1. Load the PDF
pdf_path = "data/sample.pdf"
pages = extract_pages(pdf_path)

# 2. Split PDF into chunks
chunks = chunk_pages(pages, chunk_size=500, overlap=50)
print(f"PDF loaded. Total chunks: {len(chunks)}")

# 3. Build semantic and keyword search
embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
vector_store.build(chunks)

bm25 = BM25Search()
bm25.build(chunks)

hybrid = HybridSearch(bm25, vector_store)
reranker = Reranker()
generator = AnswerGenerator()

print("\nAsk questions about your PDF. Type 'exit' to stop.")

while True:
    question = input("\nYour question: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    # 4. Retrieve relevant chunks
    candidates = hybrid.search(question, top_k=10)

    # 5. Rerank the chunks
    results = reranker.rerank(question, candidates, top_k=3)

    # 6. Generate answer using Groq API
    answer = generator.generate(question, results)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for result in results:
        print(
            f"- {result['filename']}, "
            f"Page {result['page_number']}"
        )
