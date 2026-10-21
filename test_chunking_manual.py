
from src.pdf_extraction import PageText
from src.chunking import chunk_pages, chunk_text

# Test 1: Simple text chunking
text = "Artificial Intelligence " * 100
chunks = chunk_text(text, chunk_size=500, overlap=50)

print("Test 1: Text chunking")
print("Number of chunks:", len(chunks))
print("First chunk length:", len(chunks[0]))
print("Second chunk length:", len(chunks[1]))

# Test 2: Check source information
pages = [
    PageText(
        filename="sample.pdf",
        page_number=1,
        text="This is a sample document for testing chunking. " * 10,
    )
]

result = chunk_pages(pages, chunk_size=500, overlap=50)

print("\nTest 2: Source information")
print("Filename:", result[0].filename)
print("Page number:", result[0].page_number)
print("Chunk index:", result[0].chunk_index)

# Test 3: Invalid settings should raise an error
try:
    chunk_text("hello world", chunk_size=10, overlap=10)
except ValueError:
    print("\nTest 3: Invalid overlap correctly rejected")
