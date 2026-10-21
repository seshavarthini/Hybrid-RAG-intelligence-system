
from pathlib import Path
from src.pdf_extraction import extract_pages, extract_text

pdf_path = Path("data/sample.pdf")

# Check whether the PDF exists
if not pdf_path.is_file():
    print("PDF not found:", pdf_path.resolve())
else:
    # Test page-by-page extraction
    pages = extract_pages(pdf_path)

    print("Number of pages:", len(pages))

    for page in pages:
        print("\nFilename:", page.filename)
        print("Page number:", page.page_number)
        print("Extracted text:")
        print(page.text)

    # Test full-document extraction
    full_text = extract_text(pdf_path)

    print("\nFull text extraction successful:", bool(full_text.strip()))
