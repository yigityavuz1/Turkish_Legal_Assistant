import fitz
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate

# ====== EMBEDDINGS ======
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, 
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs
)

# ====== PDF -> PAGE DOCS + CHUNK DOCS ======
def chunk_text(text, max_words=300, overlap=50):
    """
    Splits text into overlapping segments of `max_words` length, 
    where each chunk overlaps the previous one by `overlap` words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def pdf_to_docs(pdf_file):
    """
    Reads a single PDF, removes specific unwanted strings, 
    skips the first 22 pages, and returns both page-level and chunk-level documents.
    """
    reader = fitz.open(pdf_file)
    number_of_pages = reader.page_count

    page_docs = []
    chunk_docs = []

    for i in range(number_of_pages):
        # Skip first 22 pages
        if i < 22:
            continue
        
        page = reader[i]
        text = page.get_text()

        # Example of removing one repeating string
        text = text.replace("T Ü R K İ Y E  C U M H U R İ Y E T İ  A N AYA S A S I", "")

        if len(text.strip()) < 10:
            # Skip pages with no meaningful text
            continue
        
        # ====== PAGE DOCS ======
        page_doc = Document(
            page_content=text,
            metadata={"source": pdf_file, "page_number": i + 1, "type": "page"}
        )
        page_docs.append(page_doc)

        # ====== CHUNK DOCS ======
        word_chunks = chunk_text(text, max_words=300, overlap=50)
        for chunk in word_chunks:
            chunk_doc = Document(
                page_content=chunk,
                metadata={"source": pdf_file, "page_number": i + 1, "type": "chunk"}
            )
            chunk_docs.append(chunk_doc)

    return page_docs, chunk_docs


# ====== GET DB INSTANCE ======
def get_db_instance(pdf_file):
    """
    Creates a local Weaviate client, 
    converts PDF pages to documents and chunked documents, 
    and returns two vector stores.
    """
    # Connect to a local Weaviate instance (e.g., running on localhost:8080)
    client = weaviate.connect_to_local()

    page_docs, chunk_docs = pdf_to_docs(pdf_file)

    page_db = WeaviateVectorStore.from_documents(page_docs, embeddings, client=client)
    chunk_db = WeaviateVectorStore.from_documents(chunk_docs, embeddings, client=client)
    
    return page_db, chunk_db
