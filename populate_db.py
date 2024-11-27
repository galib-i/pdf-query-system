import os

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from utils import initialise_embeddings, initialise_chroma, reset_chroma_db, DATA_PATH


def populate_database():
    """Populates the Chroma database with all PDFs in the data folder"""
    documents = load_documents()
    text_parts = split_documents(documents)
    save_to_chroma(text_parts)


def add_document(document):
    """Adds a single document to the Chroma database"""
    file_path = save_document(document)
    document_content = PyPDFLoader(file_path).load()
    text_parts = split_documents(document_content)

    append_to_chroma(text_parts)

    return file_path


def save_document(document):
    """Saves an uploaded document to the data folder"""
    file_path = os.path.join(DATA_PATH, document.name)
    os.makedirs(DATA_PATH, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(document.getbuffer())

    return file_path


def load_documents():
    """Loads all PDF documents from the data folder"""
    loader = PyPDFDirectoryLoader(path=DATA_PATH)

    return loader.load()


def split_documents(documents: list[Document]):
    """Splits documents into smaller chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    return splitter.split_documents(documents=documents)


def save_to_chroma(parts: list[Document]):
    """Resets and stores new document parts in the Chroma database"""
    reset_chroma_db()
    append_to_chroma(parts)


def append_to_chroma(parts: list[Document]):
    """Appends new document parts to the Chroma database."""
    embeddings = initialise_embeddings()
    db = initialise_chroma(embeddings)
    db.add_documents(documents=parts)

    print(f"Stored {len(parts)} parts in the Chroma database.")


if __name__ == "__main__":
    populate_database()
