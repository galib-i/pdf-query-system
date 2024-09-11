from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document  # object is a better format for AI usage

from utils import initialise_embeddings, initialise_chroma, reset_chroma_db

DATA_PATH = "data"


def main():
    """Populates the Chroma database with document parts"""
    documents = load_documents()
    text_parts = split_documents(documents)
    save_to_chroma(text_parts)


def load_documents():
    """Loads documents stored in the data directory"""
    document_loader = PyPDFDirectoryLoader(path=DATA_PATH)

    return document_loader.load()


def split_documents(documents: list[Document]):
    """Divides documents into manageable parts for indexing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    return text_splitter.split_documents(documents=documents)


def save_to_chroma(parts: list[Document]):
    """Saves document parts to a Chroma database"""
    reset_chroma_db()

    embeddings = initialise_embeddings()
    db = initialise_chroma(embeddings)
    db.add_documents(documents=parts)

    print(f"Stored {len(parts)} parts in the Chroma database.")


if __name__ == "__main__":
    main()
