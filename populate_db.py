import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# generate a Google AI API key, save it as GOOGLE_API_KEY="<key>" in a .env
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    documents = load_documents()
    text_parts = split_documents(documents=documents)
    save_to_chroma(parts=text_parts)


def load_documents():
    """Loads documents from the DATA_PATH directory

    Returns:
        list[Document]: a list of documents
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """Divdes documents into more manageable parts (indexing and storing)

    Args:
        documents (list[Document]): documents to split

    Returns:
        list[Document]: parts of the original documents
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    return text_splitter.split_documents(documents)


def save_to_chroma(parts: list[Document]):
    """Saves the documents to a Chroma database

    Args:
        parts (list[Document]): documents to be saved
    """
    if os.path.exists(CHROMA_PATH):  # delete old content from the folder
        shutil.rmtree(CHROMA_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(documents=parts, embedding=embeddings,
                               persist_directory=CHROMA_PATH)
    print(f"Saved {len(parts)} parts in {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
