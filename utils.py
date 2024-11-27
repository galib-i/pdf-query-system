import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


CHROMA_PATH = "chroma"
DATA_PATH = "data"

# save a Google AI API key as GOOGLE_API_KEY="<key>" in a .env file
load_dotenv()


def get_data_file_names():
    """Returns filenames in the data directory"""
    os.makedirs(DATA_PATH, exist_ok=True)

    return os.listdir(DATA_PATH)


def check_chroma_db(directory=CHROMA_PATH):
    """Checks if the Chroma database exists"""
    return os.path.exists(directory)


def initialise_embeddings():
    """Initialises the embeddings model"""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def initialise_chroma(embedding_function, directory=CHROMA_PATH):
    """Initialises a Chroma database"""
    return Chroma(persist_directory=directory, embedding_function=embedding_function)


def reset_chroma_db(directory=CHROMA_PATH):
    """Deletes and recreates the Chroma database."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
