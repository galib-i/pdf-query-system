import os
import shutil

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# save a Google AI API key as GOOGLE_API_KEY="<key>" in a .env file
load_dotenv()  # hide API key

CHROMA_PATH = "chroma"


def initialise_embeddings():
    """Initialises Google Generative AI embeddings"""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def initialise_chroma(embedding_function, directory=CHROMA_PATH):
    """Initialises a Chroma database, storing document data and vector embeddings"""
    return Chroma(persist_directory=directory, embedding_function=embedding_function)


def reset_chroma_db(directory=CHROMA_PATH):
    """Removes and recreates the existing Chroma directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
