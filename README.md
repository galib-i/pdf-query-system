# PDF Query System
A Streamlit app using Langchain and the Gemini AI model for retrieval-augmented generation (RAG) to query stored PDFs in Python.

## Overview
![one](https://github.com/user-attachments/assets/9ca6bcb7-8097-4363-b403-c4560af6842f)
![two](https://github.com/user-attachments/assets/c0047282-2cd8-4ee7-8bd8-f3c9d0e9e88c)

## Getting Started
1. **Set your Google API key** in a `.env` file

   Obtain a free Google API key from [here](https://aistudio.google.com/app/apikey)

   ```.env
   GOOGLE_API_KEY="<key>"
   ```
   
3. **Clone the repository**

   ```bash
   git clone https://github.com/galib-i/pdf-query-system.git
   ```
   
4. **Install dependencies**
   ```
   langchain_chroma, langchain_community, langchain_google_genai, pypdf, python-dotenv, streamlit
   ```
   
5. **Insert PDF files into the `\data` directory**

   Two files are present from the cloning of this repository

7. **Populate the Chroma vector database**

   Run `populate_db.py`
  
8. **Run the Streamlit program**

   ```bash
   streamlit run query_db.py
   ```
   
9. **Start asking questions!**
