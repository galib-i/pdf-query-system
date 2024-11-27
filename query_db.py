import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from populate_db import populate_database, add_document
from utils import get_data_file_names, check_chroma_db, initialise_embeddings, initialise_chroma

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    """Runs the Streamlit app for querying PDFs with RAG"""
    st.set_page_config(page_title="Query PDFs", page_icon="💬")
    st.title("Query your PDF documents")

    with st.sidebar:
        handle_sidebar()

    initialise_chat()

    if query := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": query})

    display_chat_history()

    if st.session_state.messages[-1]["role"] == "user":
        handle_user_query()


def handle_sidebar():
    """Handles file uploads and displays stored documents"""
    if "uploader_key" not in st.session_state:  # https://discuss.streamlit.io/t/clear-the-file-uploader-after-using-the-file-data/66178/4
        st.session_state.uploader_key = 0  # set a key for the uploader

    uploaded_file = st.file_uploader("Upload", label_visibility="hidden", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")

    if uploaded_file:
        file_path = add_document(document=uploaded_file)
        st.success(f"File saved at: {file_path}")
        # increment the key to reset the uploader
        st.session_state.uploader_key += 1

    file_names = get_data_file_names()

    if not file_names:
        st.warning("No files found. Please upload PDFs.")

    elif not check_chroma_db():
        st.warning("Database does not exist.")
        st.button("Create Database", type="primary", on_click=populate_database)

    else:
        st.write("Stored documents:")

        for file_name in file_names:
            st.markdown(f"- {file_name}")


def initialise_chat():
    """initialises the chat state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": "Ask me a question about your documents."}]


def display_chat_history():
    """Displays the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_query():
    """Handles user queries by providing AI responses"""
    query = st.session_state.messages[-1]["content"]

    with st.chat_message("ai"):
        response, sources = query_rag(query)
        output = f"{response}\n\nScanned: {sources}"
        st.write(output)
        st.session_state.messages.append({"role": "assistant", "content": output})


def query_rag(query_text):
    """Queries the RAG model with a user's input"""
    embeddings = initialise_embeddings()
    db = initialise_chroma(embeddings)
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_text = generate_prompt(context=context_text, question=query_text)

    try:
        return return_response(prompt=prompt_text, results=results)

    except Exception as e:
        return f"An error occurred: {e}", ""


def generate_prompt(context, question):
    """Generates the RAG prompt"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return prompt_template.format(context=context, question=question)


def return_response(prompt, results):
    """Gets a response and sources from the AI"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    response = model.invoke(prompt)

    sources = ", ".join({doc.metadata.get("source", "Unknown") for doc, _ in results})

    return response.content, sources


if __name__ == "__main__":
    main()
