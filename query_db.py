import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import initialise_embeddings, initialise_chroma

PROMPT_TEMPLATE = """
Answer the question based only on the following context, from the given files:

{context}

---

Answer the question based on the above context, from the given files: {question}
"""


def main():
    """Run the Streamlit program to query PDFs using the RAG model"""
    st.set_page_config(page_title="Query PDFs", page_icon="💬")
    st.title("Query your PDF documents")

    if "messages" not in st.session_state.keys():  # initialises the chat history
        st.session_state.messages = [{
            "role": "ai",
            "content": "Ask me a question about your documents!"
        }]

    if query := st.chat_input("Ask a question"):  # saves input to chat history
        st.session_state.messages.append({"role": "user", "content": query})

    for message in st.session_state.messages:  # displays chat history
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "ai":  # if the user has asked a question
        with st.chat_message("ai"):  # AI responds to the user's question
            response_text, sources = query_rag(st.session_state.messages[-1]["content"])

            output = f"""
            {response_text}

            Scanned: {sources}
            """
            st.write(output)
            # saves the AI response to the chat history
            message = {"role": "assistant", "content": output}
        st.session_state.messages.append(message)


def query_rag(query_text):
    """Queries the RAG model using the input"""
    embeddings = initialise_embeddings()

    db = initialise_chroma(embeddings)  # loads the Chroma database
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    # extracts content and their assigned scores from the found results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_text = generate_prompt(context=context_text, question=query_text)

    try:
        response_text, sources = return_response(prompt=prompt_text, results=results)
        return response_text, sources

    except Exception as e:
        return f"An error occurred: {e}"


def generate_prompt(context, question):
    """Generates a prompt to feed to Google Generative AI"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return prompt_template.format(context=context, question=question)


def return_response(prompt, results):
    """Returns the response and sources used from Google Generative AI"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    response = model.invoke(prompt)

    # lists the sources used to formulate the response
    sources = {doc.metadata.get("source", None) for doc, _score in results}
    sources = ", ".join(sources)

    return response.content, sources


if __name__ == "__main__":
    main()
