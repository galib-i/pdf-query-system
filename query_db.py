import argparse

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import initialise_embeddings, initialise_chroma

PROMPT_TEMPLATE = """
Answer the question based only on the following context, if you cannot answer, say "Answer not found":

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    """Parses the CLI query to perform a RAG search"""
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    query_rag(args.query_text)


def query_rag(query_text):
    """Queries the RAG model using the input"""
    embeddings = initialise_embeddings()

    db = initialise_chroma(embeddings)  # loads the Chroma database
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    # extracts content and their assigned scores from the found results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt = generate_prompt(context=context_text, question=query_text)
    response_text, sources = return_response(prompt=prompt, results=results)

    print(f"Response: {response_text}\nScanned: {sources}")


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

    return response.content, sources


if __name__ == "__main__":
    main()
