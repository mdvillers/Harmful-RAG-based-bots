from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize
project_id = os.getenv("PROJECT_ID")
location = os.getenv("REGION")
data_store_id = os.getenv("DATA_STORE_ID")


def retrieve_faq_answer(user_query):
    client_options = ClientOptions(api_endpoint="us-discoveryengine.googleapis.com")

    client = discoveryengine.SearchServiceClient(client_options=client_options)
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        serving_config="default_config",
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=user_query,
        page_size=3,  # Get top 3 most relevant Q&A pairs
        # Optional: Filter by category if needed
        # filter="category: \"System Specs\""
    )

    response = client.search(request)

    retrieved_knowledge = []

    for result in response.results:
        faq_entry = result.document.struct_data
        question = faq_entry.get("question_text")
        answers = faq_entry.get("answers")

        answer_texts = []
        for answer in answers:
            answer_text = answer.get("answer_text")
            answer_texts.append("A: " + answer_text)
        combined_answer = "\n".join(answer_texts)
        formatted_knowledge = f"Q: {question}\n{combined_answer}"
        retrieved_knowledge.append(formatted_knowledge)

    return "\n\n".join(retrieved_knowledge)


user_query = "What windows version do I need?"

context = retrieve_faq_answer(user_query)

print(f"--- Retrieved Context ---\n{context}\n")

prompt = f"""
You are a support bot. Use the following FAQ entry to answer the user.
FAQ Context:
{context}

User Question: {user_query}
"""

print(f"--- Generated Prompt for LLMs ---\n{prompt}\n")
