from flask import Flask, render_template, request, jsonify
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from get_embedding_function import get_embedding_function
from responses import responses  # Import the responses from the responses.py file
from extensions import manager_extensions # Import the manager_extensions from the extensions.py file

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# System prompt for the large language model (LLM)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Configuration for Chroma vector store
CHROMA_PATH = "chroma"

# Initialize components: LLM model, embeddings, vector store, and retrieval chain
model = OllamaLLM(model="llama3.1:8b")
embeddings = get_embedding_function()
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function)
retriever = vectorstore.as_retriever()
question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def normalize_input(user_input):
    """
    This function normalizes user input by converting it to lowercase and removing unnecessary spaces.
    """
    return user_input.strip().lower()

def find_response(user_input):
    """
    This function searches for a predefined response based on patterns in the user input.
    It iterates through a dictionary (imported from responses.py) and returns the corresponding response if a pattern matches.
    """
    for pattern, response in responses.items():
        if re.search(pattern, user_input):
            return response
    return None

def get_reply(user_input):
    """
    This function generates a reply based on the user's input.
    1. It first normalizes the user input using the `normalize_input` function.
    2. It then attempts to find a predefined response using the `find_response` function. If a match is found, it returns that response.
    3. If no predefined response is found, it uses the Retrieval-Augmented Generation (RAG) chain to retrieve relevant information and generate an answer using the LLM model.
    4. In case of any exceptions, it logs the error and returns a generic error message.
    """
    try:
        user_input = normalize_input(user_input)
        logging.info(f"Normalized user input: {user_input}")

        # Check for predefined response
        response = find_response(user_input)
        if response:
            logging.info(f"Response matched: {response}")
            return response

        # Use RAG chain for answer generation
        output = rag_chain.invoke({"input": user_input})
        return output["answer"]
    except Exception as e:
        logging.error(f"Error generating reply: {e}")
        return "Sorry, I couldn't process your request at the moment."

@app.route('/')
def home():
    """
    This function renders the home page template (likely 'index.html').
    """
    return render_template('index.html')


@app.route('/get_reply', methods=['POST'])
def get_reply_route():
    """
    This function handles POST requests to the '/get_reply' endpoint.
    1. It retrieves the user's message from the JSON payload.
    2. It checks if the message is empty and returns an appropriate error message if so.
    3. It calls the `get_reply` function to generate a response and returns it as a JSON object.
    4. If an exception occurs during the process, it logs the error and returns an error message.
    """
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({"reply": "No input received."})
        
        reply = get_reply(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        logging.error(f"Error in get_reply_route: {e}")
        return jsonify({"reply": "Sorry, an error occurred."})

@app.route('/get_extension', methods=['POST'])
def get_extension():
    """
    This function handles POST requests to the '/get_extension' endpoint.
    1. It retrieves the query from the JSON payload.
    2. It checks if the query is empty and returns an appropriate error message if so.
    3. It normalizes the query to match the dictionary keys.
    4. It looks up the extension(s) based on the normalized query.
    5. If extensions are found, it returns them as a JSON object.
    6. If no extensions are found, it returns an appropriate error message.
    7. If an exception occurs during the process, it logs the error and returns an error message.
    """
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({"extension": "No query provided."}), 400  # Bad request if no query

        # Normalize the query to match the dictionary keys
        normalized_query = normalize_input(query)
        
        # Lookup the extension(s) based on the normalized query
        extensions = manager_extensions.get(normalized_query, None)
        
        if extensions:
            extension_str = ', '.join(extensions)
            # Return the successful response with the found extensions
            return jsonify({"extension": f"The extension number(s) for {query} are: {extension_str}."}), 200
        else:
            # Return a response indicating that the extension was not found
            return jsonify({"extension": "Sorry, the extension number could not be found."}), 404
    except Exception as e:
        logging.error(f"Error in get_extension: {e}")
        return jsonify({"extension": "Sorry, an error occurred."}), 500  # Internal Server Error



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
