from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import LlamafileEmbeddings

# Comment out or uncomment the desired embedding model

def get_embedding_function():
    #embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    #embeddings = HuggingFaceEmbeddings()
    #embeddings = LlamafileEmbeddings()
    return embeddings