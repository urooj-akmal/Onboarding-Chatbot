import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    """
    Main function to populate the Chroma database with documents.

    Args:
        --reset (optional): If specified, clears the existing database before populating.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    # Clear the database if the --reset flag is provided
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load documents from the specified directory
    documents = load_documents()
    
    # Split documents into smaller chunks for efficient processing
    chunks = split_documents(documents)
    
    # Add the document chunks to the Chroma database
    add_to_chroma(chunks)


def load_documents():
    """
    Loads PDF documents from the specified data directory.

    Returns:
        A list of Document objects.
    """

    # Create a document loader for PDF files
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    
    # Load the documents from the directory
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    Splits a list of documents into smaller chunks for efficient processing.

    Args:
        documents: A list of Document objects.

    Returns:
        A list of Document objects, each representing a chunk of the original documents.
    """

    # Create a text splitter to divide documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents into chunks
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Adds a list of document chunks to the Chroma database.

    Args:
        chunks: A list of Document objects.
    """

    # Load the existing Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate unique IDs for each chunk
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing document IDs from the database
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add new documents to the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Add new documents to the database
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    """
    Calculates unique IDs for each document chunk based on source, page number, and chunk index.

    Args:
        chunks: A list of Document objects.

    Returns:
        A list of Document objects with calculated IDs.
    """

    # Initialize variables
    last_page_id = None
    current_chunk_index = 0

    # Iterate over each chunk
    for chunk in chunks:
        # Extract source and page information
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        # Generate a unique page ID
        current_page_id = f"{source}:{page}"

        # Increment the chunk index if the page ID is the same as the previous one
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Create the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Update the chunk's metadata with the ID
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Clears the existing Chroma database.
    """

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
