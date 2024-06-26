import uuid
import os
import html
import requests

from typing import Dict, List, Optional, Tuple
from models.models import Document, DocumentChunk, DocumentChunkMetadata


# from services.pigro_embedder import get_pigro_embeddings

# Global variables
PIGRO_SPLITTER_HOST = os.environ.get("PIGRO_HOST", None) + "chunks"
PIGRO_KEY = os.environ.get("PIGRO_KEY", None)
PIGRO_LANGUAGE = os.environ.get("PIGRO_LANGUAGE", None)
EMBEDDINGS_BATCH_SIZE = 128


def get_text_chunks(text: str) -> List[str]:
    """
    Split a text into chunks of by providing AI-based text chunking services that split the content as a human would do taking into account.

    Args:
        text: The html text to split into chunks.

    Returns:
        A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
    """
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return []

    chunks = []
    values = {
        'text': text,  # text is here a html text.
        'language': PIGRO_LANGUAGE
    }
    headers = {
        "x-api-key": PIGRO_KEY,
        'Content-Type': 'application/json'
    }
    try:
        r = requests.post(
            PIGRO_SPLITTER_HOST,
            headers=headers,
            json=values
        )

        if r.status_code == 200:
            response = r.json()
            if response['success']:
                for ch in response["data"]['paragraphs']:
                    chunks.append(html.unescape(ch['paragraph']))
            else:
                raise Exception(response['error'])
        else:
            r.raise_for_status()

    except Exception as e:
        print(f"Error: {e}")
        raise e

    return chunks


def create_document_chunks(doc: Document) -> Tuple[List[DocumentChunk], str]:
    """
    Create a list of document chunks from a document object and return the document id.

    Args:
        doc: The document object to create chunks from. It should have a text attribute containing the html text data of the document and optionally an id and a metadata attribute.

    Returns:
        A tuple of (doc_chunks, doc_id), where doc_chunks is a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute,
        and doc_id is the id of the document object, generated if not provided. The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
    """
    # Check if the document text is empty or whitespace
    if not doc.text or doc.text.isspace():
        return [], doc.id or str(uuid.uuid4())

    # Generate a document id if not provided
    doc_id = doc.id or str(uuid.uuid4())

    # Split the document text into chunks
    text_chunks = get_text_chunks(doc.text)

    metadata = (
        DocumentChunkMetadata(**doc.metadata.__dict__)
        if doc.metadata is not None
        else DocumentChunkMetadata()
    )

    metadata.document_id = doc_id

    # Initialize an empty list of chunks for this document
    doc_chunks = []

    # Assign each chunk a sequential number and create a DocumentChunk object
    for i, text_chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{i}"
        doc_chunk = DocumentChunk(
            id=chunk_id,
            text=text_chunk,  # text_chunk also a part of the html text of the document
            metadata=metadata,
        )
        # Append the chunk object to the list of chunks for this document
        doc_chunks.append(doc_chunk)

    # Return the list of chunks and the document id
    return doc_chunks, doc_id


def get_pigro_document_chunks(documents: List[Document]) -> Dict[str, List[DocumentChunk]]:
    """
    Convert a list of documents into a dictionary from document id with list of document chunks.

    Args:
        documents: The list of documents to convert.

    Returns:
        A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object
        with html text in text parameter, metadata, and embedding attributes.
    """
    # Initialize an empty dictionary of lists of chunks
    chunks: Dict[str, List[DocumentChunk]] = {}

    # Initialize an empty list of all chunks
    all_chunks: List[DocumentChunk] = []

    # Loop over each document and create chunks
    for doc in documents:
        doc_chunks, doc_id = create_document_chunks(doc)

        # Append the chunks for this document to the list of all chunks
        all_chunks.extend(doc_chunks)

        # Add the list of chunks for this document to the dictionary with the document id as the key
        chunks[doc_id] = doc_chunks

    # Check if there are no chunks
    if not all_chunks:
        return {}

    # Get all the embeddings for the document chunks in batches, using get_pigro_embeddings
    # embeddings: List[List[float]] = []
    # for i in range(0, len(all_chunks), EMBEDDINGS_BATCH_SIZE):
    #     # Get the text of the chunks in the current batch
    #     # batch_texts = [
    #     #     chunk.text for chunk in all_chunks[i: i + EMBEDDINGS_BATCH_SIZE]
    #     # ]
    #     # Get the embeddings for the batch texts
    #     batch_embeddings = []  # get_pigro_embeddings(batch_texts)

    #     # Append the batch embeddings to the embeddings list
    #     embeddings.extend(batch_embeddings)

    # Update the document chunk objects with the embeddings
    for i, chunk in enumerate(all_chunks):
        # Assign the embedding from the embeddings list to the chunk object
        chunk.embedding = [] #embeddings[i]

    return chunks
