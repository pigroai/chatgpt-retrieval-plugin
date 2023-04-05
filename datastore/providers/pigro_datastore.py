import asyncio
import os
import requests
import json
import urllib.parse

from datastore.datastore import DataStore
from typing import Dict, List, Optional

from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadataFilter,
    DocumentChunkWithScore,
    Query,
    QueryResult,
    QueryWithEmbedding,
)
from services.pigro_chunks import get_pigro_document_chunks
from services.pigro_embedder import get_pigro_embeddings

PIGRO_API_HOST = os.environ.get("PIGRO_HOST", None)
PIGRO_KEY = os.environ.get("PIGRO_KEY", None)


class PigroDataStore(DataStore):
    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        """
        Takes in a list of documents and inserts them into Pigro's Api.
        First deletes all the existing chunks inside pigro system with the document id, then inserts the new ones.
        Return a list of chunks ids.
        """
        # Delete any existing vectors for documents with the input document ids
        await asyncio.gather(
            *[
                self.delete(
                    filter=DocumentMetadataFilter(
                        document_id=document.id,
                    ),
                    delete_all=False,
                )
                for document in documents
                if document.id
            ]
        )

        chunks = get_pigro_document_chunks(documents)
        return await self._upsert(chunks)

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of chunks ids.
        """
        data = []
        chunks_id = []
        for doc_id, chunks_list in chunks.items():
            for chunk in chunks_list:
                chunks_id.append(chunk.id)
                data.append({
                    "id": chunk.id,
                    "body": chunk.text
                })
        data = {
            "documents": data
        }
        if self._call_pigro_api(data, "add_documents"):
            self._call_pigro_api({"webhook": ""}, "train")
            return chunks_id

        return []

    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = get_pigro_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings)

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores using pigro's hybrid search api.
        """
        results: List[QueryResult] = []
        for query in queries:
            question = query.query
            append = "search?k=-1&query"+urllib.parse.quote(question)
            pigro_result = self._get_pigro_api(append)
            query_results = []
            if pigro_result != None and pigro_result != False:
                for chunk_id, score in pigro_result:
                    append = "get_paragraph_info?id="+chunk_id
                    chunk_info = self._get_pigro_api(append)
                    if chunk_info != None and chunk_info != False:
                        result = DocumentChunkWithScore(
                            id=chunk_id,
                            score=score,
                            text=chunk_info["body"],
                            metadata=chunk_info["metadata"]
                        )
                        query_results.append(result)

            results.append(QueryResult(
                query=query.query, results=query_results))
        return results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes chunks by ids, filter, or everything in Pigro's Api.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """
        if delete_all:
            data = []
            return self._call_pigro_api([], "delete_all")

        if filter:
            ids = []
            for field, value in filter.dict().items():
                if field == "document_id":
                    ids.append(value)

            data = {
                "has_doc_id": ids
            }
            return self._call_pigro_api(data, "delete_with_filter")

        if ids:
            data = {
                "ids": ids
            }
            return self._call_pigro_api(data, "delete")

    async def _call_pigro_api(self, data, method):
        headers = {
            "x-api-key": PIGRO_KEY,
            'Content-Type': 'application/json'
        }
        try:
            r = requests.post(
                PIGRO_API_HOST+method,
                headers=headers,
                json=json.dump(data)
            )

            if r.status_code == 200:
                response = r.json()
                if response['success']:
                    return True
                else:
                    raise Exception(response['message'])

        except Exception as e:
            print(f"Error: {e}")
            raise e
        return False

    async def _get_pigro_api(self, append):
        headers = {
            "x-api-key": PIGRO_KEY,
            'Content-Type': 'application/json'
        }
        try:
            r = requests.get(
                PIGRO_API_HOST+append,
                headers=headers
            )

            if r.status_code == 200:
                response = r.json()
                if response['success']:
                    return response['result']
                else:
                    raise Exception(response['message'])

        except Exception as e:
            print(f"Error: {e}")
            raise e
        return False
