import os
import requests
import json

from typing import List

from tenacity import retry, wait_random_exponential, stop_after_attempt

PIGRO_EMBEDDER_HOST = os.environ.get("PIGRO_HOST", None) + "/embedder"
PIGRO_KEY = os.environ.get("PIGRO_KEY", None)
PIGRO_LANGUAGE = os.environ.get("PIGRO_LANGUAGE", None)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_pigro_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using Pigro's Embedder model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the Pigro's Embedder call fails.
    """

    values = {
        'sentences': json.dump(texts)
    }
    headers = {
        "x-api-key": PIGRO_KEY,
        'Content-Type': 'application/json'
    }
    embeddings = []
    try:
        r = requests.post(
            PIGRO_EMBEDDER_HOST,
            headers=headers,
            json=values
        )

        if r.status_code == 200:
            response = r.json()
            if response['status']:
                for e in response['embeddings']:
                    embeddings.append([float(f) for f in e])
            else:
                raise Exception(response['message'])

    except Exception as e:
        print(f"Error: {e}")
        raise e

    return embeddings
