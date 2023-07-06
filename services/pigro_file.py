import os
import hashlib
import mimetypes
import time
import requests

from typing import Optional
from fastapi import UploadFile

from models.models import Document, DocumentMetadata

PIGRO_CONVERTER_HOST = os.environ.get("PIGRO_HOST", None) + "convert"
PIGRO_KEY = os.environ.get("PIGRO_KEY", None)


async def get_document_from_file(
    file: UploadFile, metadata: DocumentMetadata
) -> Document:
    extracted_text = await extract_text_from_form_file(file)

    doc = Document(text=extracted_text, metadata=metadata)

    return doc


def extract_text_from_filepath(filename: str, filepath: str, mimetype: Optional[str] = None) -> str:
    """Return the text content of a file given its filepath."""
    extracted_text = ""
    if mimetype is None:
        # Get the mimetype of the file based on its extension
        mimetype, _ = mimetypes.guess_type(filepath)

    if not mimetype:
        if filepath.endswith(".md"):
            mimetype = "text/markdown"
        else:
            raise Exception("Unsupported file type")
    try:
        # call pigro's converter and set the returned data inside the response itself...
        files = {
            'file': (filename, open(filepath, 'rb'))
        }
        headers = {
            "x-api-key": PIGRO_KEY
        }

        r = requests.post(
            PIGRO_CONVERTER_HOST,
            headers=headers,
            files=files
        )
        if r.status_code == 200:
            response = r.json()
            if response['success']:
                extracted_text = response['data']
                if len(extracted_text) == 0:
                    raise Exception(
                        "The returned converted data is empty, the response is: " + r.text)
            else:
                raise Exception(response['error'])
    except Exception as e:
        print(f"Error: {e}")
        raise e

    return extracted_text


# Extract text from a file based on its mimetype, Remember this function will return a rich text html content not just a text, so you've to strip tags.
async def extract_text_from_form_file(file: UploadFile):
    """Return the text content of a file."""
    # get the file body from the upload file object
    mimetype = file.content_type
    # print(f"mimetype: {mimetype}")
    # print(f"file.file: {file.file}")
    # print("file: ", file)
    ext = os.path.splitext(file.filename.strip())[1]

    tmp_name = hashlib.md5((str(time.time()) + file.filename.strip()).encode()).hexdigest()

    file_stream = await file.read()

    temp_file_path = "/tmp/" + tmp_name+ext

    # write the file to a temporary location
    with open(temp_file_path, "wb") as f:
        f.write(file_stream)

    try:
        extracted_text = extract_text_from_filepath(tmp_name+ext, temp_file_path, mimetype)
    except Exception as e:
        print(f"Error: {e}")
        os.remove(temp_file_path)
        raise e

    # remove file from temp location
    os.remove(temp_file_path)

    return extracted_text
