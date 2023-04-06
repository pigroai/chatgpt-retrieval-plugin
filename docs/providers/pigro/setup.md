# Pigro

[Pigro](https://openai.pigro.ai/) is a managed API solution for AI enterprise search. It offers a wide range of features, including native support for Office-like files and PDFs, advanced text chunking based on semantics and document structure, document expansion techniques based on generative AI, hybrid search, and multi-language. Pigro has been built from scratch to provide clear-cut answers, the precise portion of a document that answers the user's query. This type of output is exactly what ChatGPT expects.

Pigro provides AI-based text chunking services that split the content as a human would do taking into account:
- the look and structure of the document, such as pagination, headings, tables, lists, images, etc.
- the semantics of the text, keeping together related sentences and splitting where is needed

Our API natively supports Office-like documents, PDF, HTML, and plain text in many languages. We expand each content with generative AI: we generate all the questions that are answered within the document. We then combine keyword and semantic search, considering the title, the body, and the generated questions.

## Getting started
Signup [here](https://openai.pigro.ai/) to request your configuration details. We will come back to you with the host and the key to properly configure the environment.
If you need multiple document libraries for the moment you have to signup with different accounts.
During the first API call to add documents, the system will set up the environment (indexes, vector databases, document storing).

**Environment Variables:**

| Name                    | Required | Description                                                                                                            | Default     |
| ----------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------- | ----------- |
| `DATASTORE`             | Yes      | Datastore name, set to `pigro`                                                                                         |             |
| `PIGRO_HOST`            | Yes      | The end point for Pigro's API more info about it from [here](https://openai.pigro.ai)|             |
| `PIGRO_KEY`             | Yes      | The secret key `API_KEY` for accessing pigro's api services                                                            |             |
| `PIGRO_LANGUAGE`        | Yes      | The main language for your documents/data                                                                              |             |
