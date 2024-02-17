# Unstract Adapters

This is Unstract's python package which helps to configure to a number of different LLMs, Embeddings and VectorDBs.

## LLMs
The following LLMs are supported:

| LLM          | Version |
|--------------|---------|
| OpenAI       | 1.3.9   |
| Azure OpenAI | 1.3.9   |
| Anthropic    | 0.7.8   |
| PaLM         | 0.3.1   |
| Replicate    | 0.22.0  |
| AnyScale     | 0.5.165 |
| Mistral      | 0.0.8   |

## Embeddings
The following Embeddings are supported:

| Embedding   | Version |
|-------------|---------|
| OpenAI      |   1.3.9      |
| Azure OpenAI |    1.3.9     |
| Qdrant FastEmbed   |    0.1.3     |
| HuggingFace        |    0.0.1     |
| PaLM    |    0.3.1     |

## VectorDBs
The following VectorDBs are supported:

| Vector DB        | Version |
|------------------|--------|
| Milvus           |   2.3.4     |
| Pinecone    |    2.2.4    |
| Postgres |    0.2.4    |
| Qdrant      |    1.7.0    |
| Supabase             |    2.2.1     |
| Weaviate             |    3.25.3    |

## Installation

### Local Development

To get started with local development, 
- Create and source a virtual environment if you haven't already following [these steps](/README.md#create-your-virtual-env).
- If you're using Mac, install the below library needed for PyMSSQL
```
brew install pkg-config freetds
```
- Install the required dependencies with
```shell
pdm install
```
