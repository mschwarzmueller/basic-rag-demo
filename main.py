import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import yaml
import re
import os
import uuid

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(collection_name="articles"):
    client.create_collection(
        collection_name="articles",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


def extract_metadata_from_mdx(file_path: str):
    with open(file_path, "r") as file:
        content = file.read()

    parts = content.split('---')
    if len(parts) < 3:
        # No frontmatter found or malformed
        return {}, content

    raw_metadata = parts[1]
    article_content = '---'.join(parts[2:]).strip()

    try:
        metadata = yaml.safe_load(raw_metadata)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML metadata: {e}")
        return {}, article_content

    return metadata, article_content


def create_chunks(article_content: str):
    chunks = []
    current_chunk = ""

    for line in article_content.split("\n\n"):
        if line.startswith("#"):
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    return chunks


def clean_article_content(article_content: str):
    # Remove import statements
    cleaned_content = re.sub(r"^import .*\n?", "",
                             article_content, flags=re.MULTILINE)
    # Remove XML/HTML tags but keep content
    cleaned_content = re.sub(r"<[^>]+>", "", cleaned_content)
    return cleaned_content.strip()


def generate_response(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:12b-it-qat",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 10000
            }
        }
    )
    return response.json()["response"]


def generate_embeddings(text: str):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": text},
    )
    if len(response.json()["embeddings"]) > 0:
        return response.json()["embeddings"][0]
    else:
        return None


def store_article(metadata: dict, chunks: list[str]):
    for chunk in chunks:
        # Generate a unique ID for each chunk
        chunk_id = str(uuid.uuid4())
        adjusted_metadata = {
            **metadata,
            "content": chunk
        }
        embeddings = generate_embeddings(chunk)

        if embeddings is not None:
            client.upsert(
                collection_name="articles",
                wait=True,
                points=[PointStruct(
                    id=chunk_id, vector=embeddings,
                    payload=adjusted_metadata
                )],
            )


def main():
    # article_files = [f for f in os.listdir("articles") if f.endswith(".mdx")]
    # for article_file in article_files:
    #     file_path = os.path.join("articles", article_file)
    #     metadata, article_content = extract_metadata_from_mdx(file_path)
    #     cleaned_article_content = clean_article_content(article_content)
    #     chunks = create_chunks(cleaned_article_content)
    #     metadata["slug"] = article_file.replace(".mdx", "")
    #     store_article(metadata=metadata, chunks=chunks)

    prompt = input("Enter a prompt: ")
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": adjusted_prompt},
    )
    data = response.json()
    embeddings = data["embeddings"][0]

    results = client.query_points(
        collection_name="articles",
        query=embeddings,
        with_payload=True,
        limit=10
    )

    relevant_passages = "\n".join(
        [f"- Article Title: {point.payload['title']} -- Article Slug: {point.payload['slug']} -- Article Content: {point.payload['content']}" for point in results.points])

    # print(relevant_passages)

    augmented_prompt = f"""
      The following are relevant passages:
      <retrieved-data>
      {relevant_passages}
      </retrieved-data>

      Here's the original user prompt, answer with help of the retrieved passages:
      <user-prompt>
      {prompt}
      </user-prompt>
    """

    response = generate_response(augmented_prompt)
    print(response)


if __name__ == "__main__":
    main()
