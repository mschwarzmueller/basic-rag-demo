import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(collection_name="demo"):
    client.create_collection(
        collection_name="demo",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


# dummy_data = [
#     "My name is Max",
#     "I like to eat pizza",
#     "I like to play basketball",
#     "I like to play football",
#     "My name is Manuel"
# ]

def generate_response(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:12b-it-qat",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def main():
    # for i, text in enumerate(dummy_data):
    #     response = requests.post(
    #         "http://localhost:11434/api/embed",
    #         json={"model": "mxbai-embed-large", "input": text},
    #     )
    #     data = response.json()
    #     embeddings = data["embeddings"][0]
    #     client.upsert(
    #         collection_name="demo",
    #         wait=True,
    #         points=[PointStruct(id=i, vector=embeddings, payload={"text": text})],
    #     )

    prompt = input("Enter a prompt: ")
  
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": adjusted_prompt},
    )
    data = response.json()
    embeddings = data["embeddings"][0]
    
    results = client.query_points(
        collection_name="demo",
        query=embeddings,
        with_payload=True,
        limit=2
    )

    relevant_passages = "\n".join([f"- {point.payload['text']}" for point in results.points])

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

