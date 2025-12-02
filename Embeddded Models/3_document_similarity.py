from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", dimension=300)

document = ["virat kohli is a cricketer.", "he is a good batsman.", "he is a former captain of indian cricket team."," he is one of the best cricket players in the world."]


query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(document)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index , score =sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]  # prints the index and similarity score of the most similar document

print(query)
print(document[index])
print(f"Document: {document[index]} \nScore: {score}")




