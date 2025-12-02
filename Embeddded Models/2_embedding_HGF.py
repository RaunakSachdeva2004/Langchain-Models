from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

text = "LangChain is a framework for developing applications powered by language models."
vector = embedding.embed_query(text)
print(vector)
