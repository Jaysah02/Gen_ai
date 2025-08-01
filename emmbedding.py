import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
document = ["The capital of India is New Delhi.","The capital of USA is Wasinghton.","The capital of France is Paris."]
embedding_vector = embedding_model.embed_documents(document)
print(embedding_vector)