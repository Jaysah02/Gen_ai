from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your .env file which should contain GOOGLE_API_KEY
load_dotenv()

# Create embedding model with Google Gemini's embedding API
embedding = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',  # correct for Gemini Flash 2.5
    task_type='retrieval_document',  # recommended task type
    dimensions=300
)

# Your list of documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = 'tell me about bumrah'

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get the most relevant document index
index = int(np.argmax(scores))
score = scores[index]

# Output results
print("Query:", query)
print("Most Relevant Document:", documents[index])
print("Similarity Score:", score)




