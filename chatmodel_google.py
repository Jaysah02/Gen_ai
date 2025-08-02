
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
# Instantiate the model
model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash-latest',
    temperature=0
)
result = model.invoke('write five line poem on great india')

print(result.content)

