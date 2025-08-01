
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ✅ Insert your API key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyDEIcwsBNN95oKxy9cKW2RKO3It20KNI1M"

# ✅ Create the model instance
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')  # or 'gemini-1.5-pro' if supported

# ✅ Invoke the model
result = model.invoke('What is the capital of India?')

# ✅ Print the result
print(result.content)

