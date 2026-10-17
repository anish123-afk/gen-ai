from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model  = "gemini-2.5-flash")

result = model.invoke("hi i am  anish")
result1 = model.invoke("who was chatrapati shivaji maharaj")
result2 = model.invoke("what is my name")

print(result.content)
print(result1.content)
print(result2.content)