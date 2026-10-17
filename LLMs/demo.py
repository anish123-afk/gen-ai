from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()

llm = GoogleGenerativeAI(model = "gemini-2.5-flash")

s = input("Enter the promt : ")

result = llm.invoke(s)

with open ("tp.json", "w") as f:
    json.dump(result, f)

