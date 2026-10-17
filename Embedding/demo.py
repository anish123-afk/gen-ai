from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Delhi is captial of india",
    "Toxic movie was very boring",
    "Today we have soya wadi and lassi in lunch"
]

embedding = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001', output_dimensionality= 5)

result = embedding.embed_query('sambhaji maharaj was the son of shivaji maharaj')
result1 = embedding.embed_documents(documents)


print(str(result))
print(str(result1))
