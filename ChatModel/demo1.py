from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3.8-27B',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("Who was chatrapati shivaji maharaj, and for what cause he fought wars?")


print(result.content)