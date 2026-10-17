from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages  import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3.8-27B',
    task='chat-bot'
)

model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content='This is a chatbot developed by ANISH on the QWEN model'),
]

while True:
    user_input = input("User : ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "exit":
        break

    result = model.invoke(chat_history)
    print(f"AI : {result.content.strip()}")

    chat_history.append(AIMessage(content= result.content.strip()))