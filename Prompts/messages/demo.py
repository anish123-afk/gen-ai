from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3.8-27B',
    task='chat-bot'
)

model = ChatHuggingFace(llm = llm)

messges = [
    SystemMessage(content='This is a chatbot'),
    HumanMessage(content='hi i am anish')
]

result = model.invoke(messges)

messges.append(AIMessage(content=result.content))

print(messges)