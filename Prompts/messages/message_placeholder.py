from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


chat_template = ChatPromptTemplate([
    ('system', 'You are the support agent'),
    MessagesPlaceholder(variable_name='chat_hist'),
    ('human', '{query}')
])

chat_history = []

with open('history.txt') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

prompt = chat_template.invoke({
    'chat_hist' : chat_history,
    'query' : 'no one came to pick up'
})

print(prompt)