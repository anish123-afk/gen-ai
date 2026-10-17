from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a {domain} expert and give answers only to the questions related to {domain}.'),
    ('human', 'Who is {topic}?')
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'MS Dhoni'
})

print(prompt)