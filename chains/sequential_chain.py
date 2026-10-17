from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Extract the few key point from \n{text}',
    input_variables=['text']
)

model = ChatOllama(model = 'qwen3.5:4b')

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    'topic' : 'football'
})

print(result)