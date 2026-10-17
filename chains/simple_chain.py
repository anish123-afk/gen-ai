from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

template = PromptTemplate(
    template='generate a summary on {topic}',
    input_variables=['topic']
)

model = ChatOllama(model = "qwen3.5:4b")

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({
    'topic' : 'blackhole'
})

print(result)