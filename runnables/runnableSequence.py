from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence


template1 = PromptTemplate(
    template='Write a joke on {topic} in {language}',
    input_variables=['topic', 'language']
)

template2 = PromptTemplate(
    template='Explain the following joke in one line - {text}',
    input_variables=['text']
)

model = ChatOllama(model='gemma3:4b')

parser = StrOutputParser()

chain = RunnableSequence(template1,  model, parser, template2, model, parser)

result = chain.invoke({
    'topic' : 'pregnancy',
    'language' : 'english'
})


print(result)

