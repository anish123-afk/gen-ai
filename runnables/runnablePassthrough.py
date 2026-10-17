from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

model = ChatOllama(model='gemma3:4b')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a joke on topic - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate the explaination of the following joke - {joke}',
    input_variables=['joke']
)

joke_gen = RunnableSequence(prompt1, model, parser)

main_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen, main_chain)


result = chain.invoke({'topic' : 'cricket'})

print(result)