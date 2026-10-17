from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

def word_count(text):
    return len(text.split())

model = ChatOllama(model='gemma3:4b')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate a joke on topic - {topic}',
    input_variables=['topic']
)

joke_gen = RunnableSequence(prompt, model, parser)

main_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

chain = RunnableSequence(joke_gen, main_chain)


result = chain.invoke({'topic' : 'cricket'})

print(result)