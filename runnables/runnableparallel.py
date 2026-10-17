from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableSequence

prompt1 = PromptTemplate(
    template='Generate  me only one tweet regarding - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate me one one linkedin post regarding - {topic}',
    input_variables=['topic']
)

model = ChatOllama(model='gemma3:4b')

parser = StrOutputParser()


chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({'topic' : 'AI'})

print(result['tweet'])
print(result['linkedin'])