from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableBranch

model = ChatOllama(model='gemma3:4b')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a detailed report on topic - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a summary of report - {report}',
    input_variables=['report']
)

report_gen = RunnableSequence(prompt1, model, parser)

conditional_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen, conditional_chain)


result = chain.invoke({'topic' : 'AI'})

print(result)