from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

loader = PyPDFLoader('pdf.pdf')

docs = loader.load()

# print(len(docs))

# print(docs[0].metadata)
# print(docs[0].page_content)


model = ChatOllama(model = 'gemma3:4b')

prompt = PromptTemplate(
    template='From the give text - {text}, answer the following question - {que}',
    input_variables=['text', 'que']
)

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({
    'text' : docs[0].page_content,
    'que' : 'What is the definition of intrapersonal communication ?'
})


print(result)