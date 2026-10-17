from langchain_community.document_loaders import TextLoader

loader = TextLoader("example.txt")

documents = loader.load()

print(len(documents))
print(documents[0].metadata)
print(documents[0].page_content)