from langchain_community.document_loaders import WebBaseLoader


url = 'https://www.amazon.in/EvoFox-One-Universal-Wireless-Controller/dp/B0DZ6T16SD/ref=sr_1_3?sr=8-3'

loader = WebBaseLoader(url)

docs = loader.load()


print(len(docs))
print(docs[0])

