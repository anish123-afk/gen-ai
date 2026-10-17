from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(chunks)"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 50,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))