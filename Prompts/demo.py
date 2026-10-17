# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.loading import load_prompt

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = 'Qwen/Qwen3.8-27B',
#     task = 'text-generation'
# )

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

st.header("Research Tool")

paper_input = st.text_input("Enter the name of research paper that you want to study")

style_input = st.selectbox("Select the explanation type", ["Beginner-friendly", "Technical", "Code oriented", "Mathematical"])

length_input = st.selectbox("Select explanation length", ["Short(1 - 2 paragraph)", "Medium(3 - 5 paragraph)", "Long (more than 5 paragraphs)"])

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input' : paper_input,
#     'style_input' : style_input,
#     'length_input' : length_input
# })

if st.button('Submit'):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
    })
#     result = model.invoke(prompt)
    st.markdown(result.content)

# print(model.invoke("hello i am from maharastra!").content)