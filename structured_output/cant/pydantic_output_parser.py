from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

class person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt = 18, description='Age of the person')
    city : str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='Generate a name, age and city of a fictional character which belongs to {country}.\n{format_instructions}',
    input_variables=['country'],
    partial_variables={
        'format_instructions' : parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({
    'country' : 'india'
})

print(result)