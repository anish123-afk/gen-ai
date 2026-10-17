from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

model = ChatOllama(model = 'qwen3.5:4b')

class feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description='Gives the sentiment of the review')

parser1 = PydanticOutputParser(pydantic_object= feedback)

parser2 = StrOutputParser()

template1 = PromptTemplate(
    template='Based on the given review find its sentiment, \'positive\' or \'negative\'\n{review}\n{format_instructions}',
    input_variables=['review'],
    partial_variables={
        'format_instructions' : parser1.get_format_instructions()
    }
)

template2 = PromptTemplate(
    template='Write a appropriate response for this positive reviw, in short 4-5 lines.\n{review}',
    input_variables=['review']
)

template3 = PromptTemplate(
    template='Write a appropriate response for this negative reviw, in short 4-5 lines.\n{review}',
    input_variables=['review']
)


classifier_chain = template1 | model | parser1

chain1 = template2 | model | parser2

chain2 = template3 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', chain1),
    (lambda x: x.sentiment == 'negative', chain2),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({
    'review' : 'I honestly expected much more from this product. The design looks good at first, but once you actually start using it, the problems become pretty obvious. The performance is inconsistent, the build quality feels cheaper than expected, and some of the features seem poorly thought out.'
})

print(result)