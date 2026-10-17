from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

prompt = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors"""

llm = HuggingFaceEndpoint(
    repo_id= 'deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(
        name="summary",
        description="A brief summary of the review"
    ),
    ResponseSchema(
        name="sentiment",
        description="The sentiment: positive, negative, or neutral"
    ),
    ResponseSchema(
        name="key_points",
        description="Important points from the review"
    )
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= 'Give the summary, key_points and sentiment of the review given below. \n{review}\n{format_instructions}',
    input_variables=['review'],
    partial_variables={
        'format_instructions' : parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({
    'review' : prompt
})

print(result)