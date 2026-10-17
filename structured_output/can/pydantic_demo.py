from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

#schema
class Review(BaseModel):

    summary : str = Field(description='A breif summary of the review')
    sentiment : Literal['positive', 'negative', 'neutral'] = Field(description='Return the sentiment of the review, either negative, positive or neutral')
    key_points : list[str] = Field(description='Write the key points of the review, in a list')
    pros : Optional[list[str]] = Field(description='Write down all the pros, in a list', default=None)
    cons : Optional[list[str]] = Field(description='Write down all the cons, in a list', default= None)
    name : Optional[str] = Field(description='Name of the person', default=None)
    

structured_model = model.with_structured_output(Review)

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


result = structured_model.invoke(prompt)


print(result)

