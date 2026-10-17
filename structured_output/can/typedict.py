# from typing import TypedDict


# class person(TypedDict):

#     name : str
#     age : int

# p : person = {
#     'name' : 'anish',
#     'age' : '18'
# }

# print(p)

from typing import TypedDict, Annotated, Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V4-Flash-0731',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

#schema
class review(TypedDict):
    summary : Annotated[str, 'A breif summary of the review']
    sentiment : Annotated[str, 'Return the sentiment of the review, either negative, positive or neutral']
    key_points : Annotated[list[str], 'Write the key points of the review, in a list']
    pros : Annotated[Optional[list[str]], 'Write down all the pros, in a list']
    cons : Annotated[Optional[list[str]], 'Write down all the cons, in a list']

structured_model = model.with_structured_output(review)

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

