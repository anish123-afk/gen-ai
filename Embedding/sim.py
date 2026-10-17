from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


cricketers = [
    "Virat Kohli is one of India's greatest modern batsmen and is known for his consistency and aggressive batting style.",
    
    "Rohit Sharma is a right-handed batsman and former captain of the Indian cricket team, famous for his ability to score big hundreds.",
    
    "MS Dhoni is a legendary Indian wicketkeeper-batsman and captain who led India to major ICC tournament victories.",
    
    "Sachin Tendulkar is regarded as one of the greatest batsmen in cricket history and was the first player to score 100 international centuries.",
    
    "Jasprit Bumrah is an Indian fast bowler known for his unique bowling action, accuracy, pace, and excellent performance in all formats.",
    
    "Ravindra Jadeja is an Indian all-rounder who contributes with left-arm spin bowling, aggressive batting, and exceptional fielding.",
    
    "Hardik Pandya is an Indian all-rounder known for powerful batting, fast bowling, and his ability to contribute in crucial matches.",
    
    "KL Rahul is an Indian batsman and wicketkeeper who has represented India across Test, ODI, and T20 cricket.",
    
    "Ravichandran Ashwin is one of India's finest spin bowlers and is particularly known for his variations, accuracy, and tactical approach.",
    
    "Yuvraj Singh is a former Indian all-rounder famous for his powerful batting, six sixes in an over, and important performances in India's 2011 World Cup victory."
]

embedding = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001', output_dimensionality= 100)

query = "who hit 6 6's in 6 ball"

doc_embed = embedding.embed_documents(cricketers)
query_embed = embedding.embed_query(query)

max = 0
index = 0
for i in range(len(doc_embed)):
    t = cosine_similarity([query_embed], [doc_embed[i]])
    if t > max:
        max = t
        index = i

print(cricketers[index])