from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model1 = ChatOllama(
    model = 'qwen3.5:4b'
)

model2 = ChatOllama(
    model = 'qwen3.5:4b'
)

template1 = PromptTemplate(
    template='Generate the detailed notes for the topic -> {text}',
    input_variables=['text']
)

template2 = PromptTemplate(
    template = 'Generate the quiz on this text -> {text}',
    input_variables=['text']
)

template3 = PromptTemplate(
    template='On the basis of the notes -> {notes} and quiz -> {quiz} given, merge them and give the finished output, cover only the most important things.',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model1 | parser,
    'quiz' : template2 | model2 | parser
})

merged_chain = template3 | model1 | parser

text = """The Bijapur Sultanate was displeased with their losses to Shivaji's forces, with their vassal Shahaji disavowing his son's actions. After a peace treaty with the Mughals, and the general acceptance of the young Ali Adil Shah II as the sultan, the Bijapur government became more stable, and turned its attention towards Shivaji.[49] In 1657, the sultan, or more likely his mother and regent, sent Afzal Khan, a veteran general, to arrest Shivaji. Before engaging him, the Bijapuri forces desecrated the Tulja Bhavani Temple, a holy site for Shivaji's family, and the Vithoba temple at Pandharpur, a major pilgrimage site for Hindus.[50][51][52]

Pursued by Bijapuri forces, Shivaji retreated to Pratapgad fort, where many of his colleagues pressed him to surrender.[53] The two forces found themselves at a stalemate, with Shivaji unable to break the siege, while Afzal Khan, having a powerful cavalry but lacking siege equipment, was unable to take the fort. After two months, Afzal Khan sent an envoy to Shivaji suggesting the two leaders meet in private, outside the fort, for negotiations.[54][55]

The two met in a hut at the foothills of Pratapgad fort on 10 November 1659. The arrangements had dictated each come armed only with a sword, and attended by one follower. Shivaji, suspecting Afzal Khan would arrest or attack him,[56][b] wore armour beneath his clothes, concealed a bagh nakh (metal "tiger claw") on his left arm, and had a dagger in his right hand.[58] What transpired is not known with historical certainty, as mainly Maratha legends tell the tale; however, it is agreed the two wound up in a physical struggle that proved fatal for Khan.[c] Khan's dagger failed to pierce Shivaji's armour, but Shivaji disembowelled him; Shivaji then fired a cannon to signal his hidden troops to attack the Bijapuri army.[60]

In the ensuing Battle of Pratapgarh, Shivaji's forces decisively defeated the Bijapur Sultanate's forces. More than 3,000 soldiers of the Bijapur army were killed; and one sardar of high rank, two sons of Afzal Khan, and two Maratha chiefs were taken prisoner.[61] After the victory, a grand review was held by Shivaji below Pratapgarh. The captured enemy, both officers and soldiers, were set free and sent back to their homes with money, food, and other gifts. The Maratha troops were rewarded accordingly.[61]"""

chain = parallel_chain | merged_chain

result = chain.invoke({
    'text' : text
})

print(result)

chain.get_graph().print_ascii()