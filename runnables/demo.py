import random
from abc import ABC, abstractmethod

class runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass

class nakliLLM(runnable):

    def __init__(self):
        print('LLM Created')

    def invoke(self, prompt):
        response_list = [
                    'Delhi is a capital of India',
                    'IPL stands for indian premier league',
                    'this is a dummy llm created by campusx',
                    'anish chutya h '
                ]
        
        return {'response' : random.choice(response_list)}


llm = nakliLLM()
# print(llm.predict('What iis thi'))



class NakliPromptTemplate(runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTemplate(
    template='Write a {length} poem on topic \'{topic}\'',
    input_variables=['length', 'topic']
)


# prompt = template.format({'length' : 'Short', 'topic' : 'cricket'})
# print(prompt)

# print(llm.predict(prompt))

class connector(runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            runnable.invoke(input_data)

# class Chain:
#     def __init__(self, llm, prompt):
#         self.llm = llm
#         self.prompt = prompt

#     def run(self, input_dict):
#         final_prompt = self.prompt.format(input_dict)
#         result = self.llm.predict(final_prompt)

#         return result

# chain = Chain(llm, template)
# print(chain.run({'length' : 'long', 'topic' : 'cricket'}))


chain = connector([template, llm])

print(chain.invoke({'length' : 'long', 'topic' : 'india'}))