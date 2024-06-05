from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os
os.environ["OPENAI_API_KEY"] = "sk-7j6askdjhfjsudfjsdfnjsdl"

llm = OpenAI(temperature=0.5)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = "I want to open a new restaurant for {cuisine} cuisine. Can you suggest a name for my restaurant?"
    )
    
    name_chain = LLMChain(llm, prompt_template_name, output_key = "restaurant_name")
    
    # Chain 2: Menu items
    prompt_template_name = PromptTemplate(
        input_variables= ['restaurant_name'],
        template = "Suggest some menu items for {restaurant_name}. Return it as a comma seperated string"
    )
    
    food_item_chain = LLMChain(llm, prompt_template_name, output_key = "menu_items")
    
    chain = SequentialChain (
        [name_chain, food_item_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items']
    )
    
    response = chain({'inputs': {'cuisine': cuisine}})
    
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Indian"))