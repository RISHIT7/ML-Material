from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

# model using model_id
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = HuggingFacePipeline.from_model_id(
    model_id = model_id,
    task = "text-generation",
    device = 0,
    model_kwargs={"temperature": 0.5, "do_sample": True},
    pipeline_kwargs={"max_new_tokens": 500}
)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = "I want to open a new restaurant for {cuisine} cuisine. Can you suggest ONLY ONE name for my restaurant?, Give only name, nothing else!"
    )
    
    chain = prompt_template_name | model
    name = chain.invoke({'cuisine': cuisine})
    
    # Chain 2: Menu items
    prompt_template_name = PromptTemplate(
        input_variables= ['restaurant_name'],
        template = "Suggest some menu items for {restaurant_name} No sentence, only name. Return it as a comma seperated string"
    )
    
    chain = prompt_template_name | model
    menu_items = chain.invoke({'restaurant_name': cuisine})
    
    return {
        'restaurant_name': name,
        'menu_items': menu_items
    }

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Indian")['restaurant_name'])
    print()
    print(generate_restaurant_name_and_items("Indian")['menu_items'])
