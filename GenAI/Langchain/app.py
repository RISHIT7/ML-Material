import streamlit as st # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
# from ctransformers import CTransformers # type: ignore
from transformers import pipeline, LlamaTokenizer# , LlamaForCausalLM # type: ignore

## Funtion to get response from the Llama 3 8B
def get_Llama_response(input_text: str, no_words: int, blog_style: str) -> str:
    ### Load the model
    llm = pipeline(
        task = 'text-generation',
        model = './Llama-3-8b-instruct',
        model_type='llama',
        trust_remote_code = True,
        use_safetensors = True 
    )
    tokenizer = LlamaTokenizer.from_pretrained(model="./Llama-3-8b-instruct")

    ## Prompt Template
    
    template = """
        Write a blog for {blog_style} jog for a topic {input_text}
        within {no_words} words.
    """
    
    prompt_template = PromptTemplate(
        input_variables = ['input_text', 'no_words', 'blog_style'],
        template = template
    )
    
    ## Generate the reponse from the Llama 3 8B
    response = llm(prompt_template.format
            (
                input_text=input_text, 
                no_words=no_words, 
                blog_style=blog_style
            )
        )
    
    print(response)
    return response

st.set_page_config(
    page_title="Llama 3 8B", 
    page_icon="🦙", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Llama 3 8B 🦙")

input_text = st.text_input("Enter the Blog Topic")

## creating two more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No Of Words')
with col2:
    blog_style = st.selectbox('Writing the Blog for', ('Researchers', 'DataScientists', 'Developers', 'Students'), index = 0)
    
submit = st.button('Generate')

## Final response
if submit:
    st.write(get_Llama_response(input_text, no_words, blog_style))