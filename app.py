import streamlit as st
from transformers import AutoTokenizer
import torch

from model import Mamba, ModelArgs, generate

mamba_model_names = [
    'state-spaces/mamba-370m',
    'state-spaces/mamba-130m'
]

@st.cache_resource(show_spinner=False)
def load_models_and_tokenizer():
    """Function to load all Mamba models and the tokenizer before showing the UI."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loading all models
    models = {}
    for model_name in mamba_model_names:
        model = Mamba.from_pretrained(model_name, device=device)
        models[model_name] = model
    
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    
    return models, tokenizer

# Show a loading spinner while models and tokenizer are loading
with st.spinner("Loading models and tokenizer, please wait..."):
    models, tokenizer = load_models_and_tokenizer()

# Once loading is complete, display the UI
st.success("All models and tokenizer loaded successfully!")

# Streamlit App UI
st.title("Mamba Language Model Demo")

# Dropdown menu for selecting a model
selected_model_name = st.selectbox("Select a Mamba Model", mamba_model_names)

# Text area for user input
user_input = st.text_area("Enter a prompt", value="", height=100)

# Button to generate the response
if st.button("Generate Answer"):
    # Retrieve the selected model from the models dictionary
    model = models[selected_model_name]
    
    # Generate the answer using the provided prompt
    with st.spinner("Generating..."):
        output = generate(model, tokenizer, user_input)
    
    # Display the generated text
    st.text_area("Generated Answer", value=output, height=200)