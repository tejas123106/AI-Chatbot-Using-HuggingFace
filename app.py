import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load model only once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

st.markdown(
    """
    ### Welcome! 👋
    I'm your AI assistant, powered by **Qwen2.5-0.5B-Instruct**.

    Ask me anything about:
    - Artificial Intelligence
    - Programming and Python
    - Machine Learning
    - General knowledge

    How can I help you today?
    """
)

st.divider()
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    st.rerun()

max_tokens = st.sidebar.slider(
    "Maximum Response Length",
    min_value=50,
    max_value=500,
    value=150,
    step=50
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

top_k = st.sidebar.slider(
    "Top-K",
    min_value=1,
    max_value=100,
    value=50
)

top_p = st.sidebar.slider(
    "Top-P",
    min_value=0.1,
    max_value=1.0,
    value=0.9,
    step=0.05
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write("**Model:** Qwen2.5-0.5B-Instruct")
st.sidebar.write("**Architecture:** Decoder-only Transformer")
st.sidebar.write("**Framework:** Hugging Face Transformers")
st.sidebar.write("**Interface:** Streamlit")
st.sidebar.write("**Device:** CPU" if not torch.cuda.is_available() else "Device: GPU")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Download chat history
chat_text = "\n\n".join(
    f"{message['role'].capitalize()}: {message['content']}"
    for message in st.session_state.messages
    if message["role"] != "system"
)

st.sidebar.download_button(
    label="📥 Download Chat History",
    data=chat_text,
    file_name="chat_history.txt",
    mime="text/plain"
)

# User input
user_input = st.chat_input("Ask me anything...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            try:
                inputs = tokenizer.apply_chat_template(
                    st.session_state.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                )

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                response = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                st.write(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")
