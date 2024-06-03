import streamlit as st

prompt = st.chat_input("Diga alguma coisa")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")