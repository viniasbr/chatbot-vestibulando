import os

import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="ChatVest", page_icon="📖", layout="centered", initial_sidebar_state="expanded", menu_items=
                   {
                    'About': 'https://github.com/viniasbr/chatbot-vestibulando',
                    'Report a Bug': 'https://github.com/viniasbr/chatbot-vestibulando/issues/new'
                   })

st.title("Chatbot Vestibulando 📖")

if "is_key_valid" not in st.session_state: 
    st.session_state.is_key_valid = False

with st.sidebar:
    st.header("Olá! Para começar, insira sua chave de API da OpenAI:")
    with st.form(key='Chave API OpenAI'):
        possible_key = st.text_input('Chave API OpenAI', type='password')
        submitted = st.form_submit_button("Enviar", type="primary")
    st.caption("Pode ficar em paz! Sua chave de API _nunca_ será guardada. Ela é usada somente na memória volátil desse site, e será apagada no instante que a guia for fechada ou recarregada!")

@st.experimental_dialog("Saudações!")
def welcome():
    st.markdown("Esse é o **Chatbot Vestibulando**! Ele é um assistente baseado em **ChatGPT-3.5** que responde \
                dúvidas sobre o **Vestibular da Unicamp de 2024**. Ele usa um sistema chamado _Retrieval Augmented Generation_, que busca as \
                informações necessárias diretamente da [Resolução GR-031/2023](https://www.pg.unicamp.br/norma/31594/0).\
                Por favor, insira sua chave de API da OpenAI para usar a aplicação. Boa experiência! 😀")
    st.session_state.welcome_dialog_shown = True
    if st.button("Entendi!"):
        st.rerun()

if "welcome_dialog_shown" not in st.session_state:
    st.session_state.welcome_dialog_shown = False
    welcome()

if submitted:
    if not possible_key.startswith('sk-'):
        st.sidebar.warning('Chave de API OpenAI inválida!', icon = '⚠')
    else:
        st.sidebar.success("Chave válida! Pode começar a perguntar!", icon = "😅")
        st.session_state.is_key_valid = True
        if "api_key" not in st.session_state:
            st.session_state.api_key = possible_key

if st.session_state.is_key_valid:
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = Chroma(persist_directory="./database/chroma_db", embedding_function=OpenAIEmbeddings())
    if "model" not in st.session_state:
        st.session_state.model = ChatOpenAI(model="gpt-3.5-turbo")
    if "store" not in st.session_state:
        st.session_state.store = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    def filter_history(history, k = 6):
        return history[-k:]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if "history_aware_retriever" not in st.session_state:
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        st.session_state.history_aware_retriever = create_history_aware_retriever(
            st.session_state.model, st.session_state.vectorstore.as_retriever(), contextualize_q_prompt
        )
    if "conversational_rag_chain" not in st.session_state:
        qa_system_prompt = """You're the assistant for question-answering tasks specifically about \
        the vestibular da Unicamp, and you speak in portuguese. You answer questions about the \
        vestibular da Unicamp of 2024. Your informations are taken from Resolução GR-031/2023, \
        which is available at https://www.pg.unicamp.br/norma/31594/0. \
        Use the following pieces of retrieved context to answer questions about vestibular da Unicamp of 2024. \
        If you don't know the answer, just say that you don't know. \
        Do not stray too far from the theme. Use three sentences maximum and keep the answer concise.\n\n {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(st.session_state.model, qa_prompt)

        rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, question_answer_chain)
        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
            (RunnablePassthrough.assign(chat_history=lambda x: filter_history(x["chat_history"], k = 6)) | rag_chain), # k = 6 prevents the prompts from getting too long. The chat memory can be short spanned.
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    def response_generator(input_text: str):
        config = {"configurable": {"session_id": "abc123"}}
        for chunk in st.session_state.conversational_rag_chain.stream({"input": input_text},config=config,):
            if('answer' in chunk.keys()):
                yield str(chunk["answer"])

    

if prompt := st.chat_input(placeholder="Faça uma pergunta",disabled= not st.session_state.is_key_valid):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
