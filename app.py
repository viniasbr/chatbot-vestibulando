import os

import streamlit as st

#Seleção da versão correta do sqlite3 para o streamlit

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

#Definição das configurações do streamlit

st.set_page_config(page_title="ChatVest", page_icon="📖", layout="centered", initial_sidebar_state="expanded", menu_items=
                   {
                    'About': 'https://github.com/viniasbr/chatbot-vestibulando',
                    'Report a Bug': 'https://github.com/viniasbr/chatbot-vestibulando/issues/new'
                   })

st.title("Chatbot Vestibulando 📖")

if "is_key_valid" not in st.session_state: #Parâmetro que vai impedir o uso da aplicação se a chave de API estiver claramente errada
    st.session_state.is_key_valid = False

with st.sidebar: #Coletar chave de API
    st.header("Olá! Para começar, insira sua chave de API da OpenAI:")
    with st.form(key='Chave API OpenAI'):
        possible_key = st.text_input('Chave API OpenAI', type='password')
        submitted = st.form_submit_button("Enviar", type="primary")
    st.caption("Pode ficar em paz! Sua chave de API _nunca_ será guardada. Ela é usada somente na memória volátil desse site, e será apagada no instante que a guia for fechada ou recarregada!")

@st.experimental_dialog("Saudações!") #Diálogo de entrada
def welcome():
    st.markdown("Esse é o **Chatbot Vestibulando**! Ele é um assistente baseado em **ChatGPT-3.5** que responde \
                dúvidas sobre o **Vestibular da Unicamp de 2024**. Ele usa um sistema chamado _Retrieval Augmented Generation_, que busca as \
                informações necessárias diretamente da [Resolução GR-031/2023](https://www.pg.unicamp.br/norma/31594/0).\
                Por favor, insira sua chave de API da OpenAI para usar a aplicação. Boa experiência! 😀")
    st.session_state.welcome_dialog_shown = True
    if st.button("Entendi!"):
        st.rerun()

if "welcome_dialog_shown" not in st.session_state: #Garantir que diálogo só vai ser mostrado uma vez
    st.session_state.welcome_dialog_shown = False
    welcome()

if submitted: #Testa se a chave não está claramente errada e avisa o usuário
    if not possible_key.startswith('sk-'):
        st.sidebar.warning('Chave de API OpenAI inválida!', icon = '⚠')
    else:
        st.sidebar.success("Chave válida! Pode começar a perguntar!", icon = "😅")
        st.session_state.is_key_valid = True
        if "api_key" not in st.session_state: #Salva a chave de API entre reruns do streamlit
            st.session_state.api_key = possible_key

if st.session_state.is_key_valid:
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key #Carrega a chave de API para o uso
    if "vectorstore" not in st.session_state: #Carregar banco de dados vetorial para Retrieval e modelo de embeddings
        st.session_state.vectorstore = Chroma(persist_directory="./database/chroma_db", embedding_function=OpenAIEmbeddings())
    if "model" not in st.session_state: #Modelo de LLM
        st.session_state.model = ChatOpenAI(model="gpt-3.5-turbo")
    if "store" not in st.session_state: #Dicionário para o histórico do chat para o LLM
        st.session_state.store = {}
    if "messages" not in st.session_state: #Lista para o histórico do chat visual
        st.session_state.messages = []

    def get_session_history(session_id: str) -> BaseChatMessageHistory: #Função para recuperar o histórico de um usuário dado ID
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    def filter_history(history, k = 6): #Função para limitar o histórico e não estourar a quantidade de tokens por prompt
        return history[-k:]
    
    for message in st.session_state.messages: #Imprime as mensagens já enviadas, na tela, a cada rerun
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if "history_aware_retriever" not in st.session_state: #Chain que reformula o contexto do retriever considerando o histórico do chat
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
            st.session_state.model, st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6}), contextualize_q_prompt
        )
    if "conversational_rag_chain" not in st.session_state: #Chain principal que gera resposta dado o contexto
        qa_system_prompt = """You're Chatbot Vestibulando, the assistant for question-answering tasks specifically about \
        the vestibular da Unicamp, and you speak in portuguese. You answer questions about the \
        vestibular da Unicamp of 2024. Your informations are taken from Resolução GR-031/2023, \
        which is available at https://www.pg.unicamp.br/norma/31594/0. \
        You answer about any course or piece of content that might be on the document. You are not limited to any specific course areas. \
        Use the following pieces of retrieved context to answer questions about vestibular da Unicamp of 2024. \
        Make sure your answers are thoroughly consistent with the context. Inspect it carefully before giving an answer\
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

        rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, question_answer_chain) #Concatena as chains necessárias para produzir a chain final
        st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
            (RunnablePassthrough.assign(chat_history=lambda x: filter_history(x["chat_history"], k = 6)) | rag_chain), # k = 6 prevents the prompts from getting too long. The chat memory can be short spanned.
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    def response_generator(input_text: str): #Wrapper que faz o streaming da resposta
        config = {"configurable": {"session_id": "abc123"}}
        for chunk in st.session_state.conversational_rag_chain.stream({"input": input_text},config=config,):
            if('answer' in chunk.keys()):
                yield str(chunk["answer"])

    

if prompt := st.chat_input(placeholder="Faça uma pergunta",disabled= not st.session_state.is_key_valid): #Lógica do chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): #Anota o prompt na tela
        st.markdown(prompt)
    with st.chat_message("assistant"): #Faz streaming da resposta dado um prompt
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
