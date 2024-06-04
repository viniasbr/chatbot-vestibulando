import getpass
import os

import random
import string


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('.env', 'r') as file:
    data = file.read()

os.environ["OPENAI_API_KEY"] = data #this is to make testing easier
#os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt = "Chave de API da OpenAI: ")


print("\n",end="")

model = ChatOpenAI(model="gpt-3.5-turbo")

### Loading Vector Database ###

vectorstore = Chroma(persist_directory="./database/chroma_db", embedding_function=OpenAIEmbeddings())

### Creating Retriever ###
retriever = vectorstore.as_retriever()

# History Context
# If the question requires context from previous messages,
# this chain reformulates the question including necessary context,
# and passes it to the next chain

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

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Question Answering Chain

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

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


###Volatile Memory Chat History Storage###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def filter_history(history, k = 10):
    return history[-k:]

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

conversational_rag_chain = RunnableWithMessageHistory(
    (RunnablePassthrough.assign(chat_history=lambda x: filter_history(x["chat_history"], k = 6)) | rag_chain), # k = 6 prevents the prompts from getting too long. The chat memory can be short spanned.
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user_input = ""
config = {"configurable": {"session_id": generate_random_string(6)}}
print("Assistente: Eu sou o assistente virtual para o vestibular da Unicamp de 2024. Como eu posso ajudar?")
user_input = input("\nUsuário: ")
print("\nAssistente:",end=" ")
for r in conversational_rag_chain.stream(
    {"input": "Context: Eu sou um assistente virtual para o vestibular da Unicamp de 2024. Como eu posso ajudar?" + '\nQuestion: ' + user_input},
    config=config,
    ):
    if('answer' in r.keys()):
      print(r['answer'], end="")
print("\n",end="")


while user_input != "Fim.":
  user_input = input("\nUsuário: ")
  print("\nAssistente:",end=" ")
  for r in conversational_rag_chain.stream(
      {"input": user_input},
    config=config,
    ):
    if('answer' in r.keys()):
      print(r['answer'], end="")
  print("\n",end="")