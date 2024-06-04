import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

with open('.env', 'r') as file:
    data = file.read()

os.environ["OPENAI_API_KEY"] = data

#Loading already parsed data, extracted from https://www.pg.unicamp.br/norma/31594/0

loader = TextLoader("clean_data.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
for i in all_splits:
  i.metadata['source'] = 'https://www.pg.unicamp.br/norma/31594/0'
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")