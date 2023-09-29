import csv
import sys

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 巨大なcsvを扱うためにエラー回避
csv.field_size_limit(sys.maxsize)

# 日本語 splitter 
# https://www.sato-susumu.com/entry/2023/04/30/131338
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import pandas as pd

class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)

# load
df = pd.read_csv("./data/redmine_db.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
df.head()

loader = DataFrameLoader(df, page_content_column="journals")
data = loader.load()

# multilingual-e5-large は 512 まで
text_splitter = JapaneseCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0,
)

docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Stores information about the split text in a vector store
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_issues_512")
vectorstore.persist()

