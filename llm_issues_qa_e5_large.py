
from uuid import UUID
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI

# ベクトルストアの初期化
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(persist_directory="./chroma_issues_512", embedding_function=embeddings)

# Question
user_query = "Chromium起動時のメモリ消費改善の方法をしりたい"


# クエリの自動生成
# LlamaCppモデルの初期化
#'''
llm = LlamaCpp(
    #model_path="./models/vicuna-13b-v1.5-16k.Q4_K_M.gguf",
    model_path="./models/vicuna-7b-v1.5.Q4_K_M.gguf",
    #model_path="./models/llama-2-7b-chat.q4_K_M.gguf",
    #model_path="./models/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf",
    verbose=True,
    n_ctx=4096,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    streaming=True,
)
#'''

import config
config.load()
#llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), streaming=True,)
#llm = ChatOpenAI(model_name="gpt-4",temperature=0)

prompt_template = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions seperated by newlines.

# Question: 
{question}

# Response:
Answer in Japanese.
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)

from typing import Any, List, Optional, Sequence
from langchain import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
# https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
# https://tensorflow.classcat.com/2023/09/08/langchain-modules-data-connection-retrievers-multi-query-retriever/
# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)
    
output_parser = LineListOutputParser()

# for MultiQueryRetriver with own prompt
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)

# MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
retriver = MultiQueryRetriever(retriever=vectorstore.as_retriever(), llm_chain=chain, parser_key="lines")

retriver = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# search
docs = retriver.get_relevant_documents(user_query)

print(docs)

# searched list
"""
for doc in docs:
    content = doc.page_content
    print("#### ================================")
    print(doc.metadata['subject'])
    print(f"https://gate.tok.access-company.com/redmine/issues/{int(doc.metadata['id'])}")
    print(doc.page_content)
    print("================================")
"""

