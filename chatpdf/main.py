# 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
# https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#loader
loader = PyPDFLoader("data/unsu2.pdf")
pages = loader.load_and_split()

#split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

#embedding
embedding_model = OpenAIEmbeddings()

#load it into Chroma
db = Chroma.from_documents(texts, embedding_model)

#Question
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatOpenAI(temperature = 0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), 
    llm=llm
    )

# docs = retriever_from_llm.get_relevant_documents(query=question)
docs = retriever_from_llm.invoke(input=question)

print(len(docs))
print(docs)