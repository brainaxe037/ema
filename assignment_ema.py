
from google.colab import drive
drive.mount('./content/')

cd '/content/'

!pip install PdfReader
!pip install langchain
!pip install PyPDF2
!pip install InstructorEmbedding
!pip install sentence_transformers
!pip install faiss
!pip install faiss-gpu

!pip install langchain-community

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from langchain.prompts.prompt import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Path to the PDF file
path_to_pdf = ['./content/MyDrive/dataset_ema.pdf']

# Extract text from PDF
raw_text = get_pdf_text(path_to_pdf)

def retrieval_qa_chain(db, return_source_documents):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 500, "max_new_tokens": 700})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db,
                                           return_source_documents=return_source_documents,
                                           )
    return qa_chain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "you API token"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

text_chunks = get_text_chunks(raw_text)

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

!pip uninstall sentence-transformers
!pip install sentence-transformers==2.2.2

vectorstore = get_vectorstore(text_chunks)

db = vectorstore.as_retriever(search_kwargs={'k': 3})

bot = retrieval_qa_chain(db,True)

query = "what is  language model?"
sol=bot(query)
## answer given by llm
print(sol['result'])

print(sol['source_documents'])

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.7,"max_length":500, "max_new_tokens":700})

llm(query)