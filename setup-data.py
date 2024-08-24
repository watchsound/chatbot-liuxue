import dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter 
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.document_loaders import TextLoader


import argparse

# loads .env file with your OPENAI_API_KEY
dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()

#import { TextLoader } from "langchain/document_loaders/fs/text 
loader = TextLoader(args.filename)
documents = loader.load();

text_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=1000, chunk_overlap=0)
#text_splitter = TokenTextSplitter( chunk_size=800, chunk_overlap=0)


docs = text_splitter.split_documents(documents) 
faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())

filename_noext = args.filename.rsplit(".", 1)[0]
faissIndex.save_local(filename_noext)
 

