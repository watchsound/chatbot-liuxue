"""
the purpose is to set up a different vector database, for important topic
input is a file, with format
##topic 
topic detail
##topic
topic detail

output is a vector database, and a set of files
in vector databae, it contains all topics with format:  order. topic
topic detail is stored in the same directory, with order as file name

when we fetch a topic, we can get topic detail with order value

"""

import os
import argparse
import time

import dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.document_loaders import TextLoader

from langchain.schema  import Document

from StringUtils import StringUtils

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()



## split original file
def load_origin_file(file):
    documents = []
    topicdetail = ""
    filename_noext =  file.rsplit(".", 1)[0]
   # print(f" filename_noext = {filename_noext} ")
    os.makedirs(filename_noext)
    try:
        order = 0
        with open(file, "r") as f:
            lines = f.readlines() 
           
            for line in lines:
                line = line.strip()
              #  print(f" line = {line} ")
                if len(line) == 0:
                    continue
                if line.startswith("##") :
                    order += 1
                    documents.append(
                        Document(page_content=f"{str(order)}. {line[2:]}", metadata={'source':order}))
                    if order > 1:
                        afilename = filename_noext + '\\' + str(order-1)
                        print(f" afilename ={afilename}")
                        with open(afilename, 'w') as f2:
                            f2.write(topicdetail)
                    topicdetail = ""
                else:
                    topicdetail += line + "\n"
            
            if order > 1 and len(topicdetail) > 0:
                    with open(filename_noext + '\\' + str(order), 'w') as f2:
                            f2.write(topicdetail)
    except FileNotFoundError:
        return None
    # with open(file + '-topics', 'w') as f2:
    #     f2.write(topics) 
    return documents


print(f" filename = {args.filename} ")
docs = load_origin_file(args.filename)

#print(f" doc = {docs} ")
# loads .env file with your OPENAI_API_KEY
dotenv.load_dotenv()
  
faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
filename_noext = args.filename.rsplit(".", 1)[0]
faissIndex.save_local(filename_noext)
