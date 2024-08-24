"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from fastapi.middleware.cors import CORSMiddleware

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import AsyncCallbackManager

from langchain.chat_models import ChatOpenAI
 
import os, time
#import dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.memory import ChatMessageHistory

import openai
import tiktoken  # for counting tokens
from StringUtils import StringUtils

import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DEBUG = False
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
max_tokens_limit = 2000

#dotenv.load_dotenv()
 
openai.api_key = os.getenv("OPENAI_API_KEY") 
 

vectorstore: Optional[VectorStore] = None
vectorstore_topics: Optional[VectorStore] = None
 
system_intro_template = """您是出国留学中介的工作人员，提供出国留学咨询。  
"""

 

ai_message_prompt_with_extra_info = """您是出国留学中介的工作人员，提供出国留学咨询。 
你会給一段相关内容和一个问题。这段内容是你优先考虑的答案，做为你提供出国留学咨询的一部分，
如果相关内容中找不到答案，你就回答请直接咨询留学顾问，不要编造内容。  
请用中文回答, 内容长度不要超过300字。 

问题: {question}
=========
{context}
=========

"""
  
   

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


async def print_userinput(websocket, message):
    resp0 = ChatResponse(sender="you", message=message, type="stream")
    await websocket.send_json(resp0.dict())


async def print_botoutput(websocket, message):
    start_resp = ChatResponse(
        sender="bot", message="努力打工中...", type="start")
    await websocket.send_json(start_resp.dict())
    m_resp = ChatResponse(
        sender="bot", message=message, type="stream")
    await websocket.send_json(m_resp.dict())
    end_resp = ChatResponse(sender="bot", message="", type="end")
    await websocket.send_json(end_resp.dict())


async def fetchLocalTopicInfo(websocket, input, doc, chat_history):
    if DEBUG:
        print(f" fetch local topic content = {doc.page_content}")
    order = StringUtils.checkNumberRow(doc.page_content)
    afilename = os.path.join('predefined-topics', str(order) )
    with open(afilename, "r") as f:
        data = f.read().replace('\n', '')
        await print_botoutput(websocket, data)
        chat_history.append({"role": "user", "content": input})
        chat_history.append({"role": "assistant", "content": doc.page_content})


def fetchLocalInfo(input, docs):
    num_docs = len(docs)
    tokens = [
        num_tokens(
            docp[0].page_content)
        for docp in docs
    ]
    token_count = sum(tokens[:num_docs])
    while token_count > max_tokens_limit:
        num_docs -= 1
        token_count -= tokens[num_docs]

    mstr = ""
    for i in range(num_docs):
        mstr += docs[i][0].page_content
    return mstr

 

async def standaloneQueyUserIntent(websocket, input, chat_history):
    standalone_query_template = """
    你是一个提供出国留学中介的AI，请问下面的话最符合哪个主题，或者说和哪个主题的关系最大：

"{userinput}"

主题包括： 
1. 出国留学申请的成功案例，客户反馈
2. 可以解决什么问题，提供的咨询服务范围，公司的业务范围，出国咨询，留学规划咨询
3. 如何收费，服务的费用是怎么计算的，收费标准，咨询费是多少，要花多少钱，费用与优惠
4. 公司介绍，机构背景, 公司简介，你们公司成立了多长时间？你们的工作做了多久？
5. 你们和学校之间有什么合作？和哪些学校之间有合作？
6. 今年六年级小升初,怎么收费
7. 和托福，雅思英语考试相关的话题
8. 姓名，联系方式
9. 微信，微信号，电话
10. GPA，和GPA成绩相关话题
11. 和留学选择专业相关话题
12. 和申请学校相关话题
13. 其他

你只要返回最接近的一个答案。 如果主题都不符合，就回复"13. 其他"
回答的格式：只要列出选项就可以了
 
"""
    if DEBUG:
        print(standalone_query_template.format(userinput=input))

   # messages = [  # {"role": "system", "content": "你是一个提供出国留学中介的AI"},
   #     {"role": "user", "content": standalone_query_template.format(userinput=input)}]
    
    messages = chat_history.copy()
    messages.append({"role": "user", "content": standalone_query_template.format(userinput=input)})
    

    tries = 2
    while True:
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages, temperature=0
            )
        except:
            tries -= 1
            if tries:  # if tries != 0
                time.sleep(20)
                continue  # not necessary, just for clarity
            else:
                raise  # step 4
        else:
            break  # step 2


    reply = chat.choices[0].message.content
    order = StringUtils.checkNumberRow(reply.strip())
    if order > 6:
        return False
    if DEBUG:
        print(f" matched topic order = {str(order)}")

   
    chat_history.append({"role": "user", "content": input})
    await print_userinput(websocket, input)

    afilename = os.path.join('predefined-topics', str(order))
    with open(afilename, "r") as f:
        data = f.read().replace('\n', '')
        #print(data + "\n" + createSuggestion())
        await print_botoutput(websocket, data  )
        dlen = min(100, len(data))
        chat_history.append(
            {"role": "assistant", "content": data[0:dlen]  })

    return True



@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("predefined-materials-embed").exists():
        raise ValueError("predefined-materials-embed does not exist, please run setup-data.py first")
    global vectorstore
    vectorstore =  FAISS.load_local("predefined-materials-embed", OpenAIEmbeddings())
    global vectorstore_topics
    vectorstore_topics = FAISS.load_local(
        "predefined-topics", OpenAIEmbeddings())

    
    #os.environ["LANGCHAIN_TRACING"] = "true"


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    
    chat_history = []
    chat_history.append({"role": "system", "content": system_intro_template})

    await websocket.accept()
      
    useranswers = []
    maxChat = 15
    while True:
        maxChat -= 1;
        if maxChat < 0 :  
            await websocket.close()
            break
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            if len(question) < 2:
                continue
            
            topicFound =  False #await standaloneQueyUserIntent(websocket, question,  chat_history)
            if topicFound:
               maxChat -= 1
               continue
            #check local info
            r0s = vectorstore.similarity_search_with_score(question, 4)
            
            if DEBUG:
                print(f"vectorstore score = {str(r0s[0][1])} ")
 

            if topicFound is False:
                tmessage = chat_history.copy()
                mstr = fetchLocalInfo(question, r0s)
                if DEBUG:
                    print(f" local info is = {mstr}")
                ss = ai_message_prompt_with_extra_info.format(
                    question=question, context=mstr)
                tmessage.append({"role": "user", "content": ss})
                 
                respu = ChatResponse(
                    sender="you", message=question, type="stream")
                await websocket.send_json(respu.dict())
 
                start_resp = ChatResponse(sender="bot", message="努力打工中...", type="start")
                await websocket.send_json(start_resp.dict())

                reply = "" 
                tries = 2
                while True:
                    try:
                        reply = ""
                        for chunk in openai.ChatCompletion.create(
                            messages=tmessage, temperature=0, model="gpt-3.5-turbo",
                            stream=True,
                        ):
                            content = chunk["choices"][0].get(
                                "delta", {}).get("content")
                            if content is not None:
                                reply += content
                                respx = ChatResponse(
                                    sender="bot", message=content, type="stream")
                                await websocket.send_json(respx.dict()) 

                    except:
                        tries -= 1
                        if tries:  # if tries != 0
                            time.sleep(20)
                            continue  # not necessary, just for clarity
                        else:
                            raise  # step 4
                    else:
                        break  # step 2

 

              
                #we set type = 'start', as it should be 'stream', but we do it on purpose
                #to inform the font to clear up state flags 
                start_resp = ChatResponse(
                        sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())

                     
                #print(r[0] + " " + createSuggestion())
                chat_history.append({"role": "user", "content": question})
                chat_history.append(
                    {"role": "assistant", "content": reply  })
  

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

              
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            respex = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(respex.dict())


if __name__ == "__main__":
    import uvicorn
    origins = ['http://localhost:8088', 'http://xxxxx']

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8088)
