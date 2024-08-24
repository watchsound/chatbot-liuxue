"""Create a ChatVectorDBChain for question/answering.""" 
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
#from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from langchain.prompts.prompt import PromptTemplate
 
def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        #tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        #model_name="gpt-3.5-turbo"
        max_tokens=1000,
        #model_name="gpt-4",
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
       # model_name="gpt-4",
        max_tokens=1000,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    #CONDENSE_QUESTION_PROMPT has input_variables=['chat_history', 'question'],
    _template = """您是出国留学中介的工作人员，提供出国留学咨询。 
在聊天中，在每次回答结束前尽可能引导对话提供如下五类信息： 
1. 姓名。
2. 电话号码，微信号，或者其他个人联系方式。 
3. 托福,雅思,或者其他GPA成绩。
4. 想申请哪些学校、专业。
5. 对留学有什么具体的要求或期望。

在对方的提问中要尽可能找到上述五类信息。
  
会话的历史记录会提供给您，同时提供一个问题，请用一到两句中文来完成会话。
最后以提供更好服务的理由来提醒对方补充没有提供的信息。

会话历史记录:
{chat_history}
后续问题: {question} 


请用汉语回答。您返回的结果分两部分
第一部分是您对问题的回答，用简单文本格式

第二部分无需标题，直接列出您已经获得的学生信息，用如下格式表示, 如果没有信息，"="后面空白不填写内容：

姓名=
联系方式=
成绩=
申请学校和专业=
具体要求= 
"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #QA_PROMPT has  template="Q: {question} A:",
    template = """您是出国留学中介的工作人员，提供出国留学咨询。 
在聊天中，在每次回答结束前尽可能引导对话提供如下五类信息： 
1. 姓名。
2. 电话号码，微信号，或者其他个人联系方式。 
3. 托福,雅思,或者其他GPA成绩。
4. 想申请哪些学校、专业。
5. 对留学有什么具体的要求或期望。

在对方的提问中要尽可能找到上述五类信息。

你会給一段相关内容和一个问题。这段内容是你优先考虑的答案，做为你思考的一部分， 
如果相关内容中找不到答案，你可以使用你储备的相关知识，提供一句话的聊天式的回答。 
最后以提供更好服务的理由来提醒对方补充没有提供的信息。
  
请用中文回答。
 

问题: {question}
=========
{context}
=========




请用汉语回答。您返回的结果分两部分
第一部分是您对问题的回答，用简单文本格式

第二部分无需标题，直接列出您已经获得的学生信息，用如下格式表示，如果没有信息，"="后面空白不填写内容：

姓名=
联系方式=
成绩=
申请学校和专业=
具体要求= 
"""
    
    QA = PromptTemplate(template=template, input_variables=[ 
                        "question", "context"])

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA, callback_manager=manager
    )

    #this post explain how to use ConversationalRetrievalChain.   
    #we did not pass any memory model here,  in order to use chart history later on.
    #https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html


    qa = ConversationalRetrievalChain(
        #vectorstore=vectorstore,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa
