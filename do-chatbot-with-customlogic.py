import os, dotenv, time
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

DEBUG = False
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
max_tokens_limit = 2000

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
userinfometa = ['姓名', '联系方式', '成绩', '申请学校和专业', '具体要求']
userinfo = ['', '', '', '', '']
chat_history = []

vectorstore = FAISS.load_local(
    "predefined-materials-embed", OpenAIEmbeddings())

vectorstore_topics = FAISS.load_local(
    "predefined-topics", OpenAIEmbeddings())
  
system_intro_template="""您是出国留学中介的工作人员，提供出国留学咨询。  
在聊天中，在每次回答结束前尽可能引导对话提供如下五类信息： 
1. 姓名。
2. 电话号码，微信号，或者其他个人联系方式。 
3. 托福,雅思,或者其他GPA成绩。
4. 想申请哪些学校、专业。
5. 对留学有什么具体的要求或期望。

在对方的提问中也要尽可能找到上述五类信息。


您返回的结果分两部分
请用汉语回答。您返回的结果分两部分
第一部分是您对问题的回答，用简单文本格式

第二部分无需标题，直接列出您已经获得的学生信息，用如下格式表示, 如果没有信息，"="后面空白不填写内容：

姓名=
联系方式=
成绩=
申请学校和专业=
具体要求= 
"""
chat_history.append({"role": "system", "content": system_intro_template})

ai_message_prompt_template = """
您好！我是出国留学中介的工作人员，非常愿意为您提供有关留学的咨询。在我们的交谈中，请注意避免讨论与出国留学无关的政治话题。

现在，我想了解一些您的具体要求和期望，以便能更好地为您提供帮助。请回答以下问题：

您对留学有什么具体的要求或期望？
您打算申请哪些学校和专业？
您的托福雅思成绩是多少？
您能告诉我您的姓名和微信号，以便我们能够更方便地联系吗？
请提供以上信息，我将尽力回答您的问题和提供相关建议
""" 

ai_message_prompt_with_extra_info =  """您是出国留学中介的工作人员，提供出国留学咨询。 
你会給一段相关内容和一个问题。这段内容是你优先考虑的答案，做为你提供出国留学咨询的一部分，
如果相关内容中找不到答案，你可以使用你储备的相关知识，提供一句话的聊天式的回答。  
请用中文回答。 

问题: {question}
=========
{context}
=========

"""



def needUserInfo():
    return len(userinfo[0].strip()) == 0 or len(userinfo[1].strip()) == 0 or len(userinfo[2].strip()) == 0


def createBotUserInfo( ):
    result = '谢谢你提供的信息。 我记住了'
    if len(userinfo[0].strip()) > 0:
        result += '你的名字叫' + userinfo[0] + ', '
    if len(userinfo[1].strip()) > 0:
        result += '你的联系方式是：' + userinfo[1] + ', '
    if len(userinfo[2].strip()) > 0:
        result += '你的考试成绩是：' + userinfo[2] + ', '
    if len(userinfo[3].strip()) > 0:
        result += '你想申请学校和专业是：' + userinfo[3] + ', '
    if len(userinfo[4].strip()) > 0:
        result += '你留学的具体要求是：' + userinfo[4] + ', '
    #for i in range(5):
    #    if len(userinfo[i].strip()) > 0:
    #        result += userinfometa[i] + '是' + userinfo[i] + ', '
    if len(userinfo[0].strip()) > 0:
        result +=  userinfo[0] + '你好！'
    return result

def createUserInfo():
    result = ''
    for i in range(5):
        if len(userinfo[i].strip()) > 0:
            result += userinfometa[i] + '是' + userinfo[i] + ', '
    return result


def createUserInfo():
    result = ""
    if len(userinfo[0].strip()) > 0:
        result += '我的名字叫' + userinfo[0] + ', '
    if len(userinfo[1].strip()) > 0:
        result += '我的联系方式是：' + userinfo[1] + ', '
    if len(userinfo[2].strip()) > 0:
        result += '我的考试成绩是：' + userinfo[2] + ', '
    if len(userinfo[3].strip()) > 0:
        result += '我想申请学校和专业是：' + userinfo[3] + ', '
    if len(userinfo[4].strip()) > 0:
        result += '我留学的具体要求是：' + userinfo[4] + ', '
    #for i in range(5):
    #    if len(userinfo[i].strip()) > 0:
    #        result += userinfometa[i] + '是' + userinfo[i] + ', '
    return result


def createSuggestion():
    result = ""
    if len(userinfo[0].strip()) == 0:
        result += '姓名'
    if len(userinfo[1].strip()) == 0:
        if len( result ) > 0:
            result += ","
        result +=  '联系方式'
    if len(userinfo[2].strip()) == 0:
        if len(result) > 0:
            result += ","
        result += '英语考试成绩'
    if len(userinfo[3].strip()) == 0:
        if len(result) > 0:
            result += ","
        result += '申请学校和专业' 
    #for i in range(5):
    #    if len(userinfo[i].strip()) > 0:
    #        result += userinfometa[i] + '是' + userinfo[i] + ', '
    if len( result ) > 0:
        return "请留下您的" + result + ", 以方便我们联系"
    return ""


def parseGPT(input):
    if DEBUG: print(f" input ==  {input}")
    hasUserInfo = False
    fn = userinfometa[0]
    pos = input.find(fn + '=')
    flags = [False, False, False, False, False]
    if pos < 0:
        return (input, hasUserInfo, flags)

    userdata = input[pos:]

    for i in range(4):
        fn = userinfometa[i]
        pos2 = userdata.find(fn + '=')
        if pos2 < 0:
            continue
        fn3 = userinfometa[i+1]
        pos3 = userdata.find(fn3 + '=')
        if pos3 < 0:
            continue
        if len(fn) + 2 == pos3:
            continue
        f = userdata[pos2+len(fn)+1: pos3].strip()
        if f.startswith('未提供'):
            continue
        userinfo[i] += f + ' '
        hasUserInfo = True
        flags[i] = True

    fn = userinfometa[4]
    pos2 = userdata.find(fn + '=')
    if pos2 > 0 and userdata.endswith(fn+"=") is False:
        f = userdata[pos2+len(fn)+1:].strip()
        if f.startswith('未提供') is False:
            userinfo[4] += f + ' '
            hasUserInfo = True
            flags[4] = True

    return (input[0: pos], hasUserInfo, flags)


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def fetchLocalTopicInfo(input, doc):
    if DEBUG:
        print(f" fetch local topic content = {doc.page_content}")
    order = StringUtils.checkNumberRow(doc.page_content)
    afilename = 'predefined-topics' + '\\' + str(order)
    with open(afilename, "r") as f:
        data = f.read().replace('\n', '')
        print(data + "\n" + createSuggestion())
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

 


def standaloneQueyUserInfo(input ):
    standalone_query_template = """
    你是一个提供出国留学中介的AI，请在下面的话中分析收集五类信息的数据：

"{userinput}"

五类信息包括： 
1. 姓名。
2. 电话号码，微信号，或者其他个人联系方式。 
2. 托福,雅思,或者其他GPA成绩。
3. 想申请哪些学校、专业。
4. 对留学有什么具体的要求或期望。

分析结果用以下格式返回，如果没有信息，"="后面空白不填写内容
姓名=
联系方式=
成绩=
申请学校和专业=
具体要求= 
"""

    messages = [{"role": "system", "content": "你是一个提供出国留学中介的AI，需要在会话中分析收集咨询出国的人的相关信息 "},
                {"role": "user", "content": standalone_query_template.format(userinput=input)}]
   
    
    
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
     
    r = parseGPT(reply)
    if r[1] is True: 
        chat_history.append({'role': 'user', 'content': createUserInfo()})
        chat_history.append({'role': 'assistant', 'content': createBotUserInfo()})
    
    return r[2]

def standaloneQueyUserIntent(input):
    standalone_query_template = """
    你是一个提供出国留学中介的AI，请在下面的话最符合哪个主题：

"{userinput}"

主题包括： 
1. 出国留学申请的成功案例
2. 美国留学费用
3. 加拿大留学学费
4. 留学中介公司简介和业务范围 
5. 和托福，雅思英语考试相关的话题
6. 姓名，联系方式
7. 微信号，电话
8. 和GPA成绩相关话题
9. 和留学选择专业相关话题
10. 和申请学校相关话题
11. 其他

你只要返回最接近的一个答案。 如果主题都不符合，就回复"11. 其他"
回答的格式：只要列出选项就可以了
 
"""
    if DEBUG:  print(standalone_query_template.format(userinput=input) )

    #messages = [ #{"role": "system", "content": "你是一个提供出国留学中介的AI"},
    #            {"role": "user", "content": standalone_query_template.format(userinput=input)}]

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
    if order > 4:
        return False
    if DEBUG:
        print(f" matched topic order = {str(order)}")
    chat_history.append({"role": "user", "content": question})
    afilename = 'predefined-topics' + '\\' + str(order)
    with open(afilename, "r") as f:
        data = f.read().replace('\n', '')
        print(data + "\n" + createSuggestion())
        dlen = min(100, len(data))
        chat_history.append({"role": "assistant", "content": data[0:dlen]})
    
    return True


print( ai_message_prompt_template )
chat_history.append({"role": "user", "content": "我想咨询出国留学事项"})
chat_history.append(
    {"role": "assistant", "content": ai_message_prompt_template})

counts = 0
useranswers = []
while True:
    print("Your question:")
    question = input().strip()
    if len( question ) < 2:
        continue
    if question.startswith('微信') and len(question) < 14:
        question = "我的" + question
    useranswers.append(question)
    userinfoflags = [False, False, False, False, False]
    if needUserInfo():
        userinfoflags = standaloneQueyUserInfo(question)
   # chat_history.append({"role": "user", "content": question})
    oneFieldOnly = False
    if question.find(',') < 0 and question.find(' ')< 0:
        if userinfoflags[0] and len(question) <=8:
            oneFieldOnly = True
        if userinfoflags[1] and len(question) <=10:
            oneFieldOnly = True
        if userinfoflags[2] and len(question) <= 8:
            oneFieldOnly = True
    topicFound = False
    if oneFieldOnly is False:
        #check direct topic match
        topicFound = standaloneQueyUserIntent(question)
        if topicFound:
            counts += 1
            continue

    #check local info
    r0s = vectorstore.similarity_search_with_score(question, 4)
    #r1 = vectorstore_topics.similarity_search_with_score(question, 1)[0]

    #print(f"vectorstore r0s = {str(r0s)}  ")
    #print(f"vectorstore r0s[0] =   {str(r0s[0])}")

    if DEBUG:
        print(f"vectorstore score = {str(r0s[0][1])} ")
    #print(f"vectorstore_topics score = {str(r1[1])} ")

   
 #   if r0s[0][1] < r1[1]:
  #      fetchLocalTopicInfo(question, r1[0])
  #  else:
    
    if topicFound is False:
        tmessage = chat_history.copy()
        if r0s[0][1] > 0.2 and oneFieldOnly is False:
            mstr = fetchLocalInfo(question, r0s)
            if DEBUG:
                print(f" local info is = {mstr}")
            ss = ai_message_prompt_with_extra_info.format(question=question, context=mstr) 
            tmessage.append({"role": "user", "content": ss})
        else:
            tmessage.append({"role": "user", "content": question})
        
        tries = 2
        while True:
            try:
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=tmessage, temperature=0
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
        r = parseGPT(reply)
       # print( question )
        print(r[0] + " " + createSuggestion())
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": r[0]})
     
    
    counts += 1 
      