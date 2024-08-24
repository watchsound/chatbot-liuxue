import os, dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.memory import ChatMessageHistory

dotenv.load_dotenv()

llm0 = OpenAI(
    temperature=0,
    openai_api_key="OPENAI_API_KEY",
    model_name="text-davinci-003"
)

llm=ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0, model_name="gpt-3.5-turbo",  
),  
#memory=ConversationSummaryMemory(llm=llm), 
memory = ConversationBufferMemory(return_messages=True)
history = ChatMessageHistory()

chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo",
    ),
    memory = memory,
    chain_type="stuff", 
    retriever=FAISS.load_local("predefined-materials-embed", OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

system_intro_template="""您是出国留学中介的工作人员，提供出国留学咨询。  
在聊天中，在每次回答结束前尽可能引导对话提供如下五类信息： 
1. 姓名。
2. 电话号码，微信号，或者其他个人联系方式。 
3. 托福,雅思,或者其他GPA成绩。
4. 想申请哪些学校、专业。
5. 对留学有什么具体的要求或期望。

在对方的提问中也要尽可能找到上述五类信息。


您返回的结果分两部分
第一部分是您已经获得的学生信息，用如下格式表示：
姓名=
联系方式=
成绩=
申请学校和专业=
具体要求=

第二部分才是您对问题的回答，用简单文本格式

请用中文回答。
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_intro_template)

ai_message_prompt_template = """
您好！我是出国留学中介的工作人员，非常愿意为您提供有关留学的咨询。在我们的交谈中，请注意避免讨论与出国留学无关的政治话题。

现在，我想了解一些您的具体要求和期望，以便能更好地为您提供帮助。请回答以下问题：

您对留学有什么具体的要求或期望？
您打算申请哪些学校和专业？
您的托福雅思成绩是多少？
您能告诉我您的姓名和微信号，以便我们能够更方便地联系吗？
请提供以上信息，我将尽力回答您的问题和提供相关建议
"""

ai_message_prompt = AIMessagePromptTemplate.from_template(ai_message_prompt_template)

history.add_ai_message(ai_message_prompt_template)

human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
 
#chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, ai_message_prompt, human_message_prompt])
chat_prompt = ChatPromptTemplate.from_role_strings(
    [('system', system_intro_template), ('assistant', ai_message_prompt_template), ('user', human_template)])
chat_prompt2 = ChatPromptTemplate.from_messages([ human_message_prompt])

userinfometa = ['姓名', '联系方式', '成绩', '申请学校和专业', '具体要求']
userinfo = ['','','','','']

def createUserInfo():
    result = ''
    for i in range(5):
        if len(userinfo[i].strip()) > 0:
            result += userinfometa[i] + '是' + userinfo[i] + ', '
    return result


def createUserInfo():
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


def parseGPT(input):

    fn = userinfometa[0]
    pos = input.find(fn + '=')
    if pos < 0:
        return input

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

    fn = userinfometa[4]
    pos2 = userdata.find(fn + '=')
    if pos2 > 0 and userdata.endswith(fn+"=") is False:
        f = userdata[pos2+len(fn)+1:].strip()
        if f.startswith('未提供') is False:
            userinfo[4] += f + ' '

    return input[0: pos]

print( ai_message_prompt_template )
counts = 0
useranswers = []
while True:
    print("Your question:")
    question = input()
    if len( question ) < 2:
        continue
    useranswers.append(question)
    history.add_user_message(question)
    if counts == 0:
        response = chatbot.run(chat_prompt.format(text=question) ) 
    else:
        response = chatbot.run(chat_prompt2.format(text=question))
    counts += 1
    response = parseGPT( response )
    print( response ) 
    history.add_ai_message(response)
    uinfo = createUserInfo()  
    if len( uinfo ) > 0:
        history.chat_memory.add_user_message(uinfo)

    """   memory.clear()
    for m in history:
        if isinstance(m, HumanMessage):
            memory.chat_memory.add_user_message(m.content) 
        else:
            memory.chat_memory.add_ai_message(m.content) """

  
    #if len( uinfo ) > 0:
    #    memory.save_context({"input": uinfo}, {"output": "same"})
    print()
    print(history.messages )
    print()
    print( memory.load_memory_variables({}) )

# get a chat completion from the formatted messages 
#print(chatbot.run(
 #   chat_prompt.format(text="我的电话是1234567， 我想申请计算机专业") 
#)) 


"""
prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)  
print(chatbot.run(
    prompt.format(query="星加坡留学费用是多少")
)) 
"""