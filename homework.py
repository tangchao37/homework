# agent_core.py
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.tools import tool
from agent_core import create_agent, run_agent
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

langfuse_handler = CallbackHandler()

def create_agent(tools):
    """创建并配置Agent"""
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY_LLAMA33"),
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

def run_agent(agent):
    """运行Agent交互循环"""
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        result = agent.invoke(
            {"input": query}, 
            config={"callbacks": [langfuse_handler]}
        )
        print(f"Result: {result['output']}")


@tool
def doc_tool(query: str) -> str:
    """使用知识库回答关于Brainrent的问题（中文）"""
    # 加载文本文档
    docs = TextLoader("brainrent_part1.txt", encoding='utf-8').load()
    
    # 分割文本
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # 创建问答链
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
    return qa_chain.run(input_documents=chunks, question=query)

@tool
def pdf_tool(query: str) -> str:
    """处理PDF文档中的问题"""
    loader = PyPDFLoader("documentation.pdf")
    pages = loader.load_and_split()
    qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce")
    return qa_chain.run(input_documents=pages, question=query)

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建工具集
tools = [
    Tool(name="KnowledgeBase", func=doc_tool, 
         description="回答关于Brainrent产品的问题"),
    Tool(name="PDFProcessor", func=pdf_tool, 
         description="处理PDF文档中的查询"),
    Tool(name="Calculator", func=calculator, 
         description="执行数学计算")
]

# 主程序入口
if __name__ == "__main__":
    agent = create_agent(tools)
    run_agent(agent)
