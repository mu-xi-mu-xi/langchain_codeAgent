# qa/rag_qa.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CodebaseQA:
    """代码库智能问答系统"""
    
    def __init__(self, persist_dir="./langchain_code_index"):
        self.persist_dir = persist_dir
        
        # 初始化模型
        self._init_models()
        
        # 加载向量数据库
        self._load_vectorstore()
    
    def _init_models(self):
        """初始化模型"""
        # 1. 嵌入模型
        from langchain_community.embeddings import DashScopeEmbeddings
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 2. 大语言模型
        from langchain_community.chat_models import ChatTongyi
        self.llm = ChatTongyi(
            model=os.getenv("DASHSCOPE_LLM_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
            max_tokens=2000
        )
    
    def _load_vectorstore(self):
        """加载向量数据库"""
        from langchain_chroma import Chroma
        
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        # 创建检索器
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 5}  # 返回5个最相关文档
        )
    
    def ask(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """回答问题"""
        if not use_rag or self.vectordb is None:
            # 不使用RAG，直接问大模型
            return self._direct_answer(question)
        
        # 使用RAG
        return self._rag_answer(question)
    
    def _rag_answer(self, question: str) -> Dict[str, Any]:
        """使用RAG回答问题 (采用直接向量搜索模式，与test.py保持一致)"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        # === 关键修改 1：使用 similarity_search 替代 retriever.invoke ===
        # 这是与您测试成功的 test.py 脚本完全一致的方法
        try:
            source_docs = self.vectordb.similarity_search(question, k=5)
        except Exception as e:
            # 添加错误处理，便于调试
            print(f"[ERROR] 向量搜索失败: {e}")
            # 如果搜索失败，回退到直接回答
            return self._direct_answer(question)
        
        print(f"[DEBUG] 向量搜索成功，找到 {len(source_docs)} 个相关文档。")
        # === 关键修改结束 ===
        
        # 检查是否检索到相关文档
        if not source_docs:
            return {
                "question": question,
                "answer": "未能在代码库中找到与问题相关的代码片段。",
                "source_documents": [],
                "method": "RAG"
            }
        
        def format_docs(docs):
            """格式化检索到的文档"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 构建提示词模板
        template = """
        你是一个专业的代码助手，请严格基于以下代码片段回答问题。

        相关代码片段：
        {context}

        用户问题：{question}

        请根据提供的代码信息，给出准确、详细的回答。回答应聚焦于代码的功能、逻辑或使用方式。
        如果提供的代码片段不足以回答问题，请明确指出这一点，不要编造信息。
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 使用 LCEL 构建问答链
        chain = (
            {
                "context": lambda x: format_docs(source_docs),  # 使用上一步直接搜索到的文档
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 调用链生成答案
        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"[ERROR] 生成回答时发生错误: {e}")
            # 如果生成失败，提供回退答案
            answer = "无法基于现有代码生成回答，可能是模型调用出错。"
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_docs,
            "method": "RAG"
        }
    
    def _direct_answer(self, question: str) -> Dict[str, Any]:
        """直接问答（不使用RAG）"""
        response = self.llm.invoke(question)
        
        return {
            "question": question,
            "answer": response.content,
            "source_documents": [],
            "method": "Direct"
        }
    
    def search_similar_code(self, query: str, k: int = 3) -> List[Dict]:
        """搜索相似代码（不经过LLM处理）"""
        docs = self.vectordb.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "rank": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "score": None  # Chroma不直接返回分数
            })
        
        return results