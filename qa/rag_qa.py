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
            model=os.getenv("DASHSCOPE_MODEL"),
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
            search_type="similarity",
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
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        template = """
        你是一个专业的代码助手，请基于以下代码片段回答问题。

        相关代码片段：
        {context}

        问题：{input}

        请根据提供的代码信息，给出准确、详细的回答。
        如果代码片段不相关或不足，请说明你无法基于现有代码回答。
        """

        prompt = ChatPromptTemplate.from_template(template)

        # 1. 检索相关文档
        source_docs = self.retriever.invoke(question)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 2. 使用 LCEL 构建问答链（仅依赖 langchain-core）
        chain = (
            {"context": lambda _: format_docs(source_docs), "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

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