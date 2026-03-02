# web/app.py
import streamlit as st
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from qa.rag_qa import CodebaseQA
from qa.summarizer import CodeSummarizer
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="代码智能助手",
    page_icon="🤖",
    layout="wide"
)

class CodeAssistantApp:
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """初始化会话状态"""
        if 'qa_system' not in st.session_state:
            try:
                st.session_state.qa_system = CodebaseQA()
                st.session_state.summarizer = CodeSummarizer()
                st.success("✅ 系统初始化成功！")
            except Exception as e:
                st.error(f"❌ 初始化失败: {e}")
                st.session_state.qa_system = None
                st.session_state.summarizer = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("🤖 代码智能助手")
            st.markdown("---")
            
            # 索引目录配置
            st.subheader("📁 索引设置")
            index_dir = st.text_input(
                "索引目录",
                value="./langchain_code_index",
                help="向量数据库存储目录"
            )
            
            # 模型设置
            st.subheader("🧠 模型设置")
            use_rag = st.checkbox("使用RAG检索", value=True, 
                                 help="启用代码检索增强")
            
            st.markdown("---")
            
            # 清空聊天记录
            if st.button("🗑️ 清空聊天记录", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
            st.markdown("---")
            st.caption("""
            **功能说明**：
            1. 智能问答 - 基于代码库提问
            2. 代码摘要 - 上传文件生成摘要
            3. 代码搜索 - 搜索相似代码片段
            """)
            
            return index_dir, use_rag
    
    def render_chat_interface(self, use_rag: bool):
        """渲染聊天界面"""
        st.header("代码智能问答")
        
        # 显示聊天历史
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            
            with st.chat_message("assistant"):
                st.markdown(chat["answer"])
                
                # 显示引用来源
                if chat.get("sources"):
                    with st.expander("📄 参考来源"):
                        for i, source in enumerate(chat["sources"]):
                            st.markdown(f"**{i+1}. {source.get('type', '代码')}**")
                            st.code(source.get("content", "")[:300], language="python")
        
        # 输入框
        if prompt := st.chat_input("输入您关于代码的问题..."):
            # 添加用户消息
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": "思考中...",
                "sources": []
            })
            
            # 显示用户输入
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 获取回答
            with st.chat_message("assistant"):
                with st.spinner("正在分析代码..."):
                    try:
                        if st.session_state.qa_system:
                            result = st.session_state.qa_system.ask(prompt, use_rag=use_rag)
                            
                            # 更新最后一条记录
                            st.session_state.chat_history[-1] = {
                                "question": prompt,
                                "answer": result["answer"],
                                "sources": self._format_sources(result.get("source_documents", []))
                            }
                            
                            st.markdown(result["answer"])
                            
                            # 显示来源
                            if result.get("source_documents"):
                                with st.expander("📄 参考代码片段"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        metadata = doc.metadata
                                        st.markdown(f"**{i+1}. {metadata.get('type', '代码')}: {metadata.get('name', '未知')}**")
                                        st.markdown(f"文件: `{metadata.get('file', '未知')}`")
                                        st.code(doc.page_content[:500], language="python")
                                        st.markdown("---")
                        else:
                            st.error("问答系统未初始化")
                    except Exception as e:
                        st.error(f"查询失败: {e}")
    
    def _format_sources(self, source_docs):
        """格式化来源文档"""
        sources = []
        for doc in source_docs:
            sources.append({
                "type": doc.metadata.get("type", "代码"),
                "name": doc.metadata.get("name", ""),
                "file": doc.metadata.get("file", ""),
                "content": doc.page_content[:300]
            })
        return sources
    
    def render_summarizer_interface(self):
        """渲染代码摘要界面"""
        st.header("📄 代码文件摘要")
        
        uploaded_file = st.file_uploader(
            "上传代码文件",
            type=['py', 'js', 'java', 'cpp', 'c', 'go', 'rs', 'ts'],
            help="支持多种编程语言文件"
        )
        
        if uploaded_file is not None:
            # 读取文件内容
            content = uploaded_file.read().decode("utf-8")
            file_name = uploaded_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 文件信息")
                st.metric("文件名", file_name)
                st.metric("文件大小", f"{len(content):,} 字节")
                st.metric("代码行数", len(content.split('\n')))
                
                # 代码预览
                with st.expander("👀 代码预览"):
                    st.code(content[:1000], language=self._detect_language(file_name))
            
            with col2:
                st.subheader("🤖 AI分析摘要")
                
                if st.button("生成摘要", type="primary"):
                    with st.spinner("正在分析代码..."):
                        try:
                            summary = st.session_state.summarizer.summarize_file(content, file_name)
                            
                            st.markdown("### 📋 分析结果")
                            st.markdown(summary["summary"])
                            
                            # 保存记录
                            st.session_state.uploaded_files.append({
                                "file_name": file_name,
                                "summary": summary["summary"],
                                "timestamp": st.session_state.get("current_time", "")
                            })
                        except Exception as e:
                            st.error(f"分析失败: {e}")
    
    def _detect_language(self, filename: str) -> str:
        """检测编程语言"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.ts': 'typescript'
        }
        
        for ext, lang in extensions.items():
            if filename.endswith(ext):
                return lang
        
        return 'text'
    
    def render_search_interface(self):
        """渲染代码搜索界面"""
        st.header("🔍 代码语义搜索")
        
        search_query = st.text_input("搜索代码片段", placeholder="例如：错误处理、数据库连接、用户认证")
        
        if search_query and st.button("搜索", type="primary"):
            if st.session_state.qa_system:
                with st.spinner("正在搜索..."):
                    try:
                        results = st.session_state.qa_system.search_similar_code(search_query, k=5)
                        
                        st.subheader(f"搜索结果 ({len(results)} 个)")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"结果 {i+1}: {result['metadata'].get('name', '未知')}", expanded=i==0):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.metric("类型", result['metadata'].get('type', '代码'))
                                    st.text(f"文件: {result['metadata'].get('file', '')}")
                                    st.text(f"行号: {result['metadata'].get('line', 'N/A')}")
                                
                                with col2:
                                    st.code(result['content'], language="python")
                    except Exception as e:
                        st.error(f"搜索失败: {e}")
            else:
                st.error("问答系统未初始化")
    
    def render_main(self):
        """渲染主界面"""
        # 侧边栏
        index_dir, use_rag = self.render_sidebar()
        
        # 主界面标签页
        tab1, tab2, tab3 = st.tabs([
            "智能问答", 
            "代码摘要", 
            "代码搜索"
        ])
        
        with tab1:
            self.render_chat_interface(use_rag)
        
        with tab2:
            self.render_summarizer_interface()
        
        with tab3:
            self.render_search_interface()

def main():
    app = CodeAssistantApp()
    app.render_main()

if __name__ == "__main__":
    main()