import os
from pathlib import Path
from git import Repo
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv  

load_dotenv()  
class CodebaseIndexer:
    def __init__(self, repo_url, persist_dir="./chroma_db"):
        self.repo_url = repo_url
        self.repo_name = repo_url.split("/")[-1].replace(".git", "")
        self.local_path = f"./cloned_repos/{self.repo_name}"
        self.persist_dir = persist_dir
        self.embeddings = DashScopeEmbeddings(model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"), dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
    def clone_or_update_repo(self):
        """克隆或更新仓库"""
        if os.path.exists(self.local_path):
            print(f"仓库已存在，更新中...")
            repo = Repo(self.local_path)
            repo.remotes.origin.pull()
        else:
            print(f"克隆仓库到 {self.local_path}...")
            repo = Repo.clone_from(self.repo_url, self.local_path)
        return repo
    
    def get_code_files(self, extensions=['.py', '.js', '.java', '.ts']):
        """获取指定扩展名的代码文件"""
        code_files = []
        for ext in extensions:
            for file_path in Path(self.local_path).rglob(f"*{ext}"):
                # 跳过虚拟环境、node_modules等目录
                if any(skip in str(file_path) for skip in ['__pycache__', 'node_modules', '.git', 'venv']):
                    continue
                code_files.append(file_path)
        return code_files
    
    def parse_python_file(self, file_path):
        """解析Python文件，返回文档块"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        try:
            tree = ast.parse(content)
            
            # 提取模块级文档字符串
            module_doc = ast.get_docstring(tree)
            if module_doc:
                documents.append(Document(
                    page_content=f"模块文档: {module_doc}\n文件: {file_path}",
                    metadata={"type": "module_doc", "file": str(file_path)}
                ))
            
            # 提取类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(content, node)
                    docstring = ast.get_docstring(node) or "无文档字符串"
                    
                    documents.append(Document(
                        page_content=f"函数: {node.name}\n文档: {docstring}\n代码:\n{func_code}\n文件: {file_path}",
                        metadata={
                            "type": "function",
                            "name": node.name,
                            "file": str(file_path),
                            "line": node.lineno
                        }
                    ))
                    print(f"✅ 解析到函数 {node.name}，文件路径：{file_path}")
                
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(content, node)
                    docstring = ast.get_docstring(node) or "无文档字符串"
                    
                    documents.append(Document(
                        page_content=f"类: {node.name}\n文档: {docstring}\n代码:\n{class_code}\n文件: {file_path}",
                        metadata={
                            "type": "class", 
                            "name": node.name,
                            "file": str(file_path),
                            "line": node.lineno
                        }
                    ))
                    
        except SyntaxError as e:
            print(f"解析文件 {file_path} 时出错: {e}")
        
        return documents
    
    def index_codebase(self):
        """主索引函数"""
        print("开始处理仓库...")
        self.clone_or_update_repo()
        
        all_documents = []
        code_files = self.get_code_files()
        
        print(f"找到 {len(code_files)} 个代码文件")
        
        for file_path in code_files:
            if file_path.suffix == '.py':
                docs = self.parse_python_file(file_path)
                all_documents.extend(docs)
                print(f"已处理: {file_path}，提取了 {len(docs)} 个代码块")
            # 这里可以添加对其他语言的支持
        
        print(f"总共提取了 {len(all_documents)} 个代码块")
        
        # 对文档进行分块（如果需要的话）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        if all_documents:
            split_docs = text_splitter.split_documents(all_documents)
            print(f"分块后得到 {len(split_docs)} 个文档块")
            
            # 创建向量存储
            print("生成向量嵌入并存储...")
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            print(f"索引完成！已保存到 {self.persist_dir}")
            return vectordb
        else:
            print("未找到可解析的代码文件")
            return None


if __name__ == "__main__":
    indexer = CodebaseIndexer(
        repo_url="https://github.com/unlimitbladeworks/monitor-linux.git",  
        persist_dir="./langchain_code_index"
    )
    
    vectordb = indexer.index_codebase()