# qa/summarizer.py
import os
import ast
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class CodeSummarizer:
    """代码文件摘要生成器"""
    
    def __init__(self):
        self._init_model()
    
    def _init_model(self):
        """初始化大语言模型"""
        from langchain_community.chat_models import ChatTongyi
        
        self.llm = ChatTongyi(
            model=os.getenv("DASHSCOPE_MODEL", "qwen-turbo"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
            max_tokens=1000
        )
    
    def summarize_file(self, file_content: str, file_name: str = None) -> Dict[str, Any]:
        """生成代码文件摘要"""
        
        # 1. 先分析代码结构
        structure_info = self._analyze_code_structure(file_content, file_name)
        
        # 2. 构造Prompt
        prompt = self._create_summary_prompt(file_content, file_name, structure_info)
        
        # 3. 调用LLM生成摘要
        response = self.llm.invoke(prompt)
        
        return {
            "file_name": file_name or "unknown",
            "structure_info": structure_info,
            "summary": response.content,
            "file_size": len(file_content)
        }
    
    def _analyze_code_structure(self, content: str, file_name: str = None) -> Dict:
        """分析代码结构"""
        structure = {
            "file_name": file_name,
            "language": "unknown",
            "functions": [],
            "classes": [],
            "imports": [],
            "lines": len(content.split('\n'))
        }
        
        # 检测文件类型
        if file_name:
            if file_name.endswith('.py'):
                structure["language"] = "python"
                structure.update(self._analyze_python_code(content))
            elif file_name.endswith('.js'):
                structure["language"] = "javascript"
            elif file_name.endswith('.java'):
                structure["language"] = "java"
        
        return structure
    
    def _analyze_python_code(self, content: str) -> Dict:
        """分析Python代码结构"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "line": node.lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports[:10]  # 只取前10个导入
            }
        except:
            return {"functions": [], "classes": [], "imports": []}
    
    def _create_summary_prompt(self, content: str, file_name: str, structure_info: Dict) -> str:
        """创建摘要生成的Prompt"""
        # 截取前2000字符，避免超过token限制
        content_preview = content[:2000] + "..." if len(content) > 2000 else content
        
        # 结构信息文本化
        structure_text = f"""
文件信息：
- 文件名：{file_name or '未命名'}
- 编程语言：{structure_info.get('language', '未知')}
- 总行数：{structure_info.get('lines', 0)}
"""
        
        if structure_info.get('language') == 'python':
            funcs = structure_info.get('functions', [])
            classes = structure_info.get('classes', [])
            imports = structure_info.get('imports', [])
            
            if funcs:
                structure_text += f"\n函数列表 ({len(funcs)}个)：\n"
                for func in funcs[:5]:  # 只显示前5个
                    structure_text += f"  - {func['name']} (行{func['line']})\n"
            
            if classes:
                structure_text += f"\n类列表 ({len(classes)}个)：\n"
                for cls in classes[:5]:  # 只显示前5个
                    structure_text += f"  - {cls['name']} (行{cls['line']})\n"
            
            if imports:
                structure_text += f"\n主要导入 ({len(imports)}个)：\n"
                for imp in imports[:5]:
                    structure_text += f"  - {imp}\n"
        
        prompt = f"""
你是一个专业的代码分析助手。请分析以下代码文件，并生成一份清晰、准确的代码摘要。

{structure_text}

代码内容（预览）：
{content_preview}
请生成以下格式的代码摘要：

1. **文件概述**：简要说明这个文件的主要功能和用途
2. **核心功能**：列出文件实现的主要功能
3. **代码结构**：描述代码的组织方式
4. **关键函数/类**：如果有的话，说明最重要的函数或类
5. **技术特点**：使用的编程范式、设计模式或关键技术
6. **潜在用途**：这个代码可能的应用场景

请用中文回答，保持专业但易懂。
"""
        return prompt