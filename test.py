from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 加载项目代码
loader = DirectoryLoader("./cloned_repos", glob="**/*.py")
documents = loader.load()

# 分块并存入向量库
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
vectordb = Chroma.from_documents(docs, OpenAIEmbeddings())

# 查询整体功能
query = "这个项目的整体目的和功能是什么？"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)  # LLM 会基于所有代码的上下文回答