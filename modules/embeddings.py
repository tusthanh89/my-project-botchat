from document_loader import load_documents_from_folder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY trong file .env")

try:
    # Đọc và chia nhỏ tài liệu
    print("Đang đọc tài liệu từ thư mục data...")
    docs = load_documents_from_folder("data")
    
    if not docs:
        raise ValueError("Không tìm thấy tài liệu nào trong thư mục data/")
    
    print(f"Đã load {len(docs)} tài liệu.")
    
    # Chia nhỏ tài liệu
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Đã chia thành {len(chunks)} đoạn nhỏ.")

    # Sinh embeddings và lưu vào ChromaDB
    print("Đang sinh embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # Xóa thư mục db cũ nếu tồn tại
    if os.path.exists("db"):
        import shutil
        shutil.rmtree("db")
        print("Đã xóa thư mục db cũ")

    print("Đang lưu embeddings vào ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Đã lưu embeddings vào vector database thành công!")
    print(f"Tổng số vector đã lưu: {len(chunks)}")

except Exception as e:
    print(f"Đã xảy ra lỗi: {str(e)}")
