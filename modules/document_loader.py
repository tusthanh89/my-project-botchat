import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt
import pandas as pd
from langchain_core.documents import Document
from doc_converter import convert_doc_to_docx

# Import trực tiếp các loader cần thiết
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

def load_documents_from_folder(folder_path):
    """
    Load tài liệu từ thư mục với xử lý lỗi
    """
    documents = []
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.xlsx', '.csv']
    
    print(f"Đang quét thư mục: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Thư mục {folder_path} không tồn tại!")
        return documents
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in supported_extensions:
            print(f"Bỏ qua file không được hỗ trợ: {filename}")
            continue
            
        try:
            print(f"Đang đọc file: {filename}")
            
            if file_ext == '.txt':
                # Thử các encoding khác nhau
                encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
                for encoding in encodings:
                    try:
                        loader = TextLoader(file_path, encoding=encoding)
                        docs = loader.load()
                        print(f"Đã load thành công file {filename} với encoding {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                        
            elif file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
            elif file_ext == '.docx':
                # Sử dụng docx2txt cho file .docx
                text = docx2txt.process(file_path)
                docs = [Document(page_content=text)]
                
            elif file_ext == '.doc':
                # Chuyển đổi .doc sang .docx trước khi đọc
                docx_path = convert_doc_to_docx(file_path)
                if docx_path:
                    try:
                        text = docx2txt.process(docx_path)
                        docs = [Document(page_content=text)]
                        # Xóa file tạm
                        os.unlink(docx_path)
                    except Exception as e:
                        print(f"Lỗi khi đọc file .docx đã chuyển đổi: {str(e)}")
                        continue
                else:
                    print(f"Không thể chuyển đổi file .doc: {filename}")
                    continue
                
            elif file_ext == '.xlsx':
                # Đọc Excel file và chuyển đổi thành text có cấu trúc
                df = pd.read_excel(file_path)
                # Chuyển DataFrame thành text có cấu trúc
                text = "Dữ liệu từ file Excel:\n"
                for col in df.columns:
                    text += f"\n{col}:\n"
                    text += df[col].to_string() + "\n"
                docs = [Document(page_content=text)]
                
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
                docs = loader.load()
                
            documents.extend(docs)
            print(f"Đã load thành công file: {filename}")
            
        except Exception as e:
            print(f"Lỗi khi load file {filename}: {str(e)}")
            continue
            
    return documents

if __name__ == "__main__":
    # Test load documents
    docs = load_documents_from_folder("data")
    print(f"\nTổng số tài liệu đã load: {len(docs)}")
    
    if docs:
        # Chia nhỏ tài liệu
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        
        print(f"Số đoạn sau khi chia nhỏ: {len(chunks)}")
        if chunks:
            print("\nNội dung đoạn đầu tiên:")
            print("-" * 50)
            print(chunks[0].page_content[:500] + "..." if len(chunks[0].page_content) > 500 else chunks[0].page_content)
            print("-" * 50)

