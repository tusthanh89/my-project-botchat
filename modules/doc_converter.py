import win32com.client
import pythoncom
import tempfile
import os

def convert_doc_to_docx(doc_path):
    """
    Chuyển đổi file .doc sang .docx sử dụng Word
    """
    try:
        # Khởi tạo COM object
        pythoncom.CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        
        # Tạo file tạm
        temp_docx = tempfile.NamedTemporaryFile(suffix='.docx', delete=False)
        temp_docx.close()
        
        # Mở file .doc
        doc = word.Documents.Open(doc_path)
        # Lưu dưới dạng .docx
        doc.SaveAs2(temp_docx.name, FileFormat=16)  # 16 là định dạng .docx
        doc.Close()
        word.Quit()
        
        return temp_docx.name
    except Exception as e:
        print(f"Lỗi khi chuyển đổi file .doc: {str(e)}")
        return None
    finally:
        pythoncom.CoUninitialize() 