from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Khởi tạo lại embeddings và vectorstore (đọc từ db đã lưu)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Khởi tạo model Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

def ask_with_context(question, k=3):
    # Xử lý đặc biệt cho các câu hỏi về danh tính
    question_lower = question.strip().lower()
    if question_lower in [
        "bạn là ai", "mày là ai", "ai đang trả lời", "bạn tên gì", "mày tên gì"
    ]:
        return "Tôi là Trợ Lý ảo của Thanh Tú siu đẹp trai víp rồ."
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    print("DEBUG : context:\n", context)
    if not context.strip():
        # Nếu không có context, cho Gemini trả lời tự do
        prompt = f"""Bạn được tạo ra bởi Thanh Tú siu đẹp trai víp rồ, trả lời ngắn gọn, dễ hiểu, không cần chào hỏi đầu câu sau khi đã chào lần đầu tiên, khi chào thì giới thiệu luôn ,chỉ tập trung vào nội dung câu hỏi.
Câu hỏi: {question}
Trả lời:"""
    else:
        # Nếu có context, yêu cầu trả lời dựa trên context
        prompt = f"""Bạn được tạo ra bởi Thanh Tú siu đẹp trai víp rồ, luôn trả lời ngắn gọn, không cần chào hỏi đầu câu sau khi đã chào lần đầu tiên, khi chào thì giới thiệu luôn ,dễ hiểu, chỉ tập trung vào nội dung câu hỏi, lịch sự,
Dựa trên thông tin sau, hãy trả lời câu hỏi:
---
{context}
---
Câu hỏi: {question}
Trả lời:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


