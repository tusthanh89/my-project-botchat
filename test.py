from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleBot:
    def __init__(self):
        # Khởi tạo model để tạo embeddings
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Danh sách câu trả lời mẫu
        self.answers = [
            "Để nấu phở bò, bạn cần: nước dùng, bánh phở, thịt bò, gia vị",
            "Cách làm bún chả: thịt nướng, nước mắm, bún, rau sống",
            "Cách nấu canh chua: cá, dứa, cà chua, me, rau răm",
            "Cách làm cơm rang: cơm nguội, trứng, hành, nước mắm",
            "Cách nấu lẩu: nước dùng, thịt, hải sản, rau, nước chấm"
        ]
        
        # Tạo embeddings cho tất cả câu trả lời
        self.answer_embeddings = [self.model.encode(ans) for ans in self.answers]

    def find_answer(self, question):
        # Tạo embedding cho câu hỏi
        question_embedding = self.model.encode(question)
        
        # Tính độ tương đồng với tất cả câu trả lời
        similarities = []
        for ans_emb in self.answer_embeddings:
            similarity = np.dot(question_embedding, ans_emb) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(ans_emb)
            )
            similarities.append(similarity)
        
        # Lấy câu trả lời có độ tương đồng cao nhất
        best_match_index = np.argmax(similarities)
        return self.answers[best_match_index]

# Sử dụng bot
bot = SimpleBot()

# Test các câu hỏi
questions = [
    "Làm sao để nấu phở?",
    "Cách nấu canh chua như thế nào?",
    "Hướng dẫn làm bún chả",
    "Làm thế nào để rang cơm?",
    "Cách nấu lẩu tại nhà"
]

for question in questions:
    answer = bot.find_answer(question)
    print(f"\nCâu hỏi: {question}")
    print(f"Câu trả lời: {answer}")
