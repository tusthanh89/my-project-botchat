from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.qa_chain import ask_with_context

app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ trình duyệt

@app.route('/')
def home():
    template_dir = os.path.join(os.path.dirname(__file__), 'template')
    return send_from_directory(template_dir, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'answer': 'Vui lòng nhập câu hỏi!'})
    answer = ask_with_context(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)