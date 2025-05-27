from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from typing import List

class QABot:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm = OpenAI(temperature=0)
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """
        Tạo QA chain
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def answer_question(self, question: str) -> str:
        """
        Trả lời câu hỏi
        """
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}" 