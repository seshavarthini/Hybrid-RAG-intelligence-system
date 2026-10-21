
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class AnswerGenerator:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Check your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, question: str, results: list[dict]) -> str:
        
        question_lower = question.lower().strip()

        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening"
        ]

        if question_lower in greetings:
            return "Hello! 👋 I'm your Hybrid RAG PDF Assistant. Ask me anything about your PDF!"
        if not results:
            return "I could not find relevant information in the PDF."

        context = "\n\n".join(
            f"[Source: {r['filename']}, Page: {r['page_number']}]\n"
            f"{r['text']}"
            for r in results
        )

        prompt = f"""
Answer the question using only the supplied PDF context.
Do not invent facts. If the answer is unavailable, say so.
Cite the source filename and page number in your answer.

PDF context:
{context}

Question: {question}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful PDF question-answering assistant. "
                        "Use only the provided context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content or ""
