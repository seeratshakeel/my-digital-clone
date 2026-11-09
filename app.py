import os
import gradio as gr
from groq import Groq
from rag import HybridRAG

print("Loading Digital Clone...")
rag = HybridRAG()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "openai/gpt-oss-20b"


def answer_question(question, mode):
    if not question.strip():
        return "Please enter a question.", "No retrieved documents."

    results = rag.retrieve(
        question,
        mode=mode,
        top_k=3
    )

    context = ""
    retrieved_text = ""

    for item in results:
        context += (
            f"\nSource: {item['source']}\n"
            f"Content: {item['text']}\n"
        )

        retrieved_text += (
            f"Source: {item['source']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Score: {item['score']:.4f}\n"
            f"Text: {item['text']}\n"
            f"{'-' * 50}\n"
        )

    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not configured.", retrieved_text

    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI digital clone of the person "
                        "described in the supplied context. "
                        "Answer in first person as that person. "
                        "Use only the supplied context. "
                        "Do not invent personal facts. "
                        "If information is missing, honestly say so."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Knowledge context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
            max_tokens=400,
            temperature=0.3
        )

        answer = response.choices[0].message.content

    except Exception as error:
        answer = (
            "AI generation could not be completed. "
            f"Technical detail: {str(error)}"
        )

    return answer, retrieved_text


with gr.Blocks(title="My Digital Clone") as demo:

    gr.Markdown("# My Digital Clone")

    gr.Markdown(
        "**This is an AI clone, not the real person.**"
    )

    gr.Markdown(
        "Ask questions and get answers based on my personal knowledge base."
    )

    question = gr.Textbox(
        label="Ask me a question",
        placeholder="What are my current learning goals?",
        lines=2
    )

    mode = gr.Radio(
        choices=["BM25", "Dense", "Hybrid"],
        value="Hybrid",
        label="Select Retrieval Mode"
    )

    submit = gr.Button("Get Answer")

    answer = gr.Textbox(
        label="AI Clone Answer",
        lines=5
    )

    retrieved = gr.Textbox(
        label="Retrieved Chunks, Sources and Scores",
        lines=15
    )

    submit.click(
        fn=answer_question,
        inputs=[question, mode],
        outputs=[answer, retrieved]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)