import os
import streamlit as st
from groq import Groq
from rag import HybridRAG

st.set_page_config(
    page_title="My Digital Clone",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_rag():
    return HybridRAG()

st.title("🤖 My Digital Clone")
st.warning("This is an AI clone, not the real person.")

st.write(
    "Ask questions and get answers based on my personal knowledge base."
)

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Could not load knowledge base: {e}")
    st.stop()

question = st.text_area(
    "Ask me a question",
    placeholder="What are my current learning goals?"
)

mode = st.radio(
    "Select Retrieval Mode",
    ["BM25", "Dense", "Hybrid"],
    horizontal=True,
    index=2
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base..."):
            results = rag.retrieve(
                question,
                mode=mode,
                top_k=3
            )

        context = ""
        for item in results:
            context += (
                f"\nSource: {item['source']}\n"
                f"Content: {item['text']}\n"
            )

        api_key = st.secrets.get(
            "GROQ_API_KEY",
            os.getenv("GROQ_API_KEY")
        )

        if not api_key:
            st.error("GROQ_API_KEY is not configured.")
        else:
            try:
                client = Groq(api_key=api_key)

                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI digital clone of the "
                                "person described in the context. "
                                "Answer in first person. Use only "
                                "the supplied context. Do not invent "
                                "facts. Admit when information is missing."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context}\n\n"
                                f"Question: {question}"
                            )
                        }
                    ],
                    max_tokens=400,
                    temperature=0.3
                )

                st.subheader("AI Clone Answer")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Answer generation failed: {e}")

        st.subheader("Retrieved Chunks, Sources and Scores")

        for item in results:
            with st.expander(
                f"{item['source']} | Chunk {item['chunk_id']} "
                f"| Score: {item['score']:.4f}"
            ):
                st.write(item["text"])
