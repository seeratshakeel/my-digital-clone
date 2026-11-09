⸻



title: My Digital Clone

emoji: 🤖

colorFrom: blue

colorTo: purple

sdk: gradio

app\_file: app.py

pinned: false



My Digital Clone — Hybrid RAG Chatbot



This is an AI clone, not the real person.



Project Overview



This project builds an AI digital clone using a Hybrid Retrieval-Augmented Generation (RAG) system. It answers questions about my education, skills, projects, personality, opinions, and learning journey using my personal knowledge base.



Technologies Used



\* Python

\* Gradio

\* rank\_bm25

\* Sentence Transformers (all-MiniLM-L6-v2)

\* NumPy

\* Groq API (openai/gpt-oss-20b)



Knowledge Base



The knowledge base contains six Markdown documents:



1\. bio.md — Personal introduction and education.

2\. cv.md — Education, skills, and achievements.

3\. faq.md — Frequently asked questions and answers.

4\. learning.md — Learning journey and technical knowledge.

5\. opinions.md — Personality, values, and opinions.

6\. projects.md — Project information and details.



These documents provide personal and factual information for first-person chatbot responses.



RAG Implementation



\* Documents are divided into approximately 250-word chunks with 40-word overlap.

\* Each chunk keeps its source filename and chunk ID.

\* BM25 retrieval uses rank\_bm25.

\* Dense retrieval uses Sentence Transformers with all-MiniLM-L6-v2.

\* Embeddings are calculated once and saved in embeddings.npy.

\* Cosine similarity is calculated using NumPy.

\* Hybrid retrieval combines normalized BM25 and dense scores using a weighted score with alpha = 0.5.

\* The system retrieves the top three chunks for answer generation.



The alternative fusion method is Reciprocal Rank Fusion (RRF).



LLM Generation



Groq API with openai/gpt-oss-20b generates answers using retrieved context. The chatbot answers in first person, uses only the supplied context, and admits when information is unavailable.



The API key is configured as an environment secret named GROQ\_API\_KEY.



Testing and Retrieval Comparison



Five questions were tested using BM25, Dense, and Hybrid retrieval modes.



Question	BM25 top result	Dense top result	Hybrid top result

What projects have I worked on?	cv.md — relevant	bio.md — less relevant	cv.md — relevant

What are my current learning goals?	learning.md — relevant	opinions.md — less relevant	learning.md — relevant

What are my career goals?	cv.md — partially relevant	bio.md — less relevant	cv.md — partially relevant

What are my strengths and how do I solve problems?	faq.md — partially relevant	opinions.md — relevant	faq.md — partially relevant

What motivates me and what are my personal values?	opinions.md — relevant	opinions.md — relevant	opinions.md — relevant



Best overall mode: Hybrid retrieval performed consistently well across the five tests. Dense retrieval performed better for the strengths question and retrieved motivation-related information effectively. BM25 also performed well when the question matched specific document keywords.



Failure Case



For the question “What are my current learning goals?”, Dense retrieval ranked opinions.md first instead of the relevant learning.md document. This shows that semantic similarity does not always identify the most useful information.



Future Improvement



Improve document organization and use heading-aware chunking so that related information stays together. This can improve retrieval relevance and reduce unrelated results.



How to Run



Install dependencies:



pip install -r requirements.txt



Run the application:



python app.py



Public Space



https://huggingface.co/spaces/seeratsh/my-digital-clone
## Live Deployed Application

**Public App URL:** https://seeratshakeel-my-digital-clone-streamlit-app-jpruru.streamlit.app/

The application is deployed using Streamlit Community Cloud.
