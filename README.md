# 🧠 AI-Powered Document Query Chatbot

This is an AI-powered chatbot that allows users to ask questions based on the content of PDF documents. The chatbot uses **LangChain**, **Groq's LLMs**, **Google Generative AI Embeddings**, and **FAISS** for efficient retrieval and response generation. The app is built using **Streamlit** for an easy-to-use interface.

---

## 🚀 Features

- Chat with your **PDF documents**
- Powered by **Groq LLMs (Gemma, LLaMA3, Mixtral)** for blazing-fast inference
- Embeddings with **Google Generative AI**
- Document storage with **FAISS vector store**
- Clean **Streamlit** UI
- Display of context used for answering
- Response time tracking

---

## 🧩 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Python](https://www.python.org/)

---

## 📁 Folder Structure

```
├── app.py                      # Main Streamlit app
├── data/                       # Folder containing your PDF files
├── .env                        # API keys (Groq, Google)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation


---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add PDFs

Put all the documents you want to use in the `data/` folder.

### 4. Run the App

```bash
streamlit run app.py
```

---

## 💬 How It Works

1. The app loads and embeds PDFs using Google Generative AI embeddings.
2. Documents are chunked using LangChain's text splitter.
3. FAISS is used to store document embeddings for fast retrieval.
4. On user input, the relevant chunks are retrieved and passed to Groq's LLM using a LangChain retrieval chain.
5. The model returns a contextual and accurate response.
6. Optional: Show the source documents used to answer.

---

## 🧪 Example Prompt

> **User:** "How can visually impaired users use the Zendalona VisionAssist device?"

> **Bot:** "The VisionAssist device includes voice-enabled navigation, a Braille interface, and real-time text recognition. It is compatible with screen readers and offers tactile feedback."

---

## 📦 Requirements

```
streamlit
langchain
langchain_community
langchain_groq
langchain_google_genai
faiss-cpu
python-dotenv


---

## Demo

![image](https://github.com/user-attachments/assets/25e11566-cd21-4a67-bd80-a07bf957f4aa)


---


