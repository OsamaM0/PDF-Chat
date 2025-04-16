# Specky (Back-End System)

🎉 **Specky** is the cutting-edge back-end system designed to empower your applications with AI-driven capabilities for handling files, natural language, and voice interactions. After 200+ hours of development, this is a revolutionary system ready to transform how applications work with data.

---

## Core Features

### 1. **File Chat**  
💬 **Engage with Files Seamlessly**  
Specky enables users to chat with files, regardless of their type. Whether it’s a PDF, Word document, or spreadsheet, you can:  
- Ask questions about file content.  
- Extract key information in real-time.  
- Search across multiple files intelligently.  

**Use Case**: Quickly find insights from a collection of research papers or contracts.

---

### 2. **File Translation**  
🌍 **Translate Files to Any Language**  
Specky provides robust translation tools for converting file content into any desired language. This feature supports diverse document formats and ensures high-quality, context-aware translations.  

**Use Case**: Make your business documents accessible globally by translating them into multiple languages.

---

### 3. **File Summarization**  
📝 **Summarize Content Effortlessly**  
Generate concise summaries for any file, tailored to your preferred length or page limits. This feature is perfect for extracting the essence of lengthy documents.  

**Use Case**: Save time by summarizing a 100-page report into key points or a 500-word summary.

---

### 4. **Pronunciation Correction**  
🗣️ **Perfect Your Pronunciation**  
Specky’s AI-driven pronunciation correction ensures accurate verbal communication. It analyzes your input and provides precise feedback for improving spoken language skills.  

**Use Case**: Ideal for learners refining their pronunciation or professionals preparing for presentations.

---

### 5. **Text-to-Speech**  
🔊 **Turn Text into Lifelike Speech**  
Transform written text into natural, high-quality audio output. This feature enhances accessibility and usability, especially for visually impaired users or those on the go.  

**Use Case**: Generate audio guides or narrations for eBooks effortlessly.

---

### 6. **Speech-to-Text**  
🎙️ **Convert Speech into Accurate Text**  
Record and transcribe spoken words into accurate text. This feature is indispensable for creating meeting notes, captions, or analyzing conversations.  

**Use Case**: Automate transcription for meetings, lectures, or interviews.

---

## Why Specky?

- **AI-Powered**: Leveraging the power of Retrieval-Augmented Generation (RAG) for high efficiency.  
- **Multi-File Capability**: Handle diverse file types and formats.  
- **Scalable Solutions**: Perfect for individual users or large enterprises.  
- **Integrated Workflows**: Seamlessly connect NLP, voice, and data processing into one platform.

---

## Get Started

1. **Create an Account**: Sign up on [Specky](https://www.specky.com).  
2. **Setup API Access**: Secure your API credentials.  
3. **Start Using Features**: Implement the workflows into your application.

---


## Requirements

- Python 3.8 or later

#### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mini-rag python=3.8
```
3) Activate the environment:
```bash
$ conda activate mini-rag
```

### (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```


## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.

## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials



```bash
$ cd docker
$ sudo docker compose up -d
```

## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000



