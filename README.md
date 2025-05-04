# Chat with Your Video Library - RAG System

![Project Banner](https://via.placeholder.com/800x200?text=Video+Library+RAG+System)

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to query educational video content and receive answers grounded in specific video segments. The system combines vector similarity search with large language models to provide accurate, timestamped responses with relevant video clips.

## Key Features
- **Video Knowledge Base**: Indexed collection of educational videos
- **Semantic Search**: Finds relevant video segments using vector embeddings
- **Clip Generation**: Automatically extracts relevant video portions
- **Gradio Interface**: User-friendly web interface for queries
- **Streaming Responses**: Real-time generation of answers

## Technical Components
1. **Data Pipeline**
   - Video processing with FFmpeg
   - Subtitle alignment and text extraction
   - Vector embedding generation

2. **Retrieval System**
   - Qdrant vector database for semantic search
   - Sentence Transformers for embeddings
   - Hybrid search (text + vector)

3. **Generation System**
   - Ollama with Llama3 for response generation
   - Context-aware prompting
   - Timestamp references in answers

4. **Deployment**
   - Gradio web interface
   - Clip generation and display
   - Example queries and demonstrations

## File Structure
