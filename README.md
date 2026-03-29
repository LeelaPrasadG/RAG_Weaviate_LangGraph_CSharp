# RAG Weaviate Semantic Kernel C#

This is a RAG application using Microsoft Semantic Kernel for orchestration, in-memory vector storage with cosine similarity (as a substitute for Weaviate due to package availability issues), and OpenAI as LLM.

## Prerequisites

- .NET 8.0 SDK

- OpenAI API Key

## Setup

1. Install .NET 8.0 from https://dotnet.microsoft.com/download

2. Clone or download the project.

3. Set your OpenAI API key in .env file: OPENAI_API_KEY=your_key

4. Restore packages: dotnet restore

6. Build the project: dotnet build

7. Run the application: dotnet run

8. Open http://localhost:5000 in browser.

## How it works

- Documents in RAG Docs are read, chunked, embedded, and stored in memory.

- User asks question, top 3 chunks retrieved using cosine similarity.

- LLM generates answer based on the chunks.

## To verify Negative case

- Comment out Line 42-45 and uncomment line 46 in RagService.cs and ask the Below Question:

who are the authors of the paper "Attention is all you need"

- Need to get equivalent response as Not in the Context