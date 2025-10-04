# NOTE: DOCKER DDESNT WORK AS INTENDED JUST PIP INSTALL REQUIREMENTS FILE FOR BEST EXPERIENCE
# If you want docker to work, press on another input after loading/searching article
📰 Political Article → Connection Graph
An AI-powered web application that automatically extracts entities and relationships from political articles to create interactive connection graphs. Built with Streamlit and Google Gemini AI.

🌟 Features
📝 Multiple Input Methods: Paste text, upload files (PDF, DOC, TXT), upload audio, or search for articles

🤖 AI-Powered Extraction: Uses Google Gemini to automatically identify entities and relationships

🌐 Interactive Graphs: Visualize connections with color-coded nodes and detailed tooltips

💬 Chat with Your Data: Ask questions about the extracted connections and relationships

🔍 Cross-Article Analysis: Find connections across multiple articles

💾 Export Options: Download graphs as JSON or GraphML

🚀 Quick Start
Prerequisites
Python 3.8+

API keys for:

Google Gemini AI

Serper (for article search) - Optional

AssemblyAI (for audio transcription) - Optional

Installation
Clone the repository


1. git clone <repository-url>
cd political-article-graph
Install dependencies


2. pip install -r requirements.txt
Set up environment variables


3. Create a .env file in the root directory:

GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here  # Optional
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here  # Optional
Run the application

4. streamlit run smart_article_graph.py


Using Docker

1. Build and run with Docker Compose


2. docker-compose up --build
Access the application
Open your browser to http://localhost:8501

🔑 API Keys Setup

Required:
Google Gemini AI: Get from Google AI Studio

Optional:
Serper API: For article search functionality - Serper.dev

AssemblyAI: For audio transcription - AssemblyAI


📖 How to Use
1. Add Article Content
Choose from four input methods:

🔍 Search Article: Enter a search query to fetch the top article automatically

📝 Paste Text: Copy and paste article text directly

📄 Upload File: Upload PDF, DOC, DOCX, or TXT files (max 3MB)

🎤 Upload Audio: Upload MP3, WAV, M4A files for automatic transcription

2. Extract Graph
Click "Extract Graph from All" to process the articles. The AI will:

Identify entities (people, organizations, politicians, companies)

Extract relationships (lobbying, funding, employment, etc.)

Find cross-article connections

Build an interactive network graph

3. Explore Results
Interactive Graph: Hover over nodes for details, drag to explore connections

Connection List: View all relationships as text

Raw Data: Inspect the extracted JSON data

Export: Download as JSON or GraphML

4. Chat with Your Data
Use the built-in chatbot to ask questions like:

"Show all politicians and their connections"

"What financial relationships are mentioned?"

"Who received the most funding?"

🎯 Example Use Cases
Journalists: Map political influence networks

Researchers: Analyze lobbying relationships

Activists: Track corporate-political connections

Students: Study political science and policy networks


🔧 Configuration

Environment Variables
GEMINI_API_KEY: Required for AI processing

SERPER_API_KEY: Optional for article search

ASSEMBLYAI_API_KEY: Optional for audio transcription