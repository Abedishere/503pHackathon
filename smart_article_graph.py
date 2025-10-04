import os
import json
import streamlit as st
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import io
import assemblyai as aai
import time

# Load environment variables
load_dotenv()

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY


def extract_entities_and_relations(article_text):
    """Uses Gemini to extract structured graph data from article"""

    prompt = f"""You are an expert at extracting structured relationship data from political articles.

Analyze this article and extract political/lobbying connections.

Article:
{article_text}

Extract:
1. All people, organizations, companies, politicians mentioned
2. Their relationships (lobbying, funding, employment, affiliation, etc.)
3. Any money amounts involved

Return ONLY valid JSON in this EXACT format (no markdown, no code blocks, just the JSON):
{{
  "entities": [
    {{"name": "Entity Name", "type": "person|organization|politician|company", "description": "brief context"}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "lobbies|funds|works_for|affiliated_with", "details": "brief description", "amount": "optional $ amount"}}
  ]
}}

Be comprehensive but accurate. Only include entities and relationships explicitly mentioned."""

    try:
        response = model.generate_content(prompt)

        # Clean response text - remove markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON response
        result = json.loads(response_text)

        # Ensure required keys exist
        if 'entities' not in result:
            result['entities'] = []
        if 'relationships' not in result:
            result['relationships'] = []

        return result
    except Exception as e:
        # Log the error for debugging
        st.error(f"Entity extraction error: {str(e)}")
        # Return empty structure if parsing fails
        return {"entities": [], "relationships": []}


def split_into_paragraphs(article_text):
    """Split article into paragraphs"""
    # Split by double newlines, or single newlines if no double newlines exist
    paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
    if len(paragraphs) == 1:
        # Try single newlines
        paragraphs = [p.strip() for p in article_text.split('\n') if p.strip()]
    # Filter out very short paragraphs (less than 50 chars)
    paragraphs = [p for p in paragraphs if len(p) > 50]
    return paragraphs if paragraphs else [article_text]


def split_into_chunks(article_text, num_chunks=5):
    """Split article into percentage-based chunks (default 5 chunks = 20% each)"""
    words = article_text.split()
    chunk_size = max(1, len(words) // num_chunks)
    chunks = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(words)
        chunk = ' '.join(words[start_idx:end_idx])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def find_entity_context(entity_name, articles):
    """Find paragraphs containing the entity and surrounding context"""
    contexts = []

    for article_idx, article in enumerate(articles):
        paragraphs = split_into_paragraphs(article)

        for i, paragraph in enumerate(paragraphs):
            if entity_name.lower() in paragraph.lower():
                # Get context: previous paragraph + current + next paragraph
                context_parts = []
                if i > 0:
                    context_parts.append(paragraphs[i-1])
                context_parts.append(paragraph)
                if i < len(paragraphs) - 1:
                    context_parts.append(paragraphs[i+1])

                context = " ".join(context_parts)
                contexts.append({
                    "article_idx": article_idx + 1,
                    "context": context
                })

    return contexts


def find_cross_article_connections(all_entities, articles):
    """Use LLM to find connections between entities across all articles"""

    entity_names = [e["name"] for e in all_entities]

    # Create a summary of all articles and entities
    entities_summary = "\n".join([f"- {e['name']} ({e['type']}): {e.get('description', '')}" for e in all_entities])
    articles_summary = "\n\n".join([f"Article {i+1}:\n{article[:500]}..." if len(article) > 500 else f"Article {i+1}:\n{article}"
                                     for i, article in enumerate(articles)])

    prompt = f"""You are analyzing multiple political articles to find ALL relationships between entities.

Known Entities:
{entities_summary}

Articles:
{articles_summary}

Your task: Find EVERY relationship between ANY of the known entities mentioned across ALL articles.

Rules:
1. Look for direct connections (funding, lobbying, employment, affiliation)
2. Look for indirect connections (both mentioned in same context, same organization, same event)
3. Look for implied relationships (if A funds B and B employs C, note A→B and B→C)
4. Be thorough - find ALL connections, even weak ones
5. Include context and money amounts when mentioned

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "lobbies|funds|works_for|affiliated_with|mentioned_with|related_to", "details": "specific description from articles", "amount": "optional $ amount"}}
  ]
}}

Find as many connections as possible. Every entity should connect to at least one other entity."""

    try:
        response = model.generate_content(prompt)

        # Clean response
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        result = json.loads(response_text)

        # Ensure 'relationships' key exists
        if 'relationships' not in result:
            return {"relationships": []}

        return result
    except Exception as e:
        # Return empty relationships if parsing fails
        return {"relationships": []}


def build_graph(data, existing_graph=None):
    """Converts extracted data to NetworkX graph, optionally merging with existing graph"""
    if existing_graph is None:
        G = nx.Graph()
    else:
        G = existing_graph.copy()

    # Ensure data has required keys
    if not isinstance(data, dict):
        return G

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # Add nodes (merge with existing ones)
    for entity in entities:
        if not isinstance(entity, dict) or "name" not in entity:
            continue

        if G.has_node(entity["name"]):
            # Node exists, update attributes if new info is more detailed
            existing_desc = G.nodes[entity["name"]].get("description", "")
            new_desc = entity.get("description", "")
            if len(new_desc) > len(existing_desc):
                G.nodes[entity["name"]]["description"] = new_desc
                G.nodes[entity["name"]]["title"] = f"{entity['name']}\nType: {entity.get('type', 'unknown')}\n{new_desc}"
        else:
            # New node
            G.add_node(
                entity["name"],
                type=entity.get("type", "unknown"),
                description=entity.get("description", ""),
                title=f"{entity['name']}\nType: {entity.get('type', 'unknown')}\n{entity.get('description', '')}"
            )

    # Add edges (merge with existing ones)
    for rel in relationships:
        if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
            continue

        # Make sure both nodes exist
        if not G.has_node(rel["source"]):
            G.add_node(rel["source"], type="unknown", description="", title=rel["source"])
        if not G.has_node(rel["target"]):
            G.add_node(rel["target"], type="unknown", description="", title=rel["target"])

        if G.has_edge(rel["source"], rel["target"]):
            # Edge exists, append details if different
            existing_details = G.edges[rel["source"], rel["target"]].get("details", "")
            new_details = rel.get("details", "")
            if new_details and new_details not in existing_details:
                combined_details = f"{existing_details}; {new_details}" if existing_details else new_details
                G.edges[rel["source"], rel["target"]]["details"] = combined_details
                G.edges[rel["source"], rel["target"]]["title"] = f"{rel.get('type', 'related_to')}: {combined_details}"
            # Update amount if new one provided
            if rel.get("amount"):
                G.edges[rel["source"], rel["target"]]["amount"] = rel.get("amount")
        else:
            # New edge
            label = rel.get("type", "related_to")
            if rel.get("amount"):
                label += f"\n{rel['amount']}"

            G.add_edge(
                rel["source"],
                rel["target"],
                relationship=rel.get("type", "related_to"),
                details=rel.get("details", ""),
                amount=rel.get("amount", ""),
                title=f"{rel.get('type', 'related_to')}: {rel.get('details', '')}"
            )

    return G


def search_articles_serper(query):
    """Search for articles using Serper API with timeout and better error handling"""
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": 1  # Only get top 1 result
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        # Add timeout to prevent hanging
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()

        articles = []

        # Try 'news' first (if news results exist)
        if 'news' in results and results['news']:
            item = results['news'][0]  # Only get first result
            articles.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })
        # Fall back to 'organic' results
        elif 'organic' in results and results['organic']:
            item = results['organic'][0]  # Only get first result
            articles.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })

        return articles
    except requests.exceptions.Timeout:
        st.error("Search timed out. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Search error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected search error: {str(e)}")
        return []


def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF, TXT, DOC, DOCX)"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'txt':
            # Read text file
            content = uploaded_file.read().decode("utf-8", errors="replace")
            return content.strip()

        elif file_extension == 'pdf':
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()

        elif file_extension in ['doc', 'docx']:
            # Read Word document
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return text.strip()

        else:
            return f"Unsupported file type: {file_extension}"

    except Exception as e:
        raise Exception(f"Error extracting text from {file_extension} file: {str(e)}")


def transcribe_audio(audio_file):
    """Transcribe audio file to text using AssemblyAI"""
    try:
        # Configure transcription
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)

        # Create transcriber and transcribe
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_file)

        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        return transcript.text
    except Exception as e:
        raise Exception(f"Audio transcription error: {str(e)}")


def fetch_article_content(url):
    """Fetch article content from URL with timeout and better error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Add timeout to prevent hanging
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        # If no paragraphs found, try to get all text
        if not text:
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)

        return text[:15000]  # Limit to avoid token limits
    except requests.exceptions.Timeout:
        return "Could not fetch content: Request timed out"
    except Exception as e:
        return f"Could not fetch content from {url}: {str(e)}"


def visualize_graph(graph):
    """Creates interactive visualization with color coding"""
    net = Network(height="750px", width="100%", bgcolor="#1e1e1e", font_color="white")

    # Color code by entity type
    colors = {
        "person": "#FF6B6B",
        "organization": "#4ECDC4",
        "politician": "#FFD93D",
        "company": "#95E1D3"
    }

    # Add nodes with colors
    for node, attrs in graph.nodes(data=True):
        color = colors.get(attrs.get("type", ""), "#A8E6CF")
        net.add_node(
            node,
            label=node,
            title=attrs.get("title", node),
            color=color,
            size=25
        )

    # Add edges
    for source, target, attrs in graph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=attrs.get("title", ""),
            label=attrs.get("relationship", "")
        )

    # Physics settings for better layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "springLength": 200,
          "springConstant": 0.001
        }
      }
    }
    """)

    return net


def query_graph_data(question, graph_data, articles):
    """Use LLM to answer questions about the extracted graph data"""
    
    # Format the graph data for the prompt
    entities_text = "\n".join([
        f"- {entity['name']} ({entity['type']}): {entity.get('description', 'No description')}"
        for entity in graph_data.get('entities', [])
    ])
    
    relationships_text = "\n".join([
        f"- {rel['source']} → {rel['target']} ({rel['type']}): {rel.get('details', 'No details')}{' - Amount: ' + rel['amount'] if rel.get('amount') else ''}"
        for rel in graph_data.get('relationships', [])
    ])
    
    # Get article summaries
    articles_summary = "\n\n".join([
        f"Article {i+1} (first 200 chars): {article[:200]}..."
        for i, article in enumerate(articles)
    ])
    
    prompt = f"""You are an expert analyst for political and lobbying connections. 
I have extracted the following data from political articles. Please answer the user's question based on this data.

EXTRACTED ENTITIES:
{entities_text}

EXTRACTED RELATIONSHIPS:
{relationships_text}

ORIGINAL ARTICLES SUMMARY:
{articles_summary}

USER QUESTION: {question}

Please provide a comprehensive answer based on the extracted data. If the data doesn't contain enough information to answer fully, say so and make reasonable inferences where appropriate.

Focus on:
- Political connections and lobbying relationships
- Financial transactions and amounts
- Organizational hierarchies and affiliations
- Potential conflicts of interest

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Article → Graph", layout="wide")

st.title("📰 Political Article → Connection Graph")
st.markdown("Paste any political or lobbying article. AI extracts entities and relationships automatically.")

# Sidebar with examples
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Paste article** text
    2. **Gemini AI extracts** people, orgs, relationships
    3. **Graph shows** connections
    4. **Chat** with your data

    **Color Key:**
    - 🔴 Person
    - 🔵 Organization
    - 🟡 Politician
    - 🟢 Company
    """)

    if st.button("Load Example Article"):
        st.session_state.example_loaded = True

# Example article
example = """Senator Jane Smith received $50,000 in campaign contributions from the pharmaceutical industry's
main lobbying group, PhRMA, according to recent FEC filings. The donation came through the firm Akin Gump LLP,
which represents PhRMA and other healthcare companies. Smith chairs the Senate Health Committee and has been
a vocal supporter of drug pricing legislation favored by the industry. Meanwhile, her chief of staff,
Robert Johnson, previously worked as a lobbyist for Pfizer before joining Smith's office in 2023."""

# Initialize session state for articles
if 'articles' not in st.session_state:
    st.session_state.articles = []

# Main input section - all input methods together
st.subheader("📝 Add Article Content")

# Article input method - now includes Search
input_method = st.radio("Input method:", ["Search article", "Paste text", "Upload file", "Upload audio"], horizontal=True)

if input_method == "Search article":
    st.markdown("Search for a political connection and automatically fetch the top article")
    
    search_query = st.text_input("Enter your search query:", 
                                placeholder="e.g., pharmaceutical lobbying congress, oil industry climate policy")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔎 Search & Load Article", type="primary"):
            if search_query.strip():
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Search for articles
                    status_text.text("🔍 Searching for articles...")
                    progress_bar.progress(25)
                    
                    search_results = search_articles_serper(search_query)

                    if search_results:
                        st.session_state.articles = []  # Clear existing articles

                        # Step 2: Fetch article content
                        status_text.text("📰 Fetching article content...")
                        progress_bar.progress(50)

                        result = search_results[0]  # Only one result
                        
                        # Add a timeout for content fetching
                        start_time = time.time()
                        content = fetch_article_content(result['link'])
                        
                        # Step 3: Process results
                        status_text.text("⚙️ Processing article...")
                        progress_bar.progress(75)

                        # Add title and source to content
                        full_content = f"Title: {result['title']}\nSource: {result['link']}\n\n{content}"
                        st.session_state.articles.append(full_content)

                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()

                        st.success(f"✅ Found and loaded article about '{search_query}'")

                        # Show found article
                        with st.expander("📑 Article Found"):
                            st.markdown(f"**{result['title']}**")
                            st.markdown(f"🔗 {result['link']}")
                            st.markdown(f"_{result['snippet']}_")
                            
                            # Show content preview
                            if len(content) > 500:
                                st.markdown(f"**Content Preview:** {content[:500]}...")
                            else:
                                st.markdown(f"**Content:** {content}")

                        st.rerun()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("No article found. Try a different search query or use another input method.")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Search failed: {str(e)}")
                    st.info("💡 Try using 'Paste text' method instead and search for articles manually.")
            else:
                st.error("Please enter a search query!")

elif input_method == "Paste text":
    # Use session state to control the text area value
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    if st.session_state.get("example_loaded") and not st.session_state.text_input:
        st.session_state.text_input = example
        st.session_state.example_loaded = False

    article_text = st.text_area("Article Text:", value=st.session_state.text_input, height=300, key="paste_input")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add Article"):
            if article_text.strip():
                st.session_state.articles = [article_text.strip()]  # Replace with single article
                st.session_state.text_input = ""  # Clear the input
                st.success(f"✅ Article added successfully!")
                st.rerun()
            else:
                st.error("Please paste an article first!")

elif input_method == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload article file (PDF, TXT, DOC, DOCX - max 3MB)",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=False
    )

    if st.button("➕ Add Uploaded File"):
        if uploaded_file:
            # Check file size (3MB = 3 * 1024 * 1024 bytes)
            if uploaded_file.size > 3 * 1024 * 1024:
                st.error("File size exceeds 3MB limit. Please upload a smaller file.")
            else:
                try:
                    # Extract text from file
                    content = extract_text_from_file(uploaded_file)

                    if content.strip():
                        st.session_state.articles = [content.strip()]  # Replace with single article
                        st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                        st.rerun()
                    else:
                        st.error("File is empty or no text could be extracted. Please upload a file with content.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            st.error("Please upload a file first!")

else:  # Upload audio
    uploaded_audio = st.file_uploader(
        "Upload audio file (MP3, WAV, M4A, MP4 - max 3MB)",
        type=["mp3", "wav", "m4a", "mp4"],
        accept_multiple_files=False
    )

    if st.button("🎤 Transcribe & Add Audio"):
        if uploaded_audio:
            # Check file size (3MB = 3 * 1024 * 1024 bytes)
            if uploaded_audio.size > 3 * 1024 * 1024:
                st.error("File size exceeds 3MB limit. Please upload a smaller file.")
            else:
                try:
                    with st.spinner("🎤 Transcribing audio... This may take a moment."):
                        # Transcribe audio file
                        transcript_text = transcribe_audio(uploaded_audio)

                        if transcript_text.strip():
                            st.session_state.articles = [transcript_text.strip()]  # Replace with transcribed text
                            st.success(f"✅ Audio '{uploaded_audio.name}' transcribed successfully!")

                            # Show transcribed text preview
                            with st.expander("📝 Transcribed Text Preview"):
                                st.text_area("Transcript", value=transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text, height=150, disabled=True)

                            st.rerun()
                        else:
                            st.error("No speech detected in the audio file. Please upload a different file.")
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
        else:
            st.error("Please upload an audio file first!")

# Show current articles
if st.session_state.articles:
    st.markdown(f"**📚 Articles loaded: {len(st.session_state.articles)}**")

    with st.expander("View/Remove Articles"):
        for i, article in enumerate(st.session_state.articles):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text_area(f"Article {i+1}", value=article[:200] + "..." if len(article) > 200 else article,
                           height=100, key=f"article_{i}", disabled=True)
            with col2:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.articles.pop(i)
                    st.rerun()

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        extract_button = st.button("🔍 Extract Graph from All", type="primary")
    with col2:
        if st.button("🧹 Clear All"):
            st.session_state.articles = []
            st.rerun()
else:
    st.info("👆 Add articles using the input method above")
    extract_button = False

# Initialize session state for graph data
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

# Process articles
if extract_button and st.session_state.articles:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        graph = None
        all_data = {"entities": [], "relationships": []}

        # PASS 1: Extract entities and initial relationships from each article
        for i, article in enumerate(st.session_state.articles):
            status_text.text(f"🤖 Pass 1: Extracting from article {i+1}/{len(st.session_state.articles)}...")
            progress_bar.progress((i) / (len(st.session_state.articles) + 2))

            # Extract structured data from this article
            data = extract_entities_and_relations(article)

            # Merge entities and relationships (avoiding duplicates)
            for entity in data["entities"]:
                if not any(e["name"] == entity["name"] for e in all_data["entities"]):
                    all_data["entities"].append(entity)

            for rel in data["relationships"]:
                if not any(r["source"] == rel["source"] and r["target"] == rel["target"]
                          for r in all_data["relationships"]):
                    all_data["relationships"].append(rel)

            # Build/update graph incrementally
            graph = build_graph(data, graph)

        # PASS 2: Find cross-article connections
        if all_data["entities"]:  # Only if we have entities
            status_text.text(f"🤖 Pass 2: Finding connections across all {len(st.session_state.articles)} articles...")
            progress_bar.progress((len(st.session_state.articles)) / (len(st.session_state.articles) + 2))

            cross_connections = find_cross_article_connections(all_data["entities"], st.session_state.articles)

            # Add cross-article relationships
            if "relationships" in cross_connections:
                for rel in cross_connections["relationships"]:
                    if not any(r["source"] == rel["source"] and r["target"] == rel["target"]
                              for r in all_data["relationships"]):
                        all_data["relationships"].append(rel)

                # Update graph with cross-article connections
                graph = build_graph(cross_connections, graph)

        # PASS 3: Multi-iteration isolated entity connection (3 iterations)
        if graph and graph.number_of_nodes() > 0:
            # Run 3 iterations to ensure all entities get connected
            for iteration in range(1, 4):
                status_text.text(f"🤖 Pass 3.{iteration}: Connecting isolated entities (iteration {iteration}/3)...")
                progress_bar.progress((len(st.session_state.articles) + 1) / (len(st.session_state.articles) + 2))

                # Find isolated nodes (entities with no connections)
                isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

                if not isolated_nodes:
                    # No more isolated entities, we're done
                    break

                # Get all entity names for reference
                all_entity_names = [e["name"] for e in all_data["entities"]]

                # Process each isolated entity
                for isolated_entity in isolated_nodes:
                    # Find context around this entity in the articles
                    contexts = find_entity_context(isolated_entity, st.session_state.articles)

                    if contexts:
                        # Build context summary
                        context_summary = "\n\n".join([
                            f"From Article {ctx['article_idx']}:\n{ctx['context'][:500]}"
                            for ctx in contexts[:3]  # Limit to 3 contexts
                        ])

                        # On the 3rd iteration, don't allow new entities
                        if iteration == 3:
                            prompt = f"""Analyze the context around the entity "{isolated_entity}" to find relationships.

IMPORTANT: Only connect to entities from the existing list below. DO NOT create new entities.

Entity to connect: {isolated_entity}

Available entities to connect with (ONLY use these):
{', '.join(all_entity_names[:30])}

Context where "{isolated_entity}" is mentioned:
{context_summary}

Find ANY relationships between "{isolated_entity}" and other entities from the available list. Return ONLY JSON:
{{
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "lobbies|funds|works_for|affiliated_with|mentioned_with|related_to", "details": "description from context"}}
  ]
}}

Remember: Do not introduce any new entities. Only use entities from the available list."""
                        else:
                            prompt = f"""Analyze the context around the entity "{isolated_entity}" to find relationships.

Entity to connect: {isolated_entity}

Available entities to connect with:
{', '.join(all_entity_names[:30])}

Context where "{isolated_entity}" is mentioned:
{context_summary}

Find ANY relationships between "{isolated_entity}" and other entities based on this context. You can mention new entities if they appear in the context. Return ONLY JSON:
{{
  "entities": [
    {{"name": "Entity Name", "type": "person|organization|politician|company", "description": "brief context"}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "lobbies|funds|works_for|affiliated_with|mentioned_with|related_to", "details": "description from context"}}
  ]
}}"""

                        try:
                            response = model.generate_content(prompt)
                            response_text = response.text.strip()
                            if response_text.startswith("```json"):
                                response_text = response_text[7:]
                            if response_text.startswith("```"):
                                response_text = response_text[3:]
                            if response_text.endswith("```"):
                                response_text = response_text[:-3]
                            response_text = response_text.strip()

                            entity_connections = json.loads(response_text)

                            # Add new entities (only if not iteration 3)
                            if iteration < 3 and "entities" in entity_connections:
                                for entity in entity_connections["entities"]:
                                    if not any(e["name"] == entity["name"] for e in all_data["entities"]):
                                        all_data["entities"].append(entity)

                            # Add relationships
                            if "relationships" in entity_connections:
                                for rel in entity_connections["relationships"]:
                                    if not any(r["source"] == rel["source"] and r["target"] == rel["target"]
                                              for r in all_data["relationships"]):
                                        all_data["relationships"].append(rel)

                                graph = build_graph(entity_connections, graph)
                        except Exception as e:
                            # If this entity fails, continue with next
                            continue

            # FINAL VALIDATION: Keep trying to connect floating nodes until fewer than 3 remain
            if graph and graph.number_of_nodes() > 0:
                max_additional_iterations = 10  # Safety limit
                additional_iteration = 0

                while additional_iteration < max_additional_iterations:
                    # Find isolated nodes
                    isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

                    # Stop if we have fewer than 3 floating nodes (acceptable threshold)
                    if len(isolated_nodes) < 3:
                        break

                    additional_iteration += 1
                    status_text.text(f"🤖 Final validation {additional_iteration}/10: Connecting {len(isolated_nodes)} floating entities...")

                    # Get all entity names for reference
                    all_entity_names = [e["name"] for e in all_data["entities"]]

                    # Process ALL isolated entities this iteration
                    connections_made = False
                    for isolated_entity in isolated_nodes:
                        # Find ALL contexts around this entity in the articles
                        contexts = find_entity_context(isolated_entity, st.session_state.articles)

                        if not contexts:
                            continue

                        # Build FULL context summary - include more context to be thorough
                        context_summary = "\n\n".join([
                            f"From Article {ctx['article_idx']}:\n{ctx['context']}"
                            for ctx in contexts  # Include ALL contexts, not just first 3
                        ])

                        # Truncate if too long (keep last 3000 chars to stay within limits)
                        if len(context_summary) > 3000:
                            context_summary = "..." + context_summary[-3000:]

                        prompt = f"""You are analyzing text to find connections that were previously missed.

Entity needing connections: "{isolated_entity}"

STRICT RULES:
1. ONLY connect to entities from the existing list below
2. ONLY mention relationships that are EXPLICITLY stated or CLEARLY implied in the context
3. DO NOT create new entities
4. DO NOT make up connections that aren't supported by the text
5. If no connection exists in the context, return empty relationships array

Available entities (ONLY use these):
{', '.join(all_entity_names[:50])}

Full context where "{isolated_entity}" appears:
{context_summary}

Analyze the context carefully. Find ONLY real, text-supported relationships. Return ONLY JSON:
{{
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "lobbies|funds|works_for|affiliated_with|mentioned_with|related_to", "details": "exact quote or clear reference from context"}}
  ]
}}

If no clear connection exists in the text, return: {{"relationships": []}}"""

                        try:
                            response = model.generate_content(prompt)
                            response_text = response.text.strip()
                            if response_text.startswith("```json"):
                                response_text = response_text[7:]
                            if response_text.startswith("```"):
                                response_text = response_text[3:]
                            if response_text.endswith("```"):
                                response_text = response_text[:-3]
                            response_text = response_text.strip()

                            entity_connections = json.loads(response_text)

                            # Add relationships only if they exist
                            if "relationships" in entity_connections and entity_connections["relationships"]:
                                connections_made = True
                                for rel in entity_connections["relationships"]:
                                    if not any(r["source"] == rel["source"] and r["target"] == rel["target"]
                                              for r in all_data["relationships"]):
                                        all_data["relationships"].append(rel)

                                graph = build_graph(entity_connections, graph)
                        except Exception as e:
                            # If this entity fails, continue with next
                            continue

                    # If no connections were made this iteration, break to avoid infinite loop
                    if not connections_made:
                        break

                # After all attempts, remove any remaining isolated nodes (should be < 3)
                final_isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

                if final_isolated_nodes:
                    # Remove isolated nodes from graph
                    for node in final_isolated_nodes:
                        graph.remove_node(node)

                    # Remove from entities list
                    all_data["entities"] = [e for e in all_data["entities"] if e["name"] not in final_isolated_nodes]

                    # Show info or warning based on count
                    
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # Store graph data in session state for chatbot
        st.session_state.graph_data = all_data

        # Check if we have a valid graph
        if not graph or graph.number_of_nodes() == 0:
            st.error("No entities or connections were found in the articles. Please check the article content and try again.")
        else:
            # Display results
            st.success(f"✅ Found **{graph.number_of_nodes()} entities** and **{graph.number_of_edges()} connections** from {len(st.session_state.articles)} articles")

            # Show extracted data
            with st.expander("📊 View Raw Extracted Data"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Entities")
                    st.json(all_data["entities"])
                with col_b:
                    st.subheader("Relationships")
                    st.json(all_data["relationships"])

            # Show connections as text
            st.subheader("🔗 Connections Found:")
            for source, target, attrs in graph.edges(data=True):
                rel_type = attrs.get("relationship", "related to")
                details = attrs.get("details", "")
                amount = attrs.get("amount", "")

                display_text = f"**{source}** → *{rel_type}* → **{target}**"
                if amount:
                    display_text += f" ({amount})"
                if details:
                    display_text += f" — {details}"

                st.markdown(display_text)

            # Visualize graph
            st.subheader("🌐 Interactive Graph Visualization")
            st.markdown("*Hover over nodes/edges for details. Drag to explore.*")

            net = visualize_graph(graph)
            html = net.generate_html()
            st.components.v1.html(html, height=800)

            # Download options
            st.subheader("💾 Export Options")
            col1, col2 = st.columns(2)

            with col1:
                # Export as JSON
                export_json = json.dumps(all_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_json,
                    file_name="graph_data.json",
                    mime="application/json"
                )

            with col2:
                # Export graph as GraphML
                import io

                graphml_buffer = io.BytesIO()
                nx.write_graphml(graph, graphml_buffer)
                st.download_button(
                    label="Download GraphML",
                    data=graphml_buffer.getvalue(),
                    file_name="graph.graphml",
                    mime="application/xml"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure your Gemini API key is set correctly.")

# CHATBOT SECTION
if st.session_state.graph_data:
    st.markdown("---")
    st.subheader("💬 Chat with Your Graph Data")
    st.markdown("Ask questions about the entities, relationships, and connections in your graph.")
    
    # Suggested questions
    st.markdown("**💡 Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Show all politicians"):
            st.session_state.chat_input = "Show me all the politicians mentioned and their connections"
    with col2:
        if st.button("Find financial relationships"):
            st.session_state.chat_input = "What financial relationships or donations are mentioned?"
    with col3:
        if st.button("Identify key organizations"):
            st.session_state.chat_input = "What are the most connected organizations?"
    
    # Chat interface
    chat_input = st.text_area(
        "Ask a question about your graph:",
        placeholder="e.g., Who received the most funding? What are the connections between politicians and companies?",
        key="chat_input",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🚀 Ask", type="primary"):
            if chat_input.strip():
                with st.spinner("Analyzing your graph data..."):
                    # Add user question to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": chat_input,
                        "timestamp": "now"
                    })
                    
                    # Get AI response
                    response = query_graph_data(
                        chat_input, 
                        st.session_state.graph_data, 
                        st.session_state.articles
                    )
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": "now"
                    })
                    
                    st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Analyst:** {message['content']}")
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
    
   

elif extract_button and not st.session_state.graph_data:
    st.info("📊 Generate a graph first to enable the chatbot feature!")