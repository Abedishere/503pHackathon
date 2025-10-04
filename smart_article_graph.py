import os
import json
import streamlit as st
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Initialize Gemini
genai.configure(api_key="AIzaSyBNI53y5LjGOLtDm3yXjIna8hK-rL82Gxg")
model = genai.GenerativeModel('gemini-2.0-flash-exp')


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

# Main input
st.subheader("📝 Add Articles")

# Initialize session state for articles
if 'articles' not in st.session_state:
    st.session_state.articles = []

# Article input method
input_method = st.radio("Input method:", ["Paste text", "Upload files"], horizontal=True)

if input_method == "Paste text":
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
                st.session_state.articles.append(article_text.strip())
                st.session_state.text_input = ""  # Clear the input
                st.success(f"✅ Article added successfully! Total: {len(st.session_state.articles)}")
                st.rerun()
            else:
                st.error("Please paste an article first!")

else:  # Upload files
    uploaded_files = st.file_uploader(
        "Upload article files (txt, pdf, or other text files)",
        type=["txt", "pdf", "doc", "docx"],
        accept_multiple_files=True
    )

    if st.button("➕ Add Uploaded Files"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
                if content.strip():
                    st.session_state.articles.append(content.strip())
            st.success(f"Added {len(uploaded_files)} articles! Total: {len(st.session_state.articles)}")
            st.rerun()
        else:
            st.error("Please upload files first!")

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

        # PASS 3: Final validation - ensure isolated entities get connected
        if graph and graph.number_of_nodes() > 0:
            status_text.text(f"🤖 Pass 3: Validating all entities are connected...")
            progress_bar.progress((len(st.session_state.articles) + 1) / (len(st.session_state.articles) + 2))

            # Find isolated nodes (entities with no connections)
            isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

            if isolated_nodes and len(isolated_nodes) < len(all_data["entities"]):
                # Try to connect isolated nodes
                connected_nodes = [node for node in graph.nodes() if graph.degree(node) > 0]

                isolated_summary = ", ".join(isolated_nodes[:10])  # Limit to first 10
                connected_summary = ", ".join(connected_nodes[:10])

                articles_text = "\n\n".join(st.session_state.articles)

                prompt = f"""Looking at these articles, find connections for isolated entities.

Isolated entities (need connections): {isolated_summary}
Connected entities: {connected_summary}

Articles:
{articles_text[:3000]}

Find ANY connections between isolated entities and other entities. Return ONLY JSON:
{{
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "relationship_type", "details": "description"}}
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

                    final_connections = json.loads(response_text)

                    if "relationships" in final_connections:
                        for rel in final_connections["relationships"]:
                            if not any(r["source"] == rel["source"] and r["target"] == rel["target"]
                                      for r in all_data["relationships"]):
                                all_data["relationships"].append(rel)

                        graph = build_graph(final_connections, graph)
                except Exception as e:
                    # If pass 3 fails, just continue with what we have
                    pass

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

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