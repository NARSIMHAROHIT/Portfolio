import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="LLMOps Learning Journey",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .phase-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .project-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .skill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 15px;
        background-color: #667eea;
        color: white;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/logo.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Go to:",
        [" Home", " Learning Path", "Projects", "Progress", "Settings"]
    )
    
    st.markdown("---")
    st.markdown("### API Configuration")
    api_key = st.text_input("Grok API Key", type="password", help="Enter your Grok API key")
    if api_key:
        st.success("API Key set!")
        # Store in session state
        st.session_state['grok_api_key'] = api_key
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Projects Completed", "0/18")
    st.metric("Current Phase", "Phase 1")
    st.metric("Days Active", "1")

# Main Content
if page == " Home":
    st.markdown('<h1 class="main-header"> LLMOps Learning Journey</h1>', unsafe_allow_html=True)
    st.markdown("### *From Fundamentals to Production-Ready AI Systems*")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Mission")
        st.write("""
        Master the complete LLMOps stack by building 18 hands-on projects,
        from basic embeddings to multi-agent production systems.
        """)
    
    with col2:
        st.markdown("###  Tech Stack")
        st.markdown("""
        - **LLM**: Grok, Groq, Local Models
        - **Frameworks**: LangChain, LangGraph
        - **Vector DB**: ChromaDB, Pinecone
        - **UI**: Streamlit + CLI
        """)
    
    with col3:
        st.markdown("###  Journey Phases")
        st.markdown("""
        1. **Foundations** (3 projects)
        2. **RAG Systems** (3 projects)
        3. **Agents** (3 projects)
        4. **MCP Servers** (3 projects)
        5. **Multi-Agent** (3 projects)
        6. **Production** (3 projects)
        """)
    
    st.markdown("---")
    
    # Current Focus
    st.markdown("## Current Focus: Phase 1 - Foundations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What You'll Learn:
        - **Tokenization**: How text is converted to numbers
        - **Embeddings**: Vector representations of meaning
        - **Corpus Management**: Organizing and processing text data
        - **Chunking Strategies**: Breaking text into optimal pieces
        - **Vector Search**: Finding similar content efficiently
        """)
    
    with col2:
        st.markdown("### Next Projects:")
        st.markdown("""
        1. Text Embeddings Explorer
        2. Smart Chunker
        3. Similarity Search Engine
        """)
    
    st.markdown("---")
    
    # Quick Start
    st.markdown("## Quick Start")
    
    tab1, tab2, tab3 = st.tabs(["Setup", "First Project", "Resources"])
    
    with tab1:
        st.markdown("""
        ### Environment Setup
        ```bash
        # Clone the repository
        git clone https://github.com/yourusername/llmops-journey
        cd llmops-journey
        
        # Create virtual environment
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\\Scripts\\activate
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the portfolio
        streamlit run app.py
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Project 1: Text Embeddings Explorer
        
        **What you'll build:**
        - Interactive visualization of text embeddings
        - Compare different embedding models
        - Understand semantic similarity
        
        **Skills learned:**
        - Working with embedding models
        - Vector mathematics
        - Dimensionality reduction (t-SNE, UMAP)
        
        **Time estimate:** 2-3 hours
        """)
        
        if st.button("Start Project 1"):
            st.balloons()
            st.success("Great! Navigate to the Projects page to begin!")
    
    with tab3:
        st.markdown("""
        ### Learning Resources
        
        **Documentation:**
        - [LangChain Docs](https://python.langchain.com/)
        - [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
        - [Grok API Docs](https://docs.x.ai/api)
        
        **Concepts:**
        - [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)
        - [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)
        - [AI Agents Overview](https://www.anthropic.com/research/agents)
        
        **Ethics & Safety:**
        - [AI Safety Guidelines](https://www.anthropic.com/index/core-views-on-ai-safety)
        - [Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
        """)

elif page == "📚 Learning Path":
    st.title("📚 Complete Learning Path")
    
    # Phase 1
    with st.expander("**Phase 1: Foundations** (Projects 1-3)", expanded=True):
        st.markdown("""
        <div class="phase-card">
            <h3>Learning Objectives</h3>
            <ul>
                <li>Understand how LLMs process text (tokenization)</li>
                <li>Master vector embeddings and semantic search</li>
                <li>Learn optimal text chunking strategies</li>
                <li>Build foundational skills for RAG systems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        projects = [
            {
                "num": 1,
                "name": "Text Embeddings Explorer",
                "skills": ["Tokenization", "Embeddings", "Visualization"],
                "status": "Ready"
            },
            {
                "num": 2,
                "name": "Smart Chunker",
                "skills": ["Chunking", "Text Processing", "Comparison"],
                "status": "Locked"
            },
            {
                "num": 3,
                "name": "Similarity Search Engine",
                "skills": ["Vector Search", "Distance Metrics", "Ranking"],
                "status": "Locked"
            }
        ]
        
        for proj in projects:
            st.markdown(f"""
            <div class="project-card">
                <h4>Project {proj['num']}: {proj['name']} - {proj['status']}</h4>
                <p><strong>Skills:</strong> {' '.join([f'<span class="skill-badge">{s}</span>' for s in proj['skills']])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Phase 2-6 (collapsed by default)
    phases = [
        {
            "name": "Phase 2: RAG Fundamentals",
            "projects": ["Simple RAG Bot", "Multi-Source RAG", "Advanced RAG"],
            "skills": "RAG Architecture, Retrieval, Context Management"
        },
        {
            "name": "Phase 3: Agent Basics",
            "projects": ["Tool-Using Agent", "Conversational Agent", "Research Agent"],
            "skills": "ReAct Pattern, Tool Calling, Memory"
        },
        {
            "name": "Phase 4: MCP Servers",
            "projects": ["Custom MCP Server", "API Integration MCP", "Database MCP"],
            "skills": "MCP Protocol, Server Architecture, Integration"
        },
        {
            "name": "Phase 5: Multi-Agent Systems",
            "projects": ["Collaborative Agents", "Specialized Team", "Autonomous System"],
            "skills": "Orchestration, Workflows, LangGraph"
        },
        {
            "name": "Phase 6: Production & MLOps",
            "projects": ["Monitoring Dashboard", "Guardrails System", "Full Production App"],
            "skills": "Monitoring, Safety, Deployment"
        }
    ]
    
    for i, phase in enumerate(phases, start=2):
        with st.expander(f"**{phase['name']}** (Projects {i*3-2}-{i*3})"):
            st.markdown(f"**Key Skills:** {phase['skills']}")
            for j, proj in enumerate(phase['projects'], start=1):
                st.markdown(f"{i*3-3+j}. {proj} 🔒")

elif page == "🛠️ Projects":
    st.title("🛠️ Project Workspace")
    
    st.info("👈 Select a project from the sidebar to get started!")
    
    # Project selector
    project_phase = st.selectbox(
        "Select Phase",
        ["Phase 1: Foundations", "Phase 2: RAG", "Phase 3: Agents", 
         "Phase 4: MCP", "Phase 5: Multi-Agent", "Phase 6: Production"]
    )
    
    if "Phase 1" in project_phase:
        project = st.selectbox(
            "Select Project",
            ["Project 1: Text Embeddings Explorer",
             "Project 2: Smart Chunker",
             "Project 3: Similarity Search Engine"]
        )
        
        if "Project 1" in project:
            st.markdown("## 🔤 Project 1: Text Embeddings Explorer")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "🎯 Goals", "💻 Code", "✅ Checklist"])
            
            with tab1:
                st.markdown("""
                ### What You'll Build
                An interactive tool to visualize and understand how different embedding models
                convert text into vector representations.
                
                ### Features
                - Compare multiple embedding models
                - Visualize embeddings in 2D/3D space
                - Calculate semantic similarity
                - Understand token counts and costs
                
                ### Tech Stack
                - Sentence Transformers
                - UMAP/t-SNE for visualization
                - Plotly for interactive charts
                - Streamlit for UI
                """)
            
            with tab2:
                st.markdown("""
                ### Learning Goals
                - [ ] Understand what embeddings are
                - [ ] Learn about different embedding models
                - [ ] Understand vector similarity metrics
                - [ ] Learn about dimensionality reduction
                - [ ] Practice with Streamlit UI development
                """)
            
            with tab3:
                st.info("Navigate to the GitHub repository to start coding!")
                st.code("""
# Coming soon: Project structure will be generated
# You'll get a complete template with:
# - src/ directory with modular code
# - ui/ directory with Streamlit app
# - cli/ directory with command-line interface
# - tests/ directory
# - Complete README and documentation
                """, language="bash")
            
            with tab4:
                st.markdown("""
                ### Project Checklist
                
                **Setup**
                - [ ] Create GitHub repository
                - [ ] Set up project structure
                - [ ] Install dependencies
                - [ ] Configure API keys
                
                **Development**
                - [ ] Implement embedding function
                - [ ] Add visualization
                - [ ] Create Streamlit UI
                - [ ] Add CLI interface
                - [ ] Write tests
                
                **Documentation**
                - [ ] Write README
                - [ ] Add code comments
                - [ ] Create usage examples
                - [ ] Document learnings
                
                **Deployment**
                - [ ] Push to GitHub
                - [ ] Add to portfolio
                - [ ] Write blog post
                """)

elif page == "Progress":
    st.title("our Learning Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", "0/18", "0%")
    with col2:
        st.metric("Current Streak", "1 day", "+1")
    with col3:
        st.metric("Skills Learned", "0/50", "0%")
    with col4:
        st.metric("Code Commits", "0", "-")
    
    st.markdown("---")
    
    # Progress by Phase
    st.markdown("### Progress by Phase")
    
    phases_progress = {
        "Phase 1: Foundations": 0,
        "Phase 2: RAG": 0,
        "Phase 3: Agents": 0,
        "Phase 4: MCP": 0,
        "Phase 5: Multi-Agent": 0,
        "Phase 6: Production": 0
    }
    
    for phase, progress in phases_progress.items():
        st.progress(progress / 3, text=f"{phase}: {progress}/3 projects")
    
    st.markdown("---")
    
    # Skills Tracker
    st.markdown("### Skills Mastery")
    
    skills_col1, skills_col2 = st.columns(2)
    
    with skills_col1:
        st.markdown("**Foundational Skills**")
        skills = ["Tokenization", "Embeddings", "Chunking", "Vector Search"]
        for skill in skills:
            st.progress(0, text=skill)
    
    with skills_col2:
        st.markdown("**Advanced Skills**")
        skills = ["RAG", "Agents", "MCP", "Multi-Agent"]
        for skill in skills:
            st.progress(0, text=skill)

else:  # Settings
    st.title(" Settings")
    
    st.markdown("### API Keys Management")
    
    with st.form("api_keys_form"):
        grok_key = st.text_input("Grok API Key", type="password")
        groq_key = st.text_input("Groq API Key (Optional)", type="password")
        pinecone_key = st.text_input("Pinecone API Key (Optional)", type="password")
        
        submitted = st.form_submit_button("Save API Keys")
        if submitted:
            st.success("API keys saved successfully!")
    
    st.markdown("---")
    
    st.markdown("### Preferences")
    
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    code_style = st.selectbox("Code Style", ["VS Code", "Monokai", "GitHub"])
    
    st.markdown("---")
    
    st.markdown("### Project Settings")
    
    github_username = st.text_input("GitHub Username")
    project_prefix = st.text_input("Project Prefix", value="llmops-")
    
    if st.button("Save Preferences"):
        st.success("Preferences saved!")
    
    st.markdown("---")
    
    st.markdown("### Danger Zone")
    st.error(" These actions cannot be undone")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Progress"):
            st.warning("This will reset all your progress!")
    with col2:
        if st.button("Clear All Data"):
            st.warning("This will delete all stored data!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p> Built with using Streamlit | 
        <a href="https://https://github.com/NARSIMHAROHIT/Portfolio">GitHub</a> | 
    </div>
""", unsafe_allow_html=True)