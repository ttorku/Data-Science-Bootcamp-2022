import streamlit as st
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from smart_similarity_search import SmartSimilaritySearch

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="LangChain Language Lookup Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2563eb);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .success-result {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .fail-result {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>🔍 LangChain Language Lookup Tool</h1>
    <p>Advanced document analysis using LangChain, FAISS, and LangGraph</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_system' not in st.session_state:
    st.session_state.search_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv('OPENAI_API_KEY', ''),
    type="password",
    help="Enter your OpenAI API key for embeddings and LLM calls"
)

# System settings
st.sidebar.subheader("🔧 System Settings")

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Minimum similarity score for a match"
)

similarity_k = st.sidebar.selectbox(
    "Top K Matches",
    options=[1, 3, 5, 10],
    index=1,
    help="Number of top matches to retrieve for each chunk"
)

# Ensemble Retrieval Settings
st.sidebar.subheader("🔍 Ensemble Retrieval")

similarity_method = st.sidebar.selectbox(
    "Similarity Method",
    options=['openai', 'sentence_transformer', 'tfidf'],
    index=0,
    help="Method for computing final similarities"
)

top_k_retrieval = st.sidebar.slider(
    "Top K Retrieval",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
    help="Number of candidates to retrieve from ensemble"
)

top_k_final = st.sidebar.slider(
    "Top K Final",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Number of final matches to compute similarity for"
)

# Advanced options
st.sidebar.subheader("🧠 Advanced Options")

enable_workflow = st.sidebar.checkbox(
    "Enable LangGraph Workflow",
    value=True,
    help="Use LangGraph for enhanced AI-powered analysis"
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=500,
    max_value=2000,
    value=1000,
    step=100,
    help="Size of text chunks for processing"
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Overlap between consecutive chunks"
)

# Main content
if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Set environment variable
os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize system
if st.session_state.search_system is None:
    with st.spinner("🔄 Initializing Smart Ensemble System..."):
        try:
            st.session_state.search_system = SmartSimilaritySearch()
            st.session_state.system_initialized = True
            st.success("✅ Ensemble retrieval system initialized successfully!")
            
            # Show system info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Languages", len(st.session_state.search_system.ensemble_retriever.languages))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Retrieval Methods", "LLM + BM25 + TF-IDF")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Similarity Options", "3 Methods")
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            st.stop()

# File upload section
st.header("📄 Document Upload")
uploaded_file = st.file_uploader(
    "Upload a PDF document for language analysis",
    type=['pdf'],
    help="Select a PDF file to analyze for AI/ML language content"
)

if uploaded_file is not None:
    # Display file info
    st.info(f"📁 **File**: {uploaded_file.name} | **Size**: {uploaded_file.size:,} bytes")
    
    # Process document
    if st.button("🚀 Analyze Document", type="primary"):
        
        # Update processor settings
        st.session_state.search_system.pdf_processor.chunk_size = chunk_size
        st.session_state.search_system.pdf_processor.chunk_overlap = chunk_overlap
        
        with st.spinner("🔄 Running complete analysis pipeline..."):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Processing PDF document...")
            progress_bar.progress(25)
            
            # Run complete analysis
            results = st.session_state.search_system.run_complete_analysis(
                uploaded_file=uploaded_file,
                similarity_method=similarity_method,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
                threshold=similarity_threshold,
                enable_workflow=enable_workflow
            )
            
            status_text.text("🔍 Performing similarity search...")
            progress_bar.progress(50)
            
            if enable_workflow:
                status_text.text("🤖 Running LangGraph analysis...")
                progress_bar.progress(75)
            
            status_text.text("📊 Generating results...")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if results["summary"]["processing_success"]:
                st.success("✅ Analysis completed successfully!")
                
                # Summary statistics
                st.header("📊 Summary Statistics")
                summary = results["summary"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Total Chunks", summary['total_chunks'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Languages Found", summary['found_matches'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    workflow_status = "✅ Enabled" if enable_workflow else "❌ Disabled"
                    st.metric("LangGraph", workflow_status)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed results
                st.header("🔍 Detailed Results")
                
                search_results = results["ensemble_search"]["results"]
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_found = st.checkbox("Show Found Matches", value=True)
                with col2:
                    show_not_found = st.checkbox("Show Not Found", value=True)
                
                # Display results
                displayed_results = 0
                for result in search_results:
                    show_this = (
                        (result["status"] == "Language found" and show_found) or
                        (result["status"] == "Not found" and show_not_found)
                    )
                    
                    if show_this:
                        displayed_results += 1
                        
                        if result["status"] == "Language found":
                            st.markdown(f'<div class="success-result">', unsafe_allow_html=True)
                            st.write(f"**✅ Chunk {result['chunk_index'] + 1}: {result['status']}**")
                            
                            best_match = result["best_match"]
                            st.write(f"**🎯 Matched Language:** {best_match['term']}")
                            st.write(f"**📊 Combined Score:** {best_match['combined_score']:.3f}")
                            st.write(f"**🔗 Similarity Score:** {best_match['similarity_score']:.3f}")
                            st.write(f"**📈 Ensemble Score:** {best_match['ensemble_score']:.3f}")
                            st.write(f"**🔍 Sources:** {', '.join(best_match['sources'])}")
                            st.write(f"**💡 Rationale:** {best_match.get('rationale', 'No rationale provided')}")
                            
                        else:
                            st.markdown(f'<div class="fail-result">', unsafe_allow_html=True)
                            st.write(f"**❌ Chunk {result['chunk_index'] + 1}: {result['status']}**")
                            st.write(f"**🔍 Candidates Found:** {result.get('candidates_found', 0)}")
                        
                        st.write(f"**📝 Text Preview:** {result['chunk_preview']}")
                        
                        # Show all matches if available
                        if result.get("final_matches") and len(result["final_matches"]) > 1:
                            with st.expander("View all candidate matches"):
                                for i, match in enumerate(result["final_matches"]):
                                    st.write(f"{i+1}. **{match['term']}** - Combined: {match['combined_score']:.3f}, Similarity: {match['similarity_score']:.3f}")
                                    st.write(f"   💡 {match.get('rationale', 'No rationale provided')}")
                                    st.write("---")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")
                
                if displayed_results == 0:
                    st.info("No results match the current filter settings.")
                
                # LangGraph Analysis Results
                if enable_workflow and results["workflow_analysis"]["success"]:
                    st.header("🤖 LangGraph AI Analysis")
                    
                    workflow_results = results["workflow_analysis"]
                    
                    # Show enhanced chunks analysis
                    if workflow_results.get("enhanced_chunks"):
                        st.subheader("📋 Chunk Analysis")
                        
                        enhanced_chunks = workflow_results["enhanced_chunks"]
                        for chunk in enhanced_chunks[:3]:  # Show first 3
                            with st.expander(f"Chunk {chunk['index'] + 1} Analysis"):
                                st.write("**Text Preview:**", chunk["preview"])
                                st.write("**Word Count:**", chunk["word_count"])
                                
                                if "analysis" in chunk and isinstance(chunk["analysis"], dict):
                                    analysis = chunk["analysis"]
                                    if "technical_terms" in analysis:
                                        st.write("**Technical Terms:**", ", ".join(analysis["technical_terms"][:5]))
                                    if "confidence_score" in analysis:
                                        st.write("**AI/ML Relevance:**", f"{analysis['confidence_score']:.2f}")
                    
                    # Show AI report
                    if workflow_results.get("report"):
                        st.subheader("📋 AI Analysis Report")
                        st.markdown(workflow_results["report"])
                
                # Export section
                st.header("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export as JSON
                    json_data = st.session_state.search_system.export_results(results, "json")
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"langchain_analysis_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Export as Markdown
                    markdown_data = st.session_state.search_system.export_results(results, "markdown")
                    st.download_button(
                        label="📄 Download Report",
                        data=markdown_data,
                        file_name=f"langchain_report_{uploaded_file.name}.md",
                        mime="text/markdown"
                    )
            
            else:
                st.error("❌ Failed to process the document. Please check the file and try again.")
                if "error" in results["pdf_processing"]:
                    st.error(f"Error: {results['pdf_processing']['error']}")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a PDF file to get started!")
    
    st.header("📖 How it Works")
    st.markdown("""
    This smart ensemble retrieval system combines multiple AI technologies for optimal results:
    
    **🔧 Core Technologies:**
    - **LangChain**: Professional document processing and text splitting
    - **Ensemble Retrieval**: LLM + BM25 + TF-IDF for candidate selection
    - **Multiple Embeddings**: OpenAI, SentenceTransformers, or TF-IDF for similarity
    - **LangGraph**: Multi-agent workflow for enhanced analysis
    
    **⚡ Smart Process Flow:**
    1. **Document Processing**: LangChain loads and splits PDF into optimized chunks
    2. **Ensemble Retrieval**: Multiple methods find top candidate language terms
        - **LLM Retrieval**: GPT-4 identifies semantically relevant terms
        - **BM25**: Keyword-based statistical retrieval
        - **TF-IDF**: Statistical text similarity
    3. **Similarity Ranking**: Compute final similarities using chosen embedding method
    4. **AI Analysis**: Optional LangGraph workflow for deep insights
    5. **Smart Results**: Combined ensemble + similarity scores
    
    **🎯 Key Advantages:**
    - **No Vector Database Required**: More efficient and reliable
    - **Ensemble Approach**: Combines strengths of multiple retrieval methods
    - **Flexible Similarity**: Choose your preferred embedding method
    - **Smart Candidate Selection**: LLM + statistical methods
    - **Real-time Progress**: Live updates during processing
    """)
    
    # Show sample languages
    if os.path.exists("./data/languages.csv"):
        st.subheader("🗃️ Language Database Preview")
        df = pd.read_csv("./data/languages.csv")
        st.dataframe(df.head(20), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    Built with LangChain, FAISS, LangGraph, and Streamlit • 
    <a href='https://github.com/langchain-ai/langchain' target='_blank'>LangChain</a> • 
    <a href='https://github.com/langchain-ai/langgraph' target='_blank'>LangGraph</a>
</div>
""", unsafe_allow_html=True)
