import streamlit as st
import os
import tempfile
import torch
import pandas as pd
import base64
from io import BytesIO
from streamlit_agraph import agraph, Node, Edge, Config
from fpdf import FPDF  # <--- NEW IMPORT

# Import your modules
from keyframe_extractor import SemanticKeyframeExtractor
from captioning import VideoCaptioner
from rag_engine import RAGChatbot

# --- PAGE SETUP ---
st.set_page_config(page_title="SV-RAG Research", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div.stButton > button {
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        color: white; border: none; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CONVERT IMAGE TO BASE64 FOR DATAFRAME ---
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# --- NEW: PDF GENERATOR CLASS ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'SV-RAG: Video Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(data_list):
    """Generates a PDF from the analysis data."""
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for item in data_list:
        # Time and Confidence
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Timestamp: {item['Timestamp']} | Confidence: {item['RawConfidence']:.2%}", 0, 1)
        
        # Insight Text
        pdf.set_font("Arial", size=11)
        # Multi_cell handles text wrapping
        pdf.multi_cell(0, 10, f"Insight: {item['AI Insight']}")
        
        # Image (Thumbnail)
        # We use the physical path stored in 'ThumbnailPath'
        if 'ThumbnailPath' in item and os.path.exists(item['ThumbnailPath']):
            # x=10 (left margin), w=100 (width)
            pdf.image(item['ThumbnailPath'], x=10, w=100)
            pdf.ln(60) # Move down 60 units after image to prevent overlap
        else:
            pdf.ln(5)
            
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Draw separator line
        pdf.ln(10)
        
    return pdf.output(dest='S').encode('latin-1')

# --- STATE ---
if "rag_bot" not in st.session_state: st.session_state.rag_bot = None
if "messages" not in st.session_state: st.session_state.messages = []
if "graph_data" not in st.session_state: st.session_state.graph_data = {"nodes": [], "edges": []}
if "df_insights" not in st.session_state: st.session_state.df_insights = None
if "table_data" not in st.session_state: st.session_state.table_data = [] # <--- Stores data for PDF

# --- SIDEBAR ---
with st.sidebar:
    st.title("SV-RAG System")
    st.caption("Adaptive Video Question Answering")
    st.divider()
    
    st.subheader("🔬 Clustering Strategy")
    use_adaptive = st.toggle("Use Adaptive K (Silhouette)", value=True)
    
    n_clusters = 10
    if not use_adaptive:
        n_clusters = st.slider("Manual Clusters (K)", 5, 30, 10)

    sample_rate = st.slider("Sampling Rate (FPS)", 0.5, 3.0, 1.0)
    st.divider()
    st.write(f"Device: {'MPS (Mac)' if torch.backends.mps.is_available() else 'CPU'}")

# --- MAIN ---
st.markdown("# 🔬 SV-RAG: Semantic Video Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Analysis & Data", "🕸️ Interactive Graph", "💬 Chat"])

with tab1:
    uploaded_file = st.file_uploader("Upload Video (MP4)", type=["mp4"])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        col_vid, col_btn = st.columns([2, 1])
        with col_vid:
            st.video(video_path)
        with col_btn:
            if st.button("🚀 Initialize Research Pipeline", use_container_width=True):
                with st.status("Running Adaptive Pipeline...", expanded=True) as status:
                    
                    # 1. Extract
                    st.write("🔍 Extracting frames...")
                    extractor = SemanticKeyframeExtractor()
                    raw_frames, timestamps = extractor.extract_frames(video_path, sample_rate)
                    
                    # 2. Cluster
                    st.write("🧠 Clustering Scenes...")
                    keyframes, key_timestamps = extractor.cluster_and_select(
                        raw_frames, timestamps, n_clusters=n_clusters, use_adaptive=use_adaptive
                    )
                    
                    # 3. Caption & Build Table Data
                    st.write("👁️ Analyzing Keyframes (BLIP)...")
                    captioner = VideoCaptioner()
                    
                    insights = []
                    nodes = []
                    edges = []
                    existing_ids = set()
                    table_data = [] 
                    
                    # Create a temp directory for PDF images
                    # We need physical files for the PDF generator, it can't use RAM bytes
                    thumb_dir = tempfile.mkdtemp()
                    
                    progress = st.progress(0)
                    for i, (frame, ts) in enumerate(zip(keyframes, key_timestamps)):
                        caption, conf = captioner.generate_caption(frame)
                        time_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
                        
                        # A. Save Physical Image for PDF
                        img_path = os.path.join(thumb_dir, f"frame_{i}.jpg")
                        frame.save(img_path)
                        
                        # B. Convert to Base64 for Screen Display
                        img_base64 = image_to_base64(frame)
                        
                        # C. Data Collection
                        insights.append(f"[{time_str}] {caption}")
                        
                        # We store everything we need for BOTH the Table and the PDF here
                        table_data.append({
                            "Keyframe": img_base64,      # For Streamlit Table
                            "ThumbnailPath": img_path,   # For PDF Report
                            "Timestamp": time_str,
                            "AI Insight": caption,
                            "RawConfidence": conf,       # Number (0.85)
                            "Confidence": conf           # Duplicate for DataFrame display
                        })
                        
                        # D. Graph Logic
                        node_id_time = f"t_{i}"
                        if node_id_time not in existing_ids:
                            nodes.append(Node(id=node_id_time, label=time_str, size=15, color="#7c3aed", shape="ellipse"))
                            existing_ids.add(node_id_time)
                            
                        node_id_concept = f"c_{caption[:15]}"
                        if node_id_concept not in existing_ids:
                            nodes.append(Node(id=node_id_concept, label=caption[:20]+"...", title=caption, size=12, color="#2563eb", shape="box"))
                            existing_ids.add(node_id_concept)
                            
                        edges.append(Edge(source=node_id_time, target=node_id_concept))
                        progress.progress((i+1)/len(keyframes))
                    
                    # Store Results in Session State
                    st.session_state.graph_data = {"nodes": nodes, "edges": edges}
                    st.session_state.df_insights = pd.DataFrame(table_data)
                    
                    # --- THIS IS THE LINE YOU ASKED ABOUT ---
                    # We save the full list (with paths) so the PDF button can read it later
                    st.session_state.table_data = table_data 
                    
                    # 4. RAG
                    st.write("📚 Indexing Knowledge...")
                    rag = RAGChatbot()
                    rag.ingest_insights(insights)
                    st.session_state.rag_bot = rag
                    
                    status.update(label="Complete!", state="complete", expanded=False)

    # --- RESULTS DISPLAY ---
    if st.session_state.df_insights is not None:
        st.divider()
        st.subheader("📋 Structural Video Analysis Data")
        
        # 1. The Interactive Table
        st.dataframe(
            st.session_state.df_insights,
            column_config={
                "Keyframe": st.column_config.ImageColumn("Keyframe", width="small"),
                "Timestamp": st.column_config.TextColumn("Time", width="small"),
                "AI Insight": st.column_config.TextColumn("Visual Conclusion", width="large"),
                "Confidence": st.column_config.ProgressColumn(
                    "Model Confidence", 
                    min_value=0, 
                    max_value=1, 
                    format="%.2f%%"
                ),
                # Hide the internal path column from the UI
                "ThumbnailPath": None,
                "RawConfidence": None
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 2. The PDF Download Section (Feature 2)
        st.divider()
        st.subheader("📄 Export Research Report")
        
        # Check if we have data to print
        if "table_data" in st.session_state and st.session_state.table_data:
            col_pdf, col_space = st.columns([1, 2])
            with col_pdf:
                # Generate PDF Bytes on the fly
                pdf_bytes = create_pdf(st.session_state.table_data)
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="SV-RAG_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

with tab2:
    st.subheader("🕸️ Interactive Knowledge Graph")
    col_graph, col_details = st.columns([3, 1])
    
    selected_node = None 
    
    with col_graph:
        if st.session_state.graph_data["nodes"]:
            config = Config(
                width=800, height=600, 
                directed=True, physics=True, 
                hierarchical=False, 
                nodeSpacing=200, 
                solver='forceAtlas2Based'
            )
            selected_node = agraph(nodes=st.session_state.graph_data["nodes"], 
                                 edges=st.session_state.graph_data["edges"], 
                                 config=config)
        else:
            st.info("No graph data available. Please run the analysis first.")

    with col_details:
        st.markdown("### 🔎 Node Details")
        if selected_node:
            st.success(f"Selected: **{selected_node}**")
            st.write("This node represents a distinct semantic concept or timestamp found in the video.")
        else:
            st.write("Click on a node in the graph to see details here.")

with tab3:
    st.subheader("💬 Q&A Interface")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        if st.session_state.rag_bot:
            with st.spinner("Thinking..."):
                ans, ctx = st.session_state.rag_bot.ask(prompt)
            final_ans = f"{ans}\n\n_Context: {ctx}_"
            st.session_state.messages.append({"role": "assistant", "content": final_ans})
            st.chat_message("assistant").write(final_ans)