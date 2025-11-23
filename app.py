import streamlit as st
import os
import tempfile
import torch
import pandas as pd
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_agraph import agraph, Node, Edge, Config
from fpdf import FPDF

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
    .metric-card {
        background-color: #1f2937; padding: 15px; border-radius: 10px;
        text-align: center; border: 1px solid #374151;
    }
    .metric-val { font-size: 24px; font-weight: bold; color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CONVERT IMAGE TO BASE64 FOR DATAFRAME ---
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# --- PDF GENERATOR CLASS ---
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
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Timestamp: {item['Timestamp']} | Confidence: {item['RawConfidence']:.2%}", 0, 1)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, f"Insight: {item['AI Insight']}")
        
        if 'ThumbnailPath' in item and os.path.exists(item['ThumbnailPath']):
            pdf.image(item['ThumbnailPath'], x=10, w=100)
            pdf.ln(60) 
        else:
            pdf.ln(5)
            
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
    return pdf.output(dest='S').encode('latin-1')

# --- STATE MANAGEMENT ---
if "rag_bot" not in st.session_state: st.session_state.rag_bot = None
if "messages" not in st.session_state: st.session_state.messages = []
if "graph_data" not in st.session_state: st.session_state.graph_data = {"nodes": [], "edges": []}
if "df_insights" not in st.session_state: st.session_state.df_insights = None
if "table_data" not in st.session_state: st.session_state.table_data = []

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

# Added the 4th Tab here
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis & Data", "🕸️ Interactive Graph", "💬 Chat", "🕵️ Visual Detective"])

# --- TAB 1: ANALYSIS ---
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
                    
                    st.write("🔍 Extracting frames...")
                    extractor = SemanticKeyframeExtractor()
                    raw_frames, timestamps = extractor.extract_frames(video_path, sample_rate)
                    
                    st.write("🧠 Clustering Scenes...")
                    keyframes, key_timestamps = extractor.cluster_and_select(
                        raw_frames, timestamps, n_clusters=n_clusters, use_adaptive=use_adaptive
                    )
                    
                    st.write("👁️ Analyzing Keyframes (BLIP)...")
                    captioner = VideoCaptioner()
                    
                    insights = []
                    nodes = []
                    edges = []
                    existing_ids = set()
                    table_data = [] 
                    
                    # Temp dir for PDF/Search images
                    thumb_dir = tempfile.mkdtemp()
                    
                    progress = st.progress(0)
                    for i, (frame, ts) in enumerate(zip(keyframes, key_timestamps)):
                        caption, conf = captioner.generate_caption(frame)
                        time_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
                        
                        img_path = os.path.join(thumb_dir, f"frame_{i}.jpg")
                        frame.save(img_path)
                        img_base64 = image_to_base64(frame)
                        
                        insights.append(f"[{time_str}] {caption}")
                        
                        table_data.append({
                            "Keyframe": img_base64,
                            "ThumbnailPath": img_path,
                            "Timestamp": time_str,
                            "AI Insight": caption,
                            "RawConfidence": conf,
                            "Confidence": conf
                        })
                        
                        # Graph Logic
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
                    
                    st.session_state.graph_data = {"nodes": nodes, "edges": edges}
                    st.session_state.df_insights = pd.DataFrame(table_data)
                    st.session_state.table_data = table_data # Saving for Tab 4 Search
                    
                    st.write("📚 Indexing Knowledge...")
                    rag = RAGChatbot()
                    rag.ingest_insights(insights)
                    st.session_state.rag_bot = rag
                    
                    status.update(label="Complete!", state="complete", expanded=False)

    if st.session_state.df_insights is not None:
        st.divider()
        st.subheader("📋 Structural Video Analysis Data")
        
        st.dataframe(
            st.session_state.df_insights,
            column_config={
                "Keyframe": st.column_config.ImageColumn("Keyframe", width="small"),
                "Timestamp": st.column_config.TextColumn("Time", width="small"),
                "AI Insight": st.column_config.TextColumn("Visual Conclusion", width="large"),
                "Confidence": st.column_config.ProgressColumn(
                    "Model Confidence", min_value=0, max_value=1, format="%.2f%%"
                ),
                "ThumbnailPath": None,
                "RawConfidence": None
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        if "table_data" in st.session_state and st.session_state.table_data:
            col_pdf, col_space = st.columns([1, 2])
            with col_pdf:
                pdf_bytes = create_pdf(st.session_state.table_data)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="SV-RAG_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# --- TAB 2: GRAPH ---
with tab2:
    st.subheader("🕸️ Interactive Knowledge Graph")
    col_graph, col_details = st.columns([3, 1])
    selected_node = None 
    
    with col_graph:
        if st.session_state.graph_data["nodes"]:
            config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False, nodeSpacing=200, solver='forceAtlas2Based')
            selected_node = agraph(nodes=st.session_state.graph_data["nodes"], edges=st.session_state.graph_data["edges"], config=config)
        else:
            st.info("No graph data available. Please run the analysis first.")

    with col_details:
        st.markdown("### 🔎 Node Details")
        if selected_node:
            st.success(f"Selected: **{selected_node}**")
            st.write("This node represents a distinct semantic concept or timestamp.")
        else:
            st.write("Click on a node to see details.")

# --- TAB 3: CHAT ---
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

# --- TAB 4: VISUAL DETECTIVE (NEW) ---
with tab4:
    st.subheader("🕵️ Visual Detective (Zero-Shot Search)")
    st.markdown("Upload an image of an object (e.g., a backpack, a car) to find similar frames in the video.")
    
    if "table_data" not in st.session_state or not st.session_state.table_data:
        st.warning("⚠️ Please run the Video Analysis in Tab 1 first to extract keyframes.")
    else:
        search_file = st.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])
        
        if search_file:
            col_q, col_res = st.columns([1, 2])
            
            with col_q:
                st.write("**Query Image**")
                query_image = Image.open(search_file)
                st.image(query_image, width=200)
                
            with col_res:
                if st.button("🔍 Find Matches"):
                    with st.spinner("Embedding query and comparing vectors..."):
                        # 1. Load Keyframes from disk (using paths saved in Tab 1)
                        # This prevents storing massive image arrays in RAM
                        keyframes = []
                        valid_indices = []
                        for idx, item in enumerate(st.session_state.table_data):
                            if os.path.exists(item['ThumbnailPath']):
                                keyframes.append(Image.open(item['ThumbnailPath']))
                                valid_indices.append(idx)
                        
                        # 2. Initialize Extractor (Loads CLIP)
                        # We re-init here. In a prod app, we'd cache the model load.
                        extractor = SemanticKeyframeExtractor()
                        
                        # 3. Compute Embeddings
                        # Shape: (1, 512)
                        query_emb = extractor.get_embeddings([query_image])
                        # Shape: (N, 512)
                        keyframe_embs = extractor.get_embeddings(keyframes)
                        
                        # 4. Cosine Similarity
                        # Since embeddings are normalized, Dot Product = Cosine Sim
                        scores = np.dot(keyframe_embs, query_emb.T).flatten()
                        
                        # 5. Get Top 3 Matches
                        # argsort gives ascending, so we take last 3 and reverse
                        top_indices = np.argsort(scores)[-3:][::-1]
                        
                        st.success("Search Complete! Top Matches:")
                        
                        # 6. Display Results
                        for rank, idx in enumerate(top_indices):
                            real_idx = valid_indices[idx]
                            match_data = st.session_state.table_data[real_idx]
                            score = scores[idx]
                            
                            with st.container():
                                c1, c2 = st.columns([1, 3])
                                with c1:
                                    st.image(match_data['ThumbnailPath'], width=150)
                                with c2:
                                    st.markdown(f"### Match #{rank+1}")
                                    st.markdown(f"**Timestamp:** `{match_data['Timestamp']}`")
                                    st.markdown(f"**Similarity Score:** `{score:.4f}`")
                                    st.caption(f"Context: {match_data['AI Insight']}")
                                st.divider()