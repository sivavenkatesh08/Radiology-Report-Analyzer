"""
🏥 AI Radiology Assistant — Streamlit Application
===================================================
Production-level AI healthcare assistant with 3 input modes:
  1. Text Report Analysis
  2. Report Upload (Image/PDF via OCR)
  3. X-ray / MRI Scan Analysis
"""

import streamlit as st
import tempfile
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.summarizer import summarize_report
from src.nlp.ocr import extract_text_from_image, extract_text_from_pdf
from src.vision.image_processor import highlight_region
from src.vision.unet_model import predict_segmentation
from src.utils.report_generator import generate_report
from src.agents.radiology_agent import analyze as agent_analyze
from src.rag.vector_store import is_available as rag_is_available

# ════════════════════════════════════════════════════════════════
# Page Configuration
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Radiology Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════
# Premium CSS
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 40%, #1a1040 100%);
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

.hero-container {
    text-align: center; padding: 40px 20px 30px; margin-bottom: 30px;
}
.hero-title {
    font-size: 42px; font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px; letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 16px; color: #94a3b8; font-weight: 400; letter-spacing: 0.5px;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.8); border-radius: 16px;
    padding: 6px; gap: 6px;
    border: 1px solid rgba(99, 102, 241, 0.15);
    backdrop-filter: blur(20px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px; color: #94a3b8; font-weight: 600;
    font-size: 15px; padding: 12px 24px; transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25)) !important;
    color: #e2e8f0 !important; border: 1px solid rgba(99,102,241,0.4);
}

.result-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(99,102,241,0.2); border-radius: 16px;
    padding: 28px; margin: 16px 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, border-color 0.3s ease;
}
.result-card:hover {
    border-color: rgba(99,102,241,0.4); transform: translateY(-2px);
}
.card-title {
    font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px;
    color: #a5b4fc; font-weight: 700; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.card-content {
    color: #e2e8f0; font-size: 15px; line-height: 1.8; white-space: pre-line;
}

.severity-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.4); color: #fca5a5;
    padding: 16px 28px; border-radius: 12px; text-align: center;
    font-size: 22px; font-weight: 700;
}
.severity-medium {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.1));
    border: 1px solid rgba(245,158,11,0.4); color: #fcd34d;
    padding: 16px 28px; border-radius: 12px; text-align: center;
    font-size: 22px; font-weight: 700;
}
.severity-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7;
    padding: 16px 28px; border-radius: 12px; text-align: center;
    font-size: 22px; font-weight: 700;
}
.severity-unknown {
    background: rgba(107,114,128,0.15);
    border: 1px solid rgba(107,114,128,0.4); color: #d1d5db;
    padding: 16px 28px; border-radius: 12px; text-align: center;
    font-size: 22px; font-weight: 700;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 12px 32px !important;
    font-weight: 600 !important; font-size: 15px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 12px 32px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.3) !important;
}

.stTextArea textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; padding: 16px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

.stFileUploader {
    background: rgba(15,23,42,0.5);
    border: 2px dashed rgba(99,102,241,0.3);
    border-radius: 16px; padding: 20px;
}

.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px; padding: 16px 20px;
    color: #a5b4fc; font-size: 14px; margin: 12px 0;
}
.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px; padding: 16px 20px;
    color: #fbbf24; font-size: 14px; margin: 12px 0;
}

.stSpinner > div { border-top-color: #818cf8 !important; }

.stRadio > div {
    background: rgba(15,23,42,0.6); border-radius: 12px;
    padding: 12px; border: 1px solid rgba(99,102,241,0.15);
}

.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 24px 0;
}

.findings-table {
    width: 100%; border-collapse: collapse; margin-top: 8px;
}
.findings-table th {
    padding: 10px 16px; text-align: left; color: #a5b4fc;
    font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}
.findings-table td {
    padding: 10px 16px; color: #e2e8f0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}
.badge-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-medium { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-low { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

SAMPLE_REPORT = """CHEST X-RAY PA VIEW

Clinical History: Persistent cough and mild fever for 5 days.

Findings:
There is a mild opacity observed in the right lower lung zone, suggesting an early infectious process. No large pleural effusion detected. The cardiac silhouette is within normal limits. Mediastinal contours are unremarkable. Bony thorax appears intact with no acute fracture.

Impression:
1. Right lower lobe opacity — likely early pneumonia.
2. No pleural effusion.
3. Normal heart size.
"""


def get_severity_class(severity_text):
    """Map severity text to CSS class name."""
    if "High" in severity_text:
        return "severity-high"
    elif "Medium" in severity_text:
        return "severity-medium"
    elif "Low" in severity_text:
        return "severity-low"
    return "severity-unknown"


def render_findings_table(key_findings):
    """Render key findings as a styled HTML table."""
    if not key_findings:
        return
    rows = ""
    for f in key_findings:
        label = f.get("severity_label", "")
        if "High" in label:
            badge = "badge-high"
        elif "Medium" in label:
            badge = "badge-medium"
        else:
            badge = "badge-low"
        rows += f"""<tr>
            <td>{f.get('finding', 'N/A')}</td>
            <td style="text-align:center;"><span class="{badge}">{label}</span></td>
        </tr>"""

    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">🔬 Key Findings</div>
        <table class="findings-table">
            <thead><tr><th>Finding</th><th style="text-align:center;">Severity</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_rag_context(rag_context, rag_available):
    """Render RAG retrieval context as a styled card."""
    if not rag_available:
        st.markdown("""
        <div class="info-box">
            ℹ️ RAG knowledge retrieval is not available (FAISS or embedder not loaded).
            Analysis is based on rule-based interpretation only.
        </div>
        """, unsafe_allow_html=True)
        return

    if not rag_context:
        return

    rows = ""
    for ctx in rag_context:
        rows += f"""<tr>
            <td>{ctx.get('topic', 'N/A')}</td>
            <td style="text-align:center;">{ctx.get('relevance', '—')}</td>
            <td style="text-align:center;">{ctx.get('severity', '—')}</td>
        </tr>"""

    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">📚 RAG Knowledge Context</div>
        <table class="findings-table">
            <thead><tr>
                <th>Retrieved Topic</th>
                <th style="text-align:center;">Relevance</th>
                <th style="text-align:center;">Severity Guidance</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_results(summary, analysis, mode="text"):
    """Render analysis results with premium styling."""
    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">🧠 Clinical Summary</div>
        <div class="card-content">{summary}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">📋 Clinical Interpretation</div>
        <div class="card-content">{analysis['explanation']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Findings table
    render_findings_table(analysis.get("key_findings", []))

    # RAG Context
    render_rag_context(analysis.get("rag_context", []), analysis.get("rag_available", False))

    # Severity badge
    sev_class = get_severity_class(analysis["severity"])
    st.markdown(f"""
    <div class="{sev_class}" style="margin: 16px 0;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:2px;opacity:0.7;margin-bottom:4px;">
            Severity Level
        </div>
        {analysis['severity']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">💡 Recommendation</div>
        <div class="card-content">{analysis['recommendation']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Download
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    report_html = generate_report(
        summary=summary,
        severity=analysis["severity"],
        recommendation=analysis["recommendation"],
        explanation=analysis["explanation"],
        mode=mode,
        key_findings=analysis.get("key_findings", []),
    )
    st.download_button(
        label="📥 Download Full Report (HTML)",
        data=report_html,
        file_name="radiology_report.html",
        mime="text/html",
    )


# ════════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏥 AI Radiology Assistant</div>
    <div class="hero-subtitle">
        Production-Level AI Healthcare Analysis · NLP · Computer Vision · Deep Learning
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📄 Text Report",
    "🧾 Report Upload",
    "🩻 X-ray / MRI"
])


# ─────────────────────────────────────────
# TAB 1: Text Report
# ─────────────────────────────────────────

with tab1:
    st.markdown("""
    <div class="info-box">
        📝 Paste a radiology report below for AI-powered clinical analysis.
        The system uses transformer-based summarization, RAG knowledge retrieval, and agent-based reasoning.
    </div>
    """, unsafe_allow_html=True)

    col_input, col_sample = st.columns([4, 1])
    with col_sample:
        if st.button("📋 Load Sample", key="load_sample"):
            st.session_state["sample_text"] = SAMPLE_REPORT

    default_text = st.session_state.get("sample_text", "")

    text_input = st.text_area(
        "Enter Radiology Report",
        value=default_text,
        height=200,
        placeholder="Paste radiology report text here...",
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyze Report", key="analyze_text"):
        if text_input.strip():
            with st.spinner("Running RAG + Agent analysis..."):
                summary = summarize_report(text_input)
                analysis = agent_analyze(text_input)
            render_results(summary, analysis, mode="text")
        else:
            st.markdown("""
            <div class="warning-box">⚠️ Please enter a radiology report to analyze.</div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# TAB 2: Report Upload (Image/PDF)
# ─────────────────────────────────────────

with tab2:
    st.markdown("""
    <div class="info-box">
        📎 Upload a scanned radiology report (image or PDF).
        The system uses OCR with preprocessing to extract text, then analyzes it.
        Raw OCR text is not displayed — only clean medical outputs are shown.
    </div>
    """, unsafe_allow_html=True)

    uploaded_report = st.file_uploader(
        "Upload Report (Image or PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        key="report_upload",
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyze Uploaded Report", key="analyze_report"):
        if uploaded_report:
            with st.spinner("Extracting text via OCR and analyzing..."):
                suffix = ".pdf" if uploaded_report.type == "application/pdf" else ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_report.read())
                    temp_path = tmp.name

                if uploaded_report.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(temp_path)
                else:
                    extracted_text = extract_text_from_image(temp_path)

                if extracted_text.startswith("[OCR Error]") or extracted_text.startswith("[OCR Warning]"):
                    st.markdown(f"""
                    <div class="warning-box">⚠️ {extracted_text}</div>
                    """, unsafe_allow_html=True)
                else:
                    summary = summarize_report(extracted_text)
                    analysis = agent_analyze(extracted_text)
                    render_results(summary, analysis, mode="report_upload")

                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        else:
            st.markdown("""
            <div class="warning-box">⚠️ Please upload a report image or PDF first.</div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Scan Analysis Helpers
# ─────────────────────────────────────────

def _run_prototype_scan(temp_path):
    """Run OpenCV-based prototype scan analysis."""
    with st.spinner("Running OpenCV contour analysis..."):
        try:
            import cv2
            import pandas as pd

            annotated, region_count, regions = highlight_region(temp_path)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="card-title" style="margin-bottom:8px;">Original Scan</div>
                """, unsafe_allow_html=True)
                original = cv2.imread(temp_path)
                original = cv2.resize(original, (512, 512))
                st.image(original, channels="BGR", use_container_width=True)

            with col2:
                st.markdown("""
                <div class="card-title" style="margin-bottom:8px;">Annotated Analysis</div>
                """, unsafe_allow_html=True)
                st.image(annotated, channels="BGR", use_container_width=True)

            if region_count > 0:
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-title">📊 Detection Summary</div>
                    <div class="card-content">
                        Detected <strong>{region_count}</strong> suspicious region(s) using contour analysis.
                        Highlighted areas indicate potential anomalies that warrant further review.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if regions:
                    st.markdown("""
                    <div class="result-card">
                        <div class="card-title">📍 Region Details</div>
                    </div>
                    """, unsafe_allow_html=True)
                    df = pd.DataFrame(regions)
                    df.columns = ["Region", "Area (px²)", "Position", "Size"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                <div class="result-card">
                    <div class="card-title">✅ Detection Summary</div>
                    <div class="card-content">
                        No significant suspicious regions detected by the prototype analyzer.
                        Consider using the Advanced (U-Net) mode for more precise analysis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-box">
                ℹ️ <strong>Note:</strong> This prototype uses OpenCV contour detection which may produce
                false positives. For accurate clinical detection, deep learning models like
                <strong>U-Net</strong> trained on NIH Chest X-ray or CheXpert datasets should be used.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">⚠️ Scan analysis failed: {str(e)}</div>
            """, unsafe_allow_html=True)


def _run_advanced_scan(temp_path):
    """Run U-Net deep learning segmentation analysis with fallback."""
    with st.spinner("Running U-Net deep learning segmentation..."):
        result = predict_segmentation(temp_path)

        if result["available"]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="card-title">Segmentation Mask</div>
                """, unsafe_allow_html=True)
                st.image(result["mask"], use_container_width=True)
            with col2:
                st.markdown("""
                <div class="card-title">Overlay on Original</div>
                """, unsafe_allow_html=True)
                st.image(result["overlay"], channels="BGR", use_container_width=True)

            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">✅ U-Net Analysis</div>
                <div class="card-content">{result['message']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-title">🧬 Advanced Mode — U-Net Segmentation</div>
                <div class="card-content">{result['message']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <strong>How to enable Advanced Mode:</strong><br>
                1. Download NIH Chest X-ray or CheXpert dataset<br>
                2. Prepare binary segmentation masks<br>
                3. Train the U-Net model (architecture already implemented in <code>src/vision/unet_model.py</code>)<br>
                4. Save weights and update <code>MODEL_WEIGHTS_PATH</code><br>
                5. Re-run the analysis
            </div>
            """, unsafe_allow_html=True)

            # Fallback to prototype
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">🔄 Fallback: Prototype Analysis (OpenCV)</div>
            """, unsafe_allow_html=True)

            try:
                annotated, region_count, _ = highlight_region(temp_path)
                st.image(annotated, channels="BGR", use_container_width=True)
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-content">
                        OpenCV prototype detected <strong>{region_count}</strong> suspicious region(s).
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">Fallback analysis also failed: {str(e)}</div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# TAB 3: X-ray / MRI Scan
# ─────────────────────────────────────────

with tab3:
    st.markdown("""
    <div class="info-box">
        🩻 Upload an X-ray or MRI scan for automated anomaly detection.
        Choose between prototype (OpenCV) and advanced (U-Net deep learning) modes.
    </div>
    """, unsafe_allow_html=True)

    scan_mode = st.radio(
        "Analysis Mode",
        ["🔬 Prototype (OpenCV)", "🧬 Advanced (U-Net Deep Learning)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_scan = st.file_uploader(
        "Upload X-ray / MRI Scan",
        type=["png", "jpg", "jpeg"],
        key="scan_upload",
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyze Scan", key="analyze_scan"):
        if uploaded_scan:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_scan.read())
                temp_path = tmp.name

            if "Prototype" in scan_mode:
                _run_prototype_scan(temp_path)
            else:
                _run_advanced_scan(temp_path)

            try:
                os.unlink(temp_path)
            except OSError:
                pass
        else:
            st.markdown("""
            <div class="warning-box">⚠️ Please upload an X-ray or MRI scan first.</div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding:40px 20px 20px; color:#475569; font-size:13px;">
    <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent); margin-bottom:20px;"></div>
    <p>AI Radiology Assistant · Built with Transformers, OpenCV & PyTorch</p>
    <p style="color:#64748b; font-size:12px; margin-top:4px;">
        ⚠️ For informational purposes only. Not a substitute for professional medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)
