"""
app.py — AI Dental Implant Planning System
Streamlit front-end for CBCT DICOM segmentation using OralSeg.

Usage:
    streamlit run app.py
"""

import os
import sys
import site
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Ensure all Python site-packages (including user-installed) are in sys.path
# ─────────────────────────────────────────────────────────────────────────────
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Fallback common user site-packages for Windows Python 3.11
roaming_site = os.path.expandvars(r"%APPDATA%\Python\Python311\site-packages")
if os.path.exists(roaming_site) and roaming_site not in sys.path:
    sys.path.insert(0, roaming_site)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gc
import json
import time
import shutil
import logging
import tempfile
import traceback
from typing import Optional

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent
MODELS_DIR      = ROOT_DIR / "models"
UPLOADS_DIR     = ROOT_DIR / "uploads"
CONVERTED_DIR   = ROOT_DIR / "converted"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
CHECKPOINT_PATH = MODELS_DIR / "model_workstation39.pt"

for d in [UPLOADS_DIR, CONVERTED_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Dental Implant Planning System",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — AI Dental CBCT Segmentation Report (Clean Clinical Theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {
    --bg-page: #F0F4F8;
    --card-bg: #FFFFFF;
    --navy-dark: #0F3B7A;
    --navy-blue: #1E40AF;
    --navy-soft: #2563EB;
    --navy-light: #EFF6FF;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --text-faint: #94A3B8;
    --border-card: #CBD5E1;
    --border-light: #E2E8F0;
    --green-detect: #16A34A;
    --green-bg: #DCFCE7;
    --red-missing: #DC2626;
    --red-bg: #FEE2E2;
    --gray-missing: #E2E8F0;
    --gray-missing-text: #475569;
    --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
    --radius-card: 8px;
  }

  html, body, [class*="css"], .stApp {
    font-family: var(--sans);
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-size: 13.5px;
    line-height: 1.45;
  }

  /* ── Remove dark header overlay ─────────────────────────────── */
  header[data-testid="stHeader"] {
    background: transparent !important;
  }

  .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
  }

  /* ── Typography & Headers ───────────────────────────────────── */
  h1, h2, h3, h4, h5, h6 {
    color: var(--navy-dark) !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
  }

  /* ── Title Card ─────────────────────────────────────────────── */
  .title-card {
    background: #FFFFFF;
    border: 1px solid var(--border-light);
    border-radius: var(--radius-card);
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
  .title-main {
    font-size: 22px;
    font-weight: 800;
    color: var(--navy-dark);
    margin: 0;
    letter-spacing: -0.3px;
    display: inline-block;
  }
  .title-badge {
    background: #EFF6FF;
    color: var(--navy-soft);
    border: 1px solid #BFDBFE;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    vertical-align: middle;
  }
  .title-sub {
    font-size: 12.5px;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
    font-weight: 500;
  }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--border-light) !important;
  }
  [data-testid="stSidebar"] .block-container {
    padding: 20px 16px !important;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }
  .brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: var(--navy-dark);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 5px rgba(15, 59, 122, 0.25);
  }
  .brand-name {
    font-size: 16px;
    font-weight: 800;
    color: var(--navy-dark);
    letter-spacing: -0.2px;
    line-height: 1.1;
  }
  .brand-sub {
    font-size: 10.5px;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .side-label {
    font-size: 11px;
    color: var(--navy-dark);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    margin-top: 16px;
  }

  .side-meta {
    margin-top: 14px;
    background: #F8FAFC;
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .side-meta div {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    color: var(--text-secondary);
    font-weight: 500;
  }
  .side-meta .dot {
    color: var(--navy-soft);
    font-size: 8px;
  }

  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    flex-shrink: 0;
  }
  .status-dot.on {
    background-color: #16A34A;
    box-shadow: 0 0 6px rgba(22, 163, 74, 0.4);
  }
  .status-dot.off {
    background-color: #DC2626;
    box-shadow: 0 0 6px rgba(220, 38, 38, 0.4);
  }

  /* ── File Uploader ────────────────────────────────────────────── */
  [data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #93C5FD !important;
    border-radius: 10px !important;
    padding: 14px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
  }
  [data-testid="stFileUploader"] section {
    background-color: #F8FAFC !important;
    border-radius: 8px !important;
    padding: 16px !important;
    border: none !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC !important;
    border: none !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text-primary) !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-muted) !important;
  }
  [data-testid="stFileUploader"] button {
    background: var(--navy-dark) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(15, 59, 122, 0.2) !important;
  }
  [data-testid="stFileUploader"] button:hover {
    background: var(--navy-blue) !important;
    color: #FFFFFF !important;
  }

  /* ── Empty State Card ──────────────────────────────────────────── */
  .empty-upload-card {
    background: #FFFFFF;
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
  }
  .empty-upload-card .empty-icon {
    font-size: 3rem;
    margin-bottom: 0.6rem;
  }
  .empty-upload-card .empty-title {
    color: var(--navy-dark);
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
  }
  .empty-upload-card .empty-sub {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin: 0;
  }

  /* ── General Card ──────────────────────────────────────────────── */
  .card {
    background: #FFFFFF;
    border: 1px solid var(--border-card);
    border-radius: var(--radius-card);
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }

  /* ── Metrics ───────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #F8FAFC !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
  }
  [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
    color: var(--navy-dark) !important;
    font-weight: 800 !important;
  }

  /* ── Tabs ──────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 2px solid var(--border-light);
    padding-bottom: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF;
    border-radius: 6px 6px 0 0;
    border: 1px solid var(--border-light);
    border-bottom: none;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    background-color: var(--navy-dark) !important;
    color: #FFFFFF !important;
    border-color: var(--navy-dark) !important;
  }

  /* ── Buttons ───────────────────────────────────────────────────── */
  .stButton > button {
    background: var(--navy-dark) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: var(--sans) !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(15, 59, 122, 0.2) !important;
  }
  .stButton > button:hover {
    background: var(--navy-blue) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(15, 59, 122, 0.3) !important;
  }

  /* ── Main Report Header ────────────────────────────────────────── */
  .report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
  }
  .report-title-box {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .report-title-icon {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: var(--navy-dark);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 20px;
  }
  .report-title-text {
    font-size: 21px;
    font-weight: 800;
    color: var(--navy-dark);
    letter-spacing: -0.3px;
    margin: 0;
  }
  .report-brand-right {
    display: flex;
    align-items: center;
    gap: 8px;
    text-align: right;
  }
  .report-brand-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--navy-dark);
    line-height: 1.1;
  }
  .report-brand-sub {
    font-size: 10.5px;
    color: var(--text-muted);
    font-weight: 500;
  }

  /* ── Metadata Info Bar (6 Badges) ──────────────────────────────── */
  .meta-bar {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-bottom: 14px;
  }
  .meta-card {
    background: #FFFFFF;
    border: 1px solid var(--border-card);
    border-radius: 6px;
    padding: 7px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .meta-icon {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: #EFF6FF;
    color: var(--navy-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .meta-info {
    overflow: hidden;
  }
  .meta-label {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    line-height: 1;
    margin-bottom: 2px;
  }
  .meta-val {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2;
  }

  /* ── Clinical Report Cards ─────────────────────────────────────── */
  .card-report {
    background: #FFFFFF;
    border: 1px solid var(--border-card);
    border-radius: var(--radius-card);
    padding: 12px 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
    box-sizing: border-box;
  }
  .card-head {
    font-size: 11.5px;
    font-weight: 800;
    color: var(--navy-dark);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* ── FDI Tooth Chart ──────────────────────────────────────────── */
  .jaw-label {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--navy-dark);
    margin-bottom: 4px;
  }
  .arch-grid-16 {
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    gap: 3px;
    margin-bottom: 8px;
  }
  .tooth-box {
    aspect-ratio: 1.05;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9.5px;
    font-weight: 700;
  }
  .tooth-box.detected {
    background: var(--green-detect);
    color: #FFFFFF;
    border: 1px solid #15803D;
  }
  .tooth-box.missing {
    background: var(--gray-missing);
    color: var(--gray-missing-text);
    border: 1px solid #CBD5E1;
  }

  .chart-legend-row {
    display: flex;
    gap: 16px;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: 4px;
  }
  .chart-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .chart-legend-box {
    width: 11px;
    height: 11px;
    border-radius: 2px;
  }

  /* ── Structure Detection Table ────────────────────────────────── */
  .struct-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  .struct-tbl th {
    background: #F8FAFC;
    color: var(--navy-dark);
    font-weight: 700;
    padding: 3.5px 6px;
    border-top: 1px solid var(--border-card);
    border-bottom: 1px solid var(--border-card);
    text-align: left;
    font-size: 10.5px;
  }
  .struct-tbl td {
    padding: 3px 6px;
    border-bottom: 1px solid #F1F5F9;
    color: var(--text-primary);
  }

  /* ── Measurements Sub-cards ───────────────────────────────────── */
  .measure-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }
  .measure-box {
    border: 1px solid var(--border-light);
    border-radius: 6px;
    background: #F8FAFC;
    padding: 8px 6px;
    text-align: center;
  }
  .measure-title {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 2px;
  }
  .measure-sub {
    font-size: 8.5px;
    color: var(--text-faint);
    margin-bottom: 6px;
  }
  .measure-val {
    font-size: 16px;
    font-weight: 800;
    color: var(--navy-dark);
    line-height: 1.1;
  }
  .measure-approx {
    font-size: 9px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  /* ── Disclaimer Banner ────────────────────────────────────────── */
  .disclaimer-banner {
    margin-top: 10px;
    padding: 7px 14px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    font-size: 11.5px;
    color: #1E40AF;
    text-align: center;
    font-weight: 500;
  }

  /* Sliders & Radios */
  input[type=range] { accent-color: var(--navy-dark) !important; }
  [data-testid="stRadio"] label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_oralseg_model():
    """Load and cache the OralSeg model. Returns (model, device, device_name, error)."""
    try:
        from utils.oralseg_inference import load_model
        model, device, device_name = load_model(str(CHECKPOINT_PATH))
        return model, device, device_name, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"Model load error: {e}\n{traceback.format_exc()}"


def get_device_info():
    """Return device info dict without loading the model."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        return {
            "cuda": cuda_ok,
            "gpu_name": gpu_name,
            "device_str": "cuda" if cuda_ok else "cpu",
        }
    except Exception:
        return {"cuda": False, "gpu_name": "N/A", "device_str": "cpu"}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        dev = get_device_info()
        gpu_label = dev['gpu_name'] if dev['cuda'] else 'CPU Only'

        st.markdown("""
        <div class="brand">
          <div class="brand-mark">OS</div>
          <div>
            <div class="brand-name">OralSeg</div>
            <div class="brand-sub">CBCT Segmentation</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="side-label">Model Status</div>', unsafe_allow_html=True)
        model_exists = CHECKPOINT_PATH.exists()
        if model_exists:
            st.markdown("""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:8px 10px;font-size:11.5px;font-family:var(--mono);color:#0F172A;display:flex;align-items:center;gap:6px;">
              <span class="status-dot on"></span>
              <span style="font-weight:600;">model_workstation39.pt</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;padding:8px 10px;font-size:11.5px;font-family:var(--mono);color:#DC2626;display:flex;align-items:center;gap:6px;">
              <span class="status-dot off"></span>
              <span style="font-weight:600;">Model missing</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="side-label">System Hardware</div>', unsafe_allow_html=True)
        cuda_status_bg = "#ECFDF5" if dev["cuda"] else "#FFFBEB"
        cuda_status_border = "#A7F3D0" if dev["cuda"] else "#FDE68A"
        cuda_status_color = "#047857" if dev["cuda"] else "#B45309"
        cuda_text = "CUDA Accelerated" if dev["cuda"] else "CPU Fallback"

        st.markdown(f"""
        <div style="background:{cuda_status_bg};border:1px solid {cuda_status_border};border-radius:6px;padding:8px 10px;margin-bottom:8px;">
          <div style="font-size:11.5px;font-weight:700;color:{cuda_status_color};display:flex;align-items:center;gap:6px;">
            <span class="status-dot on" style="background:{cuda_status_color};box-shadow:0 0 6px {cuda_status_border};"></span>
            {cuda_text}
          </div>
          <div style="font-size:11px;color:#334155;font-weight:600;font-family:var(--mono);margin-top:3px;">
            {gpu_label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="side-meta">
          <div><span class="dot">&#9679;</span>OralSeg 3D Engine v2.0</div>
          <div><span class="dot">&#9679;</span>MONAI sliding window 96³</div>
          <div><span class="dot">&#9679;</span>35 Anatomical Classes</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="title-card">
      <div style="display:flex;align-items:center;gap:10px;">
        <h1 class="title-main">AI Dental Implant Planning System</h1>
        <span class="title-badge">BETA</span>
      </div>
      <p class="title-sub">CBCT-based 3D Dental Segmentation using OralSeg &nbsp;&bull;&nbsp; 35-Class Anatomical Segmentation</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────────────────────────────────────
def render_upload_section():
    st.markdown("### 📁 Upload CBCT Dataset (DICOM ZIP or NIfTI)")
    st.markdown(
        "<p style='color:var(--text-secondary);font-size:0.88rem;margin-bottom:12px;'>"
        "Upload a <b>ZIP archive</b> containing a CBCT DICOM series, or a direct <b>.nii / .nii.gz</b> 3D volume. "
        "Patient identifying information will not be displayed.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a ZIP file (.zip) or NIfTI volume (.nii, .nii.gz)",
        type=["zip", "nii", "gz"],
        key="cbct_upload",
        help="ZIP archive containing DICOM files or direct 3D NIfTI volume (.nii / .nii.gz).",
    )
    return uploaded


# ─────────────────────────────────────────────────────────────────────────────
# Process Upload (ZIP or NIfTI) and show CBCT info
# ─────────────────────────────────────────────────────────────────────────────
def process_upload(uploaded_file) -> Optional[dict]:
    """Extract and validate DICOM series or direct NIfTI volume. Returns state dict or None on error."""
    raw_name = uploaded_file.name
    is_nifti = raw_name.lower().endswith(".nii.gz") or raw_name.lower().endswith(".nii") or raw_name.lower().endswith(".gz")

    if is_nifti:
        from utils.nifti_utils import validate_nifti_for_inference
        import nibabel as nib

        if raw_name.lower().endswith(".nii.gz"):
            case_name = raw_name[:-7]
            nifti_path = str(CONVERTED_DIR / f"{case_name}.nii.gz")
            with st.spinner("Reading NIfTI volume (.nii.gz)..."):
                try:
                    file_bytes = uploaded_file.read()
                    with open(nifti_path, "wb") as f:
                        f.write(file_bytes)
                    nifti_info = validate_nifti_for_inference(nifti_path)
                except Exception as e:
                    st.error(f"❌ Error reading NIfTI file: {e}")
                    return None
        elif raw_name.lower().endswith(".nii"):
            case_name = raw_name[:-4]
            temp_path = str(CONVERTED_DIR / f"{case_name}_temp.nii")
            nifti_path = str(CONVERTED_DIR / f"{case_name}.nii.gz")
            with st.spinner("Processing NIfTI volume (.nii)..."):
                try:
                    file_bytes = uploaded_file.read()
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    img = nib.load(temp_path)
                    nib.save(img, nifti_path)
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                    nifti_info = validate_nifti_for_inference(nifti_path)
                except Exception as e:
                    st.error(f"❌ Error reading NIfTI file: {e}")
                    return None
        else: # .gz
            case_name = raw_name[:-3]
            nifti_path = str(CONVERTED_DIR / f"{case_name}.nii.gz")
            with st.spinner("Reading NIfTI volume..."):
                try:
                    file_bytes = uploaded_file.read()
                    with open(nifti_path, "wb") as f:
                        f.write(file_bytes)
                    nifti_info = validate_nifti_for_inference(nifti_path)
                except Exception as e:
                    st.error(f"❌ Error reading NIfTI file: {e}")
                    return None

        # Show non-PII CBCT info card for NIfTI
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        st.markdown("#### 🩻 CBCT NIfTI Information")

        shape = nifti_info["shape"]
        spacing = nifti_info["spacing_mm"]
        spacing_str = f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm" if len(spacing) >= 3 else "N/A"
        shape_str = f"{shape[0]} × {shape[1]} × {shape[2]}" if len(shape) >= 3 else "N/A"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Volume Dimensions", shape_str)
        col2.metric("Axial Slices (Z)", shape[2] if len(shape) >= 3 else "N/A")
        col3.metric("Voxel Spacing", spacing_str)
        col4.metric("Intensity Range", f"{int(nifti_info['intensity_min'])} to {int(nifti_info['intensity_max'])} HU")

        st.markdown("""</div>""", unsafe_allow_html=True)

        return {
            "is_nifti": True,
            "zip_name": case_name,
            "extract_dir": None,
            "series_map": {},
            "meta": {
                "num_slices": shape[2] if len(shape) >= 3 else 1,
                "series_count": 1,
                "modality": "CBCT",
                "pixel_spacing": [spacing[0], spacing[1]] if len(spacing) >= 2 else [1.0, 1.0],
                "spacing_mm": spacing,
                "shape": shape,
            },
            "nifti_info": nifti_info,
            "vol_info": {
                "volume_shape": shape,
                "voxel_spacing_mm": spacing,
                "intensity_range": [nifti_info["intensity_min"], nifti_info["intensity_max"]],
                "num_slices": shape[2] if len(shape) >= 3 else 1,
                "nifti_path": nifti_path,
            },
            "series_files": [],
        }

    # Otherwise DICOM ZIP:
    from utils.dicom_utils import inspect_zip, get_series_files

    zip_name = Path(uploaded_file.name).stem
    extract_dir = str(UPLOADS_DIR / zip_name)
    shutil.rmtree(extract_dir, ignore_errors=True)

    with st.spinner("Inspecting DICOM archive..."):
        try:
            zip_bytes = uploaded_file.read()
            series_map, meta = inspect_zip(zip_bytes, extract_dir)
        except ValueError as e:
            st.error(f"❌ {e}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error reading ZIP: {e}")
            return None

    # Show non-PII CBCT info
    st.markdown("""<div class="card">""", unsafe_allow_html=True)
    st.markdown("#### 🩻 CBCT Information")

    if "multi_series_warning" in meta:
        st.warning(meta["multi_series_warning"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DICOM Slices", meta["num_slices"])
    col2.metric("Series Detected", meta["series_count"])
    col3.metric("Modality", meta.get("modality", "CT"))
    ps = meta.get("pixel_spacing")
    spacing_str = f"{ps[0]:.2f} × {ps[1]:.2f} mm" if ps else "N/A"
    col4.metric("Pixel Spacing", spacing_str)

    st.markdown("""</div>""", unsafe_allow_html=True)

    return {
        "is_nifti": False,
        "zip_name": zip_name,
        "extract_dir": extract_dir,
        "series_map": series_map,
        "meta": meta,
        "series_files": get_series_files(series_map),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(upload_state: dict, overlap: float = 0.25):
    """Execute the full segmentation pipeline with progress reporting."""
    from utils.nifti_utils import dicom_series_to_nifti, validate_nifti_for_inference, load_nifti_volume
    from utils.oralseg_inference import run_inference, save_result_json

    is_nifti     = upload_state.get("is_nifti", False)
    case_name    = upload_state["zip_name"]
    meta         = upload_state["meta"]

    nifti_path   = str(CONVERTED_DIR / f"{case_name}.nii.gz")
    seg_path     = str(OUTPUTS_DIR   / f"{case_name}_segmentation.nii.gz")
    json_path    = str(OUTPUTS_DIR   / f"{case_name}_results.json")

    if is_nifti:
        pipeline_steps = [
            "Validating NIfTI volume",
            "Loading OralSeg model",
            "Running 3D inference",
            "Generating segmentation mask",
            "Saving outputs",
            "Preparing visualization",
        ]
    else:
        pipeline_steps = [
            "Validating DICOM series",
            "Extracting DICOM files",
            "Reading DICOM metadata",
            "Converting DICOM → NIfTI",
            "Validating NIfTI volume",
            "Loading OralSeg model",
            "Running 3D inference",
            "Generating segmentation mask",
            "Saving outputs",
            "Preparing visualization",
        ]

    # ── UI scaffolding ────────────────────────────────────────────────────
    progress_bar  = st.progress(0.0)
    step_ph       = st.empty()
    log_ph        = st.empty()

    step_counter = [0]

    def advance(step_name: str):
        step_counter[0] += 1
        pct = step_counter[0] / len(pipeline_steps)
        progress_bar.progress(pct)
        step_ph.markdown(
            f"<p style='color:var(--navy-blue);font-size:0.92rem;font-weight:600;'>⚙ Step {step_counter[0]}/{len(pipeline_steps)}: <b>{step_name}</b></p>",
            unsafe_allow_html=True,
        )

    def log(msg: str):
        log_ph.markdown(
            f"<p style='color:var(--text-secondary);font-size:0.82rem;'>ℹ {msg}</p>",
            unsafe_allow_html=True,
        )

    try:
        if not is_nifti:
            series_files = upload_state["series_files"]
            # Step 1: Validate
            advance(pipeline_steps[0])
            if not series_files:
                raise ValueError("No DICOM files in upload state.")

            # Step 2: Extract confirmed
            advance(pipeline_steps[1])
            log(f"{len(series_files)} DICOM files ready for conversion.")
            time.sleep(0.2)

            # Step 3: Metadata
            advance(pipeline_steps[2])
            log(f"Modality: {meta.get('modality','CT')}, Slices: {meta['num_slices']}")
            time.sleep(0.2)

            # Step 4: DICOM → NIfTI
            advance(pipeline_steps[3])
            _, vol_info = dicom_series_to_nifti(series_files, nifti_path, progress_callback=log)
        else:
            vol_info = upload_state.get("vol_info", {})

        # Step: Validate NIfTI
        advance(pipeline_steps[4 if not is_nifti else 0])
        nifti_info = validate_nifti_for_inference(nifti_path)
        log(f"Volume shape: {nifti_info['shape']}, spacing: {[f'{s:.2f}' for s in nifti_info['spacing_mm']]} mm")

        # Step: Load model
        advance(pipeline_steps[5 if not is_nifti else 1])
        model, device, device_name, err = _load_oralseg_model()
        if err:
            raise RuntimeError(err)
        log(f"OralSeg loaded on {device_name}")

        # Step: Inference
        advance(pipeline_steps[6 if not is_nifti else 2])
        seg_array, seg_info = run_inference(
            model, device, nifti_path, seg_path,
            sw_batch_size=2, overlap=overlap,
            progress_callback=log,
        )
        advance(pipeline_steps[7 if not is_nifti else 3])
        log(f"Detected {seg_info['teeth_count']} teeth, {len(seg_info['detected_labels'])-1} total structures.")

        # Step: Save JSON
        advance(pipeline_steps[8 if not is_nifti else 4])
        result_json = save_result_json(
            json_path,
            case_name=case_name,
            num_dicom_slices=meta["num_slices"],
            volume_shape=nifti_info["shape"],
            voxel_spacing=nifti_info["spacing_mm"],
            device_str=str(device),
            checkpoint_name="model_workstation39.pt",
            seg_info=seg_info,
        )

        # Step: Prepare viz
        advance(pipeline_steps[9 if not is_nifti else 5])
        log("Loading volumes for visualization...")

        # Load CBCT for display (float32, xyz)
        import nibabel as nib
        cbct_img = nib.load(nifti_path)
        cbct_vol = cbct_img.get_fdata(dtype=np.float32)

        # seg_array from inference is (D,H,W) in RAS space;
        # If shapes don't match, transpose
        seg_vol = seg_array.transpose(2, 1, 0) if seg_array.shape != cbct_vol.shape else seg_array

        # Align shapes (clamp to minimum)
        min_shape = tuple(min(a, b) for a, b in zip(cbct_vol.shape, seg_vol.shape))
        cbct_vol = cbct_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
        seg_vol  = seg_vol[:min_shape[0],  :min_shape[1],  :min_shape[2]]

        progress_bar.progress(1.0)
        step_ph.markdown(
            "<p style='color:#16A34A;font-size:1rem;font-weight:700;'>✓ Segmentation Completed Successfully</p>",
            unsafe_allow_html=True,
        )
        log_ph.empty()

        return {
            "case_name":    case_name,
            "nifti_path":   nifti_path,
            "seg_path":     seg_path,
            "json_path":    json_path,
            "seg_info":     seg_info,
            "vol_info":     vol_info,
            "nifti_info":   nifti_info,
            "result_json":  result_json,
            "cbct_vol":     cbct_vol,
            "seg_vol":      seg_vol,
            "device_name":  device_name,
        }

    except Exception as e:
        # Check for CUDA OOM specifically
        try:
            import torch
            if isinstance(e, torch.cuda.OutOfMemoryError):
                st.error(
                    "❌ **GPU out of memory.** Your RTX 2050 (4 GB) ran out of VRAM.\n\n"
                    "**Tips:**\n"
                    "- Close other GPU-heavy applications\n"
                    "- Restart the Streamlit app to free VRAM\n"
                    "- The app already retries automatically with sw_batch_size=1"
                )
                return None
        except Exception:
            pass

        err_str = str(e)
        if "out of memory" in err_str.lower() or "cuda" in err_str.lower() and "memory" in err_str.lower():
            st.error(
                "❌ **GPU out of memory.** Please close other applications and try again."
            )
            return None

        st.error(f"❌ Pipeline error: {e}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Results rendering (Exact Pixel-Matched Clinical Report Dashboard)
# ─────────────────────────────────────────────────────────────────────────────
def render_results(result: dict):
    from utils.oralseg_inference import LABEL_MAP, TOOTH_FDI_MAP
    from utils.visualization import render_slice_views, extract_mesh_for_label, export_binary_stl, build_3d_interactive_figure
    from datetime import datetime
    from utils.diagnocat_report import (
        analyze_tooth_conditions, generate_tooth_micro_slices, build_interactive_odontogram_html
    )
    from utils.implant_visualization import build_diagnocat_panoramic_3d_scene
    from utils.implant_geometry import ALL_FDI_TEETH

    seg_info   = result["seg_info"]
    cbct_vol   = result["cbct_vol"]
    seg_vol    = result["seg_vol"]
    case_name  = result["case_name"]
    nifti_info = result.get("nifti_info", {})
    vol_info   = result.get("vol_info", {})

    shape_str   = " × ".join(str(s) for s in cbct_vol.shape) if cbct_vol is not None else "N/A"
    spacing_arr = nifti_info.get("spacing_mm", [1.0, 1.0, 1.0])
    if spacing_arr is None or len(spacing_arr) < 3:
        spacing_arr = [1.0, 1.0, 1.0]

    teeth_count   = seg_info.get("teeth_count", 0)
    scan_date_str = "20 May 2025"
    patient_id_str = case_name if len(case_name) <= 12 else "K01"

    # ── Top-level Workflow Tabs ────────────────────────────────────────────
    tab_report, tab_plan = st.tabs(["📊 CBCT AI CLINICAL REPORT (Diagnocat-Style)", "🦷 3D CBCT IMPLANT WORKSTATION"])

    with tab_report:
        # ── 1. Diagnocat-Style Top Header Bar ─────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;margin-bottom:16px;box-shadow:0 1px 4px rgba(15,23,42,0.03);">
          <div style="display:flex;align-items:center;gap:14px;">
            <div style="font-size:18px;font-weight:900;color:#6366F1;display:flex;align-items:center;gap:6px;letter-spacing:-0.3px;">
              <span style="font-size:22px;">🐱</span> Diagnocat <span style="font-size:10.5px;font-weight:800;color:#4338CA;background:#EEF2FF;padding:2px 7px;border-radius:4px;border:1px solid #C7D2FE;">AI REPORT</span>
            </div>
            <div style="height:18px;width:1.5px;background:#E2E8F0;"></div>
            <div style="color:#64748B;font-size:13px;font-weight:500;">
              <span style="color:#4F46E5;font-weight:700;">{patient_id_str}</span> <span style="color:#CBD5E1;">&rsaquo;</span> CBCT AI Report &middot; {scan_date_str}
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:18px;">
            <span style="font-size:15px;cursor:pointer;color:#64748B;" title="Language">🌐</span>
            <span style="font-size:15px;cursor:pointer;color:#64748B;" title="Dark Mode">🌙</span>
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="width:30px;height:30px;border-radius:50%;background:#E0E7FF;color:#4338CA;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;">KJ</div>
              <div style="font-size:12.5px;font-weight:700;color:#1E293B;">Katherine Johnston</div>
            </div>
            <div style="background:#DCFCE7;border:1.5px solid #16A34A;color:#166534;font-size:11.5px;font-weight:800;padding:3px 12px;border-radius:6px;">
              ✓ Approved
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Precompute condition analyses for all 32 teeth
        detected_set = set(seg_info.get("detected_teeth", []))
        all_conditions = {}
        for t in ALL_FDI_TEETH:
            all_conditions[t] = analyze_tooth_conditions(
                seg_vol, cbct_vol, t, is_detected=(t in detected_set), voxel_spacing_mm=spacing_arr
            )

        # ── 2. Split View (Left: 3D Panoramic & Odontogram | Right: Tooth Cards) ──
        col_diag_left, col_diag_right = st.columns([1.12, 1.28])

        with col_diag_left:
            st.markdown("<div style='font-size:17px;font-weight:900;color:#0F172A;margin-bottom:8px;'>CBCT AI Report</div>", unsafe_allow_html=True)

            # 3D Viewport Controls & Layer Toggles (Pills)
            pill_c1, pill_c2, pill_c3, pill_c4 = st.columns(4)
            with pill_c1:
                t_perio = st.checkbox("🔴 Perio", value=True, key="diag_pill_perio")
            with pill_c2:
                t_resto = st.checkbox("🟣 Restorative", value=True, key="diag_pill_resto")
            with pill_c3:
                t_endo  = st.checkbox("🟠 Endo", value=True, key="diag_pill_endo")
            with pill_c4:
                t_anat  = st.checkbox("🦴 Anatomy", value=True, key="diag_pill_anat")

            # Selected tooth for focus
            default_diag_fdi = 36 if 36 in detected_set else (list(detected_set)[0] if detected_set else 46)
            curr_sel_fdi = st.session_state.get("diag_selected_fdi", default_diag_fdi)

            # Render Diagnocat 3D Panoramic Scene
            fig_diag_3d = build_diagnocat_panoramic_3d_scene(
                seg_vol, voxel_spacing_mm=spacing_arr,
                show_perio=t_perio, show_restorative=t_resto, show_endo=t_endo, show_anatomy=t_anat,
                selected_fdi=curr_sel_fdi,
            )
            st.plotly_chart(fig_diag_3d, use_container_width=True)

            # Interactive Odontogram
            st.markdown(build_interactive_odontogram_html(list(detected_set), all_conditions, selected_fdi=curr_sel_fdi), unsafe_allow_html=True)

            # Quick Tooth Focus Selector
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            sel_tooth_focus = st.selectbox(
                "🔍 Focus Tooth Detail View:",
                options=ALL_FDI_TEETH,
                index=ALL_FDI_TEETH.index(curr_sel_fdi) if curr_sel_fdi in ALL_FDI_TEETH else 0,
                format_func=lambda x: f"FDI Tooth {x} ({all_conditions[x]['category'].capitalize()})",
                key="diag_focus_select",
            )
            if sel_tooth_focus != curr_sel_fdi:
                st.session_state["diag_selected_fdi"] = sel_tooth_focus
                st.rerun()

        with col_diag_right:
            st.markdown("<div style='font-size:15px;font-weight:800;color:#0F172A;margin-bottom:8px;'>Tooth-by-Tooth Clinical Findings &amp; Slices</div>", unsafe_allow_html=True)

            # Display cards for focused tooth and adjacent/detected teeth
            display_teeth_order = [curr_sel_fdi] + [t for t in detected_set if t != curr_sel_fdi]
            
            # Show up to 3 active cards for smooth interactive performance
            for tooth_fdi in display_teeth_order[:3]:
                t_info = all_conditions.get(tooth_fdi, {})
                anatomy_info = t_info.get("anatomy", {})
                tag_list = t_info.get("tags", [])
                is_focused = (tooth_fdi == curr_sel_fdi)

                card_border = "2px solid #6366F1; box-shadow: 0 4px 12px rgba(99,102,241,0.08);" if is_focused else "1px solid #CBD5E1;"

                # Generate high-contrast micro-CT slices
                slice_thumbs = generate_tooth_micro_slices(cbct_vol, seg_vol, tooth_fdi, voxel_spacing_mm=spacing_arr, num_slices=6)

                # Format tag pills
                tag_pills_html = "".join([
                    f"<span style='background:{tg['bg']};color:{tg['color']};border:1px solid {tg['color']}33;padding:3px 8px;border-radius:5px;font-size:11px;font-weight:700;display:inline-block;margin:2px 3px 2px 0;'>"
                    f"{tg['label']}</span>"
                    for tg in tag_list
                ])

                # Format slice thumbnails grid
                thumbs_html = "".join([
                    f"<img src='data:image/jpeg;base64,{b64}' style='width:15.5%;border-radius:4px;border:1px solid #334155;background:#000000;' />"
                    for b64 in slice_thumbs
                ])

                st.markdown(f"""
                <div style="background:#FFFFFF;border:{card_border}border-radius:10px;padding:14px 16px;margin-bottom:14px;background:#FFFFFF;">
                  <!-- Card Header -->
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #F1F5F9;">
                    <div style="display:flex;align-items:center;gap:8px;">
                      <span style="font-size:16px;font-weight:900;color:#0F172A;">Tooth {tooth_fdi}</span>
                      <span style="font-size:11.5px;color:#64748B;font-weight:600;">{anatomy_info.get('roots', '')} &middot; {anatomy_info.get('canals', '')}</span>
                    </div>
                    <span style="font-size:12px;color:#94A3B8;cursor:pointer;" title="Reset">↺</span>
                  </div>

                  <!-- Pathology / Condition Badges -->
                  <div style="margin-bottom:10px;">
                    {tag_pills_html}
                  </div>

                  <!-- Micro-CT Slice Thumbnails -->
                  <div style="display:flex;gap:4px;justify-content:space-between;margin-bottom:10px;background:#000000;padding:4px;border-radius:6px;">
                    {thumbs_html}
                  </div>

                  <!-- Card Action Bar -->
                  <div style="display:flex;justify-content:space-between;align-items:center;padding-top:6px;border-top:1px solid #F1F5F9;">
                    <div style="display:flex;gap:8px;">
                      <button style="background:#F8FAFC;border:1px solid #CBD5E1;border-radius:5px;padding:3px 9px;font-size:11px;font-weight:700;color:#475569;cursor:pointer;">+ Condition</button>
                      <button style="background:#F8FAFC;border:1px solid #CBD5E1;border-radius:5px;padding:3px 9px;font-size:11px;font-weight:700;color:#475569;cursor:pointer;">✏ Comment</button>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-size:11.5px;color:#6366F1;font-weight:700;cursor:pointer;">Slices</span>
                      <span style="background:#DCFCE7;border:1px solid #16A34A;color:#166534;font-size:10.5px;font-weight:800;padding:2px 8px;border-radius:4px;">Approved</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── 3. Bottom Sticky Action Bar ─────────────────────────────────────────
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:#F8FAFC;border:1px solid #CBD5E1;border-radius:10px;box-shadow:0 1px 4px rgba(15,23,42,0.03);">
          <div style="display:flex;align-items:center;gap:20px;font-size:12px;color:#475569;font-weight:600;">
            <span style="display:flex;align-items:center;gap:6px;">🟣 Suspicious teeth ⓘ</span>
            <span style="display:flex;align-items:center;gap:6px;">🟣 Conditions details ⓘ</span>
          </div>
          <div style="display:flex;gap:10px;">
            <div style="font-size:11.5px;color:#64748B;align-self:center;">OralSeg 35-Class Segmentation Active</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Downloads Row
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            if Path(result["nifti_path"]).exists():
                with open(result["nifti_path"], "rb") as f:
                    st.download_button("⬇ Download CBCT NIfTI", data=f.read(), file_name=f"{case_name}.nii.gz", mime="application/gzip", key="dl_cbct")
        with col_d2:
            if Path(result["seg_path"]).exists():
                with open(result["seg_path"], "rb") as f:
                    st.download_button("⬇ Download Seg NIfTI", data=f.read(), file_name=f"{case_name}_segmentation.nii.gz", mime="application/gzip", key="dl_seg")
        with col_d3:
            stl_path = str(OUTPUTS_DIR / f"{case_name}_3d_model.stl")
            if not Path(stl_path).exists():
                v, f = extract_mesh_for_label(seg_vol, list(range(1, 36)), step_size=2, voxel_spacing=tuple(spacing_arr) if spacing_arr else None, smoothing_iterations=12)
                if v is not None:
                    export_binary_stl(v, f, stl_path)
            if Path(stl_path).exists():
                with open(stl_path, "rb") as f:
                    st.download_button("⬇ Download 3D Mesh (STL)", data=f.read(), file_name=f"{case_name}_3d_model.stl", mime="application/octet-stream", key="dl_stl")
        with col_d4:
            if Path(result["json_path"]).exists():
                with open(result["json_path"], "r") as f:
                    st.download_button("⬇ Export Results JSON", data=f.read(), file_name=f"{case_name}_results.json", mime="application/json", key="dl_json")

    with tab_plan:
        # ── AI-Assisted Implant Planning Module ──────────────────────────────
        render_implant_planning_section(result)


def render_implant_planning_section(result: dict):
    """
    Renders the interactive 3D CBCT AI-Assisted Implant Planning Workstation.
    Features:
      1. Missing tooth candidate classification (Candidate / Review / Insufficient)
      2. Physical 3D anatomical measurements in mm (Mesiodistal, Ridge Width, Bone Height)
      3. True 3D virtual implant geometry with interactive positioning and angulation
      4. Multi-parameter collision and safety engine (Canal, Roots, Cortical plates)
      5. Enhanced 3D Plotly viewport with layer controls and camera presets (Buccal/Lingual/Occlusal/Mesial/Distal)
      6. Synchronized 2D cross-sectional orthogonal planning views (Axial, Coronal, Sagittal)
      7. Dentist Review summary, clinical disclaimer, and STL/JSON/HTML multi-format exports
    """
    from utils.implant_geometry import (
        detect_potential_missing_teeth, get_tooth_site_coordinates,
        classify_missing_teeth_sites, sample_implant_cylinder_points,
        ALL_FDI_TEETH, FDI_TO_LABEL, get_adjacent_teeth_fdi
    )
    from utils.bone_measurements import get_site_measurements
    from utils.collision_analysis import analyze_implant_safety
    from utils.implant_visualization import render_2d_mpr_implant_views, build_3d_implant_scene, build_local_roi_3d_scene
    from utils.implant_report import (
        generate_implant_planning_data, save_implant_planning_json,
        generate_printable_html_report, export_virtual_implant_stl, CLINICAL_DISCLAIMER_3D,
        plan_data_to_json_str
    )
    from utils.implant_suggester_3d import (
        auto_suggest_implants_3d, seg_vol_cache_key,
        build_suggestion_summary_html, suggest_preliminary_implant_range_for_site,
        suggestions_to_dict_list as sugg3d_to_dict,
        EXCELLENT_THRESHOLD, ACCEPTABLE_THRESHOLD,
    )

    seg_info   = result["seg_info"]
    seg_vol    = result["seg_vol"]
    cbct_vol   = result["cbct_vol"]
    case_name  = result["case_name"]
    nifti_info = result.get("nifti_info", {})
    spacing    = nifti_info.get("spacing_mm", None) or [1.0, 1.0, 1.0]
    if len(spacing) < 3:
        spacing = [1.0, 1.0, 1.0]

    # ── Workflow Header ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #CBD5E1;border-radius:10px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 4px rgba(15,23,42,0.04);">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:18px;font-weight:800;color:#0F3B7A;display:flex;align-items:center;gap:8px;">
            <span>🦷</span> 3D CBCT Implant Assistance Workstation
          </div>
          <div style="font-size:12px;color:#64748B;margin-top:2px;">
            AI-Assisted Preliminary Decision Support from 35-Class Volumetric Segmentation &middot; All measurements in physical mm
          </div>
        </div>
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;padding:4px 10px;font-size:11px;font-weight:700;color:#1E40AF;">
          OralSeg 35-Class Active
        </div>
      </div>
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid #F1F5F9;display:flex;gap:6px;overflow-x:auto;font-size:11px;color:#475569;">
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">1. Site Selection</span> &rarr;
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">2. 3D Anatomical Analysis</span> &rarr;
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">3. Virtual Implant</span> &rarr;
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">4. Safety &amp; Collision Engine</span> &rarr;
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">5. Cross-Sectional Planning</span> &rarr;
        <span style="background:#F1F5F9;padding:3px 8px;border-radius:4px;font-weight:600;">6. Dentist Review &amp; Export</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Categorize Missing Teeth Sites ────────────────────────────────────────
    detected_teeth = seg_info.get("detected_teeth", [])
    categorized_sites = classify_missing_teeth_sites(seg_vol, seg_info, voxel_spacing_mm=spacing)
    candidates = categorized_sites["candidate"]
    needs_review = categorized_sites["needs_review"]
    insufficient = categorized_sites["insufficient_info"]

    # ── 1. Site Selection & AI Range Assessment ──────────────────────────────
    col_sel_left, col_sel_right = st.columns([1.1, 1.9])

    with col_sel_left:
        st.markdown("#### 🎯 1. Select Implant Site")
        
        # Build structured selectbox options
        site_options = []
        site_labels = {}

        for item in candidates:
            f = item["fdi"]
            site_options.append(f)
            site_labels[f] = f"FDI {f} (🟢 Candidate Site)"

        for item in needs_review:
            f = item["fdi"]
            site_options.append(f)
            site_labels[f] = f"FDI {f} (🟡 Needs Review)"

        for item in insufficient:
            f = item["fdi"]
            site_options.append(f)
            site_labels[f] = f"FDI {f} (⚪ Insufficient Data)"

        # If no missing teeth, fallback to all 32 teeth
        if not site_options:
            site_options = ALL_FDI_TEETH
            site_labels = {f: f"FDI {f} ({'Detected' if f in detected_teeth else 'Not Detected'})" for f in ALL_FDI_TEETH}

        # Default site
        default_site = 46 if 46 in site_options else site_options[0]
        curr_site_idx = site_options.index(st.session_state.get("cbct_plan_fdi", default_site)) if st.session_state.get("cbct_plan_fdi", default_site) in site_options else 0

        selected_fdi = st.selectbox(
            "Target FDI Tooth Site:",
            options=site_options,
            index=curr_site_idx,
            format_func=lambda x: site_labels.get(x, f"FDI {x}"),
            key="cbct_plan_fdi",
        )

        is_mandibular = (selected_fdi >= 31 and selected_fdi <= 48)
        arch_str = "Mandibular (Lower Jaw)" if is_mandibular else "Maxillary (Upper Jaw)"
        mesial_neighbor, distal_neighbor = get_adjacent_teeth_fdi(selected_fdi, detected_teeth)
        
        st.markdown(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:12px;margin-top:6px;">
          <div><b>Arch:</b> {arch_str}</div>
          <div><b>Mesial Adjacent:</b> {'FDI ' + str(mesial_neighbor) if mesial_neighbor else 'None detected'}</div>
          <div><b>Distal Adjacent:</b> {'FDI ' + str(distal_neighbor) if distal_neighbor else 'None detected'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_sel_right:
        st.markdown("#### 🤖 2. AI Preliminary Estimation & Actions")

        # Initial bone measurements at base coordinate
        base_center_vox, is_det = get_tooth_site_coordinates(seg_vol, selected_fdi, voxel_spacing_mm=spacing)
        init_meas = get_site_measurements(seg_vol, base_center_vox, selected_fdi, detected_teeth=detected_teeth, voxel_spacing_mm=spacing)
        
        # Canal clearance estimate
        init_pts = sample_implant_cylinder_points(base_center_vox, 4.0, 10.0, voxel_spacing_mm=spacing)
        init_safety = analyze_implant_safety(init_pts, seg_vol, selected_fdi, detected_teeth=detected_teeth, voxel_spacing_mm=spacing)
        init_canal_dist = init_safety.get("canal_distance_mm")

        range_info = suggest_preliminary_implant_range_for_site(
            init_meas["bone_height_mm"], init_meas["ridge_width_mm"], init_canal_dist
        )

        st.markdown(f"""
        <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px 14px;font-size:12px;color:#166534;">
          <div style="font-weight:700;font-size:13px;margin-bottom:4px;">AI-Assisted Preliminary Dimension Ranges (FDI {selected_fdi})</div>
          <div style="display:flex;gap:18px;margin:6px 0;">
            <div><b>Suggested Diameter Range:</b> <span style="font-family:monospace;font-size:12.5px;color:#0F3B7A;">{range_info['diameter_range_str']}</span></div>
            <div><b>Suggested Length Range:</b> <span style="font-family:monospace;font-size:12.5px;color:#0F3B7A;">{range_info['length_range_str']}</span></div>
          </div>
          <div style="font-size:11px;color:#15803D;">Baseline candidate: Ø{range_info['default_diameter_mm']:.1f} &times; {range_info['default_length_mm']:.0f} mm (based on measured ridge width &amp; available height)</div>
        </div>
        """, unsafe_allow_html=True)

        # Dentist Control Buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("⚡ Apply AI Candidate", key="btn_apply_ai", use_container_width=True):
                st.session_state[f"cbct_diam_{selected_fdi}"] = float(range_info["default_diameter_mm"])
                st.session_state[f"cbct_len_{selected_fdi}"]  = float(range_info["default_length_mm"])
                st.session_state[f"cbct_offx_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_offy_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_offz_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_angx_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_angy_{selected_fdi}"] = 0.0
                st.rerun()

        with btn_c2:
            if st.button("↺ Reset Parameters", key="btn_reset_pos", use_container_width=True):
                st.session_state[f"cbct_diam_{selected_fdi}"] = 4.0
                st.session_state[f"cbct_len_{selected_fdi}"]  = 10.0
                st.session_state[f"cbct_offx_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_offy_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_offz_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_angx_{selected_fdi}"] = 0.0
                st.session_state[f"cbct_angy_{selected_fdi}"] = 0.0
                st.rerun()

        with btn_c3:
            if st.button("🔍 Scan All Missing", key="btn_scan_all", use_container_width=True):
                with st.spinner("Analyzing all 32 sites..."):
                    st.session_state["cbct_sugg_all"] = auto_suggest_implants_3d(
                        seg_vol, seg_info, voxel_spacing_mm=spacing
                    )
                st.rerun()

    st.markdown("<hr style='margin:1rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── 2. Interactive Virtual Implant Placement Controls ─────────────────────
    st.markdown("#### 🔩 3. Interactive Virtual Implant Controls")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        imp_diam = st.slider(
            "Diameter (Ø mm)", min_value=3.0, max_value=6.0, value=float(st.session_state.get(f"cbct_diam_{selected_fdi}", 4.0)), step=0.1,
            key=f"cbct_diam_{selected_fdi}", help="Virtual implant diameter in mm"
        )
    with col_p2:
        imp_len = st.slider(
            "Length (L mm)", min_value=6.0, max_value=16.0, value=float(st.session_state.get(f"cbct_len_{selected_fdi}", 10.0)), step=0.5,
            key=f"cbct_len_{selected_fdi}", help="Virtual implant length in mm"
        )
    with col_p3:
        ang_x = st.slider(
            "Buccolingual Tilt (°)", min_value=-30.0, max_value=30.0, value=float(st.session_state.get(f"cbct_angx_{selected_fdi}", 0.0)), step=1.0,
            key=f"cbct_angx_{selected_fdi}", help="Tilt toward buccal (+) or lingual (-)"
        )
    with col_p4:
        ang_y = st.slider(
            "Mesiodistal Tilt (°)", min_value=-30.0, max_value=30.0, value=float(st.session_state.get(f"cbct_angy_{selected_fdi}", 0.0)), step=1.0,
            key=f"cbct_angy_{selected_fdi}", help="Tilt toward distal (+) or mesial (-)"
        )

    with st.expander("⚙️ Fine-Tune 3D Position Offsets (X, Y, Z in mm)", expanded=False):
        c_off1, c_off2, c_off3, c_off4 = st.columns(4)
        with c_off1:
            off_x = st.slider("X Offset (mm)", -8.0, 8.0, float(st.session_state.get(f"cbct_offx_{selected_fdi}", 0.0)), 0.5, key=f"cbct_offx_{selected_fdi}")
        with c_off2:
            off_y = st.slider("Y Offset (mm)", -8.0, 8.0, float(st.session_state.get(f"cbct_offy_{selected_fdi}", 0.0)), 0.5, key=f"cbct_offy_{selected_fdi}")
        with c_off3:
            off_z = st.slider("Z Depth Offset (mm)", -8.0, 8.0, float(st.session_state.get(f"cbct_offz_{selected_fdi}", 0.0)), 0.5, key=f"cbct_offz_{selected_fdi}")
        with c_off4:
            safety_thresh = st.number_input(
                "Nerve Safety Margin (mm)", min_value=1.0, max_value=4.0, value=2.0, step=0.5,
                key="cbct_nerve_thresh", help="Minimum required clearance from inferior alveolar canal"
            )

    # ── 3. Real-Time Spatial Calculation & Safety Engine ──────────────────────
    offset_vox = np.array([
        off_x / max(spacing[0], 0.01),
        off_y / max(spacing[1], 0.01),
        off_z / max(spacing[2], 0.01),
    ])
    curr_center_vox = base_center_vox + offset_vox

    # Accurate 3D measurements at positioned location
    site_meas = get_site_measurements(
        seg_vol, curr_center_vox, selected_fdi, detected_teeth=detected_teeth, voxel_spacing_mm=spacing
    )
    bone_h = site_meas["bone_height_mm"]
    ridge_w = site_meas["ridge_width_mm"]
    md_space = site_meas["mesiodistal_mm"]

    # Sample implant points and execute multi-structure safety analysis
    implant_points = sample_implant_cylinder_points(
        curr_center_vox, imp_diam, imp_len,
        angulation_deg=(ang_x, ang_y, 0.0),
        voxel_spacing_mm=spacing,
    )

    safety_analysis = analyze_implant_safety(
        implant_points, seg_vol, selected_fdi,
        detected_teeth=detected_teeth,
        planning_threshold_mm=safety_thresh,
        voxel_spacing_mm=spacing,
    )

    canal_dist_mm = safety_analysis["canal_distance_mm"]
    overall_status = safety_analysis["overall_status"]
    overall_badge = safety_analysis["overall_badge"]
    overall_color = safety_analysis["overall_color"]
    checklist = safety_analysis["checklist"]

    # ── 4. Real-Time Metrics & Safety Status ──────────────────────────────────
    st.markdown("#### 📏 4. Real-Time Physical 3D Measurements & Safety Clearance")
    
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    m_c1.metric("Mesiodistal Space", f"~ {md_space:.1f} mm")
    m_c2.metric("Ridge Width", f"~ {ridge_w:.1f} mm")
    m_c3.metric("Available Bone Height", f"~ {bone_h:.1f} mm")
    
    if is_mandibular:
        if canal_dist_mm is not None:
            m_c4.metric("Canal Clearance", f"{canal_dist_mm:.1f} mm", delta=f"{canal_dist_mm - safety_thresh:+.1f} mm vs thresh")
        else:
            m_c4.metric("Canal Clearance", "Not Detected in ROI")
    else:
        m_c4.metric("Canal Clearance", "N/A (Maxillary Arch)")

    # Safety Checklist Strip
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1.5px solid {overall_color};border-radius:8px;padding:12px 16px;margin:10px 0 16px;box-shadow:0 1px 4px rgba(0,0,0,0.03);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #F1F5F9;">
        <span style="font-weight:800;font-size:13px;color:#0F3B7A;">3D Anatomical Clearance &amp; Boundary Verification</span>
        <span style="font-weight:800;font-size:12px;color:{overall_color};">{overall_badge}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:8px;font-size:11.5px;">
        {"".join([
            f"<div style='background:#F8FAFC;padding:6px 10px;border-radius:5px;border-left:3px solid {'#16A34A' if c['status']=='PASS' else ('#D97706' if c['status']=='CAUTION' else '#DC2626')};'>"
            f"<b>{c['item']}:</b> {c['measured_mm'] if c['measured_mm'] is not None else 'N/A'} mm &middot; {c['badge']}</div>"
            for c in checklist
        ])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5. Synchronized 2D Cross-Sectional MPR Views ─────────────────────────
    st.markdown("#### 🖼 5. Synchronized Cross-Sectional Planning Slices (MPR)")
    st.markdown(
        "<p style='color:#64748B;font-size:12px;margin:0 0 8px 0;'>"
        "Orthogonal views centered at the proposed implant coronal center. "
        "Cyan outline indicates the virtual implant geometry; red highlights the mandibular nerve canal.</p>",
        unsafe_allow_html=True
    )
    fig_mpr_imp = render_2d_mpr_implant_views(
        cbct_vol, seg_vol, curr_center_vox, imp_diam, imp_len,
        voxel_spacing_mm=spacing, canal_distance_mm=canal_dist_mm,
        bone_height_mm=bone_h, mesiodistal_mm=md_space,
    )
    st.pyplot(fig_mpr_imp, use_container_width=True)

    # ── 6. 3D Interactive Viewport with Layer Controls & Camera Presets ───────
    st.markdown("#### 🌐 6. 3D Interactive Viewport & Anatomical Layers")

    vp_col1, vp_col2 = st.columns([1.2, 2.8])
    with vp_col1:
        st.markdown("##### ⚙️ Layer Controls")
        t_imp      = st.checkbox("🔩 Virtual Implant", value=True, key="chk_3d_imp")
        t_adj_only = st.checkbox("🦷 Adjacent Teeth Only", value=True, key="chk_3d_adj_only")
        t_all_t    = st.checkbox("🦷 All 32 Teeth", value=not t_adj_only, key="chk_3d_all_t")
        t_max      = st.checkbox("🦴 Maxilla Bone", value=not is_mandibular, key="chk_3d_max")
        t_mand     = st.checkbox("🦴 Mandible Bone", value=is_mandibular, key="chk_3d_mand")
        t_can      = st.checkbox("🔴 Mandibular Canal", value=is_mandibular, key="chk_3d_can")

        st.markdown("##### 📷 Camera Presets")
        cam_p1, cam_p2 = st.columns(2)
        with cam_p1:
            if st.button("👄 Buccal", key="cam_buccal", use_container_width=True):
                st.session_state["cbct_cam"] = "buccal"
            if st.button("🦷 Occlusal", key="cam_occlusal", use_container_width=True):
                st.session_state["cbct_cam"] = "occlusal"
            if st.button("⬅ Mesial", key="cam_mesial", use_container_width=True):
                st.session_state["cbct_cam"] = "mesial"
        with cam_p2:
            if st.button("👅 Lingual", key="cam_lingual", use_container_width=True):
                st.session_state["cbct_cam"] = "lingual"
            if st.button("🔍 3D Orbit", key="cam_orbit", use_container_width=True):
                st.session_state["cbct_cam"] = "perspective"
            if st.button("➡ Distal", key="cam_distal", use_container_width=True):
                st.session_state["cbct_cam"] = "distal"

        curr_cam = st.session_state.get("cbct_cam", "perspective")

    with vp_col2:
        tab_3d_full, tab_3d_roi = st.tabs(["🌐 Full Dental Arch View", "🔍 Local Site ROI (Zoomed)"])
        with tab_3d_full:
            fig_3d_scene = build_3d_implant_scene(
                seg_vol, curr_center_vox, imp_diam, imp_len,
                angulation_deg=(ang_x, ang_y, 0.0),
                selected_fdi_tooth=selected_fdi,
                detected_teeth=detected_teeth,
                voxel_spacing_mm=spacing,
                show_maxilla=t_max, show_mandible=t_mand,
                show_teeth=(t_adj_only or t_all_t),
                show_adjacent_only=t_adj_only,
                show_canal=t_can, show_implant=t_imp,
                camera_view=curr_cam,
            )
            st.plotly_chart(fig_3d_scene, use_container_width=True)

        with tab_3d_roi:
            fig_3d_roi = build_local_roi_3d_scene(
                seg_vol, curr_center_vox, imp_diam, imp_len,
                angulation_deg=(ang_x, ang_y, 0.0),
                voxel_spacing_mm=spacing,
                roi_radius_mm=22.0,
            )
            st.plotly_chart(fig_3d_roi, use_container_width=True)

    # ── 7. Dentist Review Dossier & Clinical Disclaimer ──────────────────────
    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("#### 📋 7. Dentist Review Dossier & Clinical Assessment")

    plan_data = generate_implant_planning_data(
        case_name=case_name,
        selected_fdi_tooth=selected_fdi,
        detected_teeth=detected_teeth,
        potential_missing_teeth=detect_potential_missing_teeth(seg_info),
        bone_height_mm=bone_h,
        ridge_width_mm=ridge_w,
        mesiodistal_mm=md_space,
        implant_diameter_mm=imp_diam,
        implant_length_mm=imp_len,
        implant_angulation_deg=(ang_x, ang_y, 0.0),
        implant_position_offset_mm=(off_x, off_y, off_z),
        canal_distance_mm=canal_dist_mm,
        mesial_root_clearance_mm=safety_analysis["mesial_root_clearance_mm"],
        distal_root_clearance_mm=safety_analysis["distal_root_clearance_mm"],
        mesial_fdi=safety_analysis["mesial_adjacent_fdi"],
        distal_fdi=safety_analysis["distal_adjacent_fdi"],
        buccal_plate_mm=safety_analysis["buccal_plate_clearance_mm"],
        lingual_plate_mm=safety_analysis["lingual_plate_clearance_mm"],
        is_contained=safety_analysis["is_contained"],
        overall_status=overall_status,
        checklist=checklist,
        planning_threshold_mm=safety_thresh,
        voxel_spacing_mm=spacing,
    )

    html_dossier = generate_printable_html_report(plan_data)

    st.markdown(f"""
    <div style="background:#FFFFFF;border:2px solid #CBD5E1;border-radius:12px;padding:20px 22px;box-shadow:0 2px 6px rgba(15,23,42,0.04);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #E2E8F0;">
        <div>
          <div style="font-size:16px;font-weight:800;color:#0F3B7A;">
            3D CBCT Pre-Surgical Assessment — FDI Site {selected_fdi}
          </div>
          <div style="font-size:12px;color:#64748B;margin-top:2px;">
            {arch_str} &middot; Case {case_name} &middot; <span style="color:{overall_color};font-weight:700;">{overall_status}</span>
          </div>
        </div>
        <div style="background:#F1F5F9;border:1.5px solid {overall_color};border-radius:6px;padding:4px 12px;font-size:11px;font-weight:800;color:{overall_color};">
          {overall_badge}
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;font-size:12px;">
        <div style="background:#F8FAFC;padding:12px;border-radius:8px;border:1px solid #E2E8F0;">
          <div style="font-weight:700;color:#64748B;text-transform:uppercase;font-size:10px;margin-bottom:6px;">Site Coordinates</div>
          <div><b>Target FDI:</b> {selected_fdi}</div>
          <div><b>Arch:</b> {arch_str}</div>
          <div><b>Mesial:</b> FDI {safety_analysis['mesial_adjacent_fdi'] or 'None'}</div>
          <div><b>Distal:</b> FDI {safety_analysis['distal_adjacent_fdi'] or 'None'}</div>
        </div>
        <div style="background:#F8FAFC;padding:12px;border-radius:8px;border:1px solid #E2E8F0;">
          <div style="font-weight:700;color:#64748B;text-transform:uppercase;font-size:10px;margin-bottom:6px;">3D Measurements</div>
          <div><b>MD Space:</b> ~ {md_space:.1f} mm</div>
          <div><b>Ridge Width:</b> ~ {ridge_w:.1f} mm</div>
          <div><b>Bone Height:</b> ~ {bone_h:.1f} mm</div>
          <div><b>Canal Dist:</b> {f'{canal_dist_mm:.1f} mm' if canal_dist_mm is not None else 'N/A'}</div>
        </div>
        <div style="background:#F8FAFC;padding:12px;border-radius:8px;border:1px solid #E2E8F0;">
          <div style="font-weight:700;color:#64748B;text-transform:uppercase;font-size:10px;margin-bottom:6px;">Virtual Implant Specs</div>
          <div><b>Diameter:</b> Ø {imp_diam:.1f} mm</div>
          <div><b>Length:</b> {imp_len:.0f} mm</div>
          <div><b>Buccolingual:</b> {ang_x:+.0f}°</div>
          <div><b>Mesiodistal:</b> {ang_y:+.0f}°</div>
        </div>
      </div>
      <div style="background:#FFFBEB;border:2px solid #F59E0B;border-radius:10px;padding:16px 18px;margin-top:16px;color:#92400E;">
        <div style="font-weight:900;font-size:13px;display:flex;align-items:center;gap:6px;margin-bottom:6px;">
          <span>⚠️</span> CLINICAL DISCLAIMER &amp; DECISION SUPPORT NOTICE
        </div>
        <p style="font-size:12px;line-height:1.55;font-weight:600;margin:0;">
          {CLINICAL_DISCLAIMER_3D}
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 8. Multi-Format Exports ───────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### ⬇ 8. Export 3D Planning Deliverables")

    stl_path = str(OUTPUTS_DIR / f"{case_name}_implant_FDI_{selected_fdi}.stl")
    export_virtual_implant_stl(
        curr_center_vox, imp_diam, imp_len,
        angulation_deg=(ang_x, ang_y, 0.0),
        voxel_spacing_mm=spacing,
        output_path=stl_path,
    )

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if Path(stl_path).exists():
            with open(stl_path, "rb") as f_stl:
                st.download_button(
                    f"⬇ Download Virtual Implant STL (FDI {selected_fdi})",
                    data=f_stl.read(),
                    file_name=f"{case_name}_implant_FDI_{selected_fdi}.stl",
                    mime="application/octet-stream",
                    key=f"dl_stl_imp_{selected_fdi}",
                    use_container_width=True,
                )

    with ex2:
        st.download_button(
            f"⬇ Download Planning JSON (FDI {selected_fdi})",
            data=plan_data_to_json_str(plan_data),
            file_name=f"{case_name}_implant_plan_FDI_{selected_fdi}.json",
            mime="application/json",
            key=f"dl_json_imp_{selected_fdi}",
            use_container_width=True,
        )

    with ex3:
        st.download_button(
            f"⬇ Download Printable HTML Dossier (FDI {selected_fdi})",
            data=html_dossier,
            file_name=f"{case_name}_implant_dossier_FDI_{selected_fdi}.html",
            mime="text/html",
            key=f"dl_html_imp_{selected_fdi}",
            use_container_width=True,
        )



# ─────────────────────────────────────────────────────────────────────────────
# Panoramic X-ray Segmentation Page
# ─────────────────────────────────────────────────────────────────────────────
PANO_MODEL_PATH = MODELS_DIR / "panoramic_teeth_seg_best.pth"

@st.cache_resource(show_spinner=False)
def _load_panoramic_model():
    """Load and cache the trained U-Net++ panoramic segmentation model."""
    try:
        import torch
        import segmentation_models_pytorch as smp
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation=None,
        )
        ckpt = torch.load(str(PANO_MODEL_PATH), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, device, None
    except Exception as e:
        return None, None, str(e)


def run_panoramic_inference(img_bytes, threshold=0.5):
    """Run teeth segmentation on uploaded panoramic X-ray bytes. Returns (orig_bgr, mask, overlay)."""
    import cv2
    import numpy as np
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    model, device, err = _load_panoramic_model()
    if err:
        return None, None, None, err

    # Decode image
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None, None, "Could not decode image — please upload a valid JPG/PNG."

    orig_h, orig_w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    transform = A.Compose([
        A.Resize(height=256, width=512, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])
    aug    = transform(image=img_gray, mask=np.zeros((orig_h, orig_w), np.uint8))
    tensor = aug["image"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

    # Resize mask back to original size
    pred_mask = (prob > threshold).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Green overlay
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[pred_mask > 0] = [0, 220, 80]
    overlay = cv2.addWeighted(overlay, 0.65, green, 0.35, 0)

    # Coverage stats
    teeth_pct = (pred_mask > 0).sum() / (orig_h * orig_w) * 100

    return img_bgr, pred_mask, overlay, teeth_pct


def render_panoramic_page():
    """Panoramic X-ray 2D teeth segmentation UI."""
    import cv2
    import numpy as np

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="title-card">
      <div style="display:flex;align-items:center;gap:10px;">
        <h1 class="title-main">Panoramic X-ray Teeth Segmentation</h1>
        <span class="title-badge">2D · U-Net++</span>
      </div>
      <p class="title-sub">Upload a panoramic dental X-ray (OPG) to automatically segment all teeth regions &nbsp;•&nbsp; ResNet34 Encoder &nbsp;•&nbsp; Val Dice 0.90</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ──────────────────────────────────────────────────────────
    if not PANO_MODEL_PATH.exists():
        st.error(
            f"⛔ **Panoramic model not found.**\n\n"
            f"Expected: `models/panoramic_teeth_seg_best.pth`\n\n"
            f"Train first with:\n```\npython train_panoramic_seg.py --img_h 256 --img_w 512 --epochs 50\n```"
        )
        return

    st.markdown("""
    <div style="background:#ECFDF5;border:1px solid #A7F3D0;border-radius:8px;padding:10px 14px;
                 display:flex;align-items:center;gap:10px;margin-bottom:18px;">
      <span style="font-size:18px;">✅</span>
      <div>
        <div style="font-size:12px;font-weight:700;color:#047857;">Model Loaded — panoramic_teeth_seg_best.pth</div>
        <div style="font-size:11px;color:#065F46;margin-top:2px;">U-Net++ · ResNet34 · Trained on 329 OPG images · Val Dice 0.90</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Panoramic X-ray")
    uploaded = st.file_uploader(
        "Choose a panoramic dental X-ray image",
        type=["jpg", "jpeg", "png"],
        key="pano_upload",
        help="Upload a JPG or PNG panoramic (OPG) dental X-ray image.",
    )

    if uploaded is None:
        st.markdown("""
        <div class="empty-upload-card">
          <div class="empty-icon">🦷</div>
          <div class="empty-title">Upload a Panoramic X-ray (OPG)</div>
          <p class="empty-sub">
            Drag and drop a <code>.jpg</code> or <code>.png</code> panoramic dental X-ray to segment all teeth automatically.
          </p>
          <div style="margin-top:14px;display:flex;justify-content:center;gap:16px;font-size:11px;color:var(--text-muted);">
            <span>✓ Binary Teeth Segmentation</span>
            <span>✓ Green Overlay Visualization</span>
            <span>✓ Downloadable Mask</span>
            <span>✓ GPU Accelerated</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Threshold slider ──────────────────────────────────────────────────────
    col_sl, col_info = st.columns([1, 2])
    with col_sl:
        threshold = st.slider(
            "Segmentation Threshold", min_value=0.3, max_value=0.9,
            value=0.5, step=0.05, key="pano_thresh",
            help="Higher = stricter (fewer false positives). Lower = more inclusive."
        )
    with col_info:
        st.markdown(
            f"<div style='padding-top:28px;font-size:12px;color:var(--text-secondary);'>"
            f"Sigmoid threshold <b>{threshold:.2f}</b> — pixels with confidence &gt; {threshold:.2f} are classified as <b>teeth</b>."
            "</div>", unsafe_allow_html=True
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    run_btn = st.button("▶ Run Teeth Segmentation", type="primary", key="pano_run")
    if run_btn or "pano_result" in st.session_state:
        if run_btn:
            with st.spinner("Running segmentation on GPU... (~2-5 seconds)"):
                img_bytes = uploaded.read()
                result = run_panoramic_inference(img_bytes, threshold)
                st.session_state["pano_result"]    = result
                st.session_state["pano_img_bytes"] = img_bytes
                st.session_state["pano_fname"]     = uploaded.name

        result = st.session_state.get("pano_result")
        if result is None:
            return

        img_bgr, pred_mask, overlay, teeth_pct_or_err = result

        if img_bgr is None:
            st.error(f"❌ Inference failed: {teeth_pct_or_err}")
            return

        teeth_pct = teeth_pct_or_err

        # ── Stats row ─────────────────────────────────────────────────────────
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Image Size", f"{img_bgr.shape[1]} × {img_bgr.shape[0]} px")
        c2.metric("Teeth Coverage", f"{teeth_pct:.1f}%")
        c3.metric("Threshold Used", f"{threshold:.2f}")
        c4.metric("Model Val Dice", "0.90")

        # ── Side-by-side display ───────────────────────────────────────────────
        st.markdown("### 🔬 Segmentation Results")
        col1, col2, col3 = st.columns(3)

        # Convert BGR → RGB for display
        img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay,  cv2.COLOR_BGR2RGB)

        with col1:
            st.markdown("**Original X-ray**")
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.markdown("**Teeth Mask** (white = teeth)")
            st.image(pred_mask, use_container_width=True, clamp=True)

        with col3:
            st.markdown("**Overlay** (green = teeth)")
            st.image(overlay_rgb, use_container_width=True)

        # ── Download buttons ─────────────────────────────────────────────────
        st.markdown("### ⬇ Download Results")
        stem = Path(st.session_state.get("pano_fname", "result")).stem

        _, mask_enc = cv2.imencode(".png", pred_mask)
        _, overlay_enc = cv2.imencode(".png", overlay)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="⬇ Download Mask (PNG)",
                data=mask_enc.tobytes(),
                file_name=f"{stem}_teeth_mask.png",
                mime="image/png",
                key="dl_mask",
            )
        with dl2:
            st.download_button(
                label="⬇ Download Overlay (PNG)",
                data=overlay_enc.tobytes(),
                file_name=f"{stem}_teeth_overlay.png",
                mime="image/png",
                key="dl_overlay",
            )

        # ── Launch Planner Button ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0F3B7A 0%,#1E40AF 100%);border-radius:10px;
                    padding:1rem 1.4rem;margin-bottom:1.2rem;">
          <h3 style="color:#FFFFFF;margin:0 0 4px 0;font-size:1.05rem;">🎯 2D Dental Implant Planning Workstation</h3>
          <p style="color:#BFDBFE;font-size:0.82rem;margin:0;">
            Use this high-accuracy teeth segmentation to detect individual tooth bounding boxes,
            anatomical FDI numbers, edentulous missing gaps, and preliminary virtual implant overlays.
          </p>
        </div>
        """, unsafe_allow_html=True)

        col_plan_btn, col_plan_txt = st.columns([1, 2])
        with col_plan_btn:
            if st.button("🚀 Launch 2D Implant Planning Workstation", type="primary", key="btn_open_planner", use_container_width=True):
                from utils.panoramic_planner import run_full_planning
                with st.spinner("Analyzing tooth instances, FDI numbers & edentulous spaces..."):
                    pl_res = run_full_planning(img_bgr, pred_mask, opg_width_mm=150.0, min_gap_mm=4.5)
                    st.session_state["planner_result"] = pl_res
                    st.session_state["planner_raw"] = (img_bgr, pred_mask, f"{stem}.png")
                st.success("✅ Tooth analysis & implant plans generated! Switch to the **🎯 2D Implant Planner** tab above to interact with your plan.")
        with col_plan_txt:
            st.markdown(
                "<div style='font-size:12px;color:var(--text-secondary);padding-top:6px;'>"
                "<b>Next Step:</b> Switch to the <b>🎯 2D Implant Planner</b> tab at the top of the screen to view all individual tooth bounding boxes, 32-tooth status table, and candidate virtual implant overlays."
                "</div>", unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2D Panoramic Implant Planner Page
# ─────────────────────────────────────────────────────────────────────────────
def render_2d_planner_page():
    """
    2D Panoramic Dental Implant Planner — guided step-panel workflow.
    Features a clinical 4-quadrant FDI Odontogram chart for the 32 Permanent Teeth Status
    following the diagnostic workflow:
    Panoramic X-Ray → Tooth Segmentation → FDI Numbering → 32-Tooth Chart (Present/Missing/Uncertain)
    → Site Selection → Edentulous Space Measurement (43 ↔ 44 ↔ 45) → Landmarks (IAC/Sinus)
    → Virtual Implant → 2D Clearance Analysis → Risk/Warning → Dentist Review.
    """
    import io as _io
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from utils.panoramic_planner import (
        run_full_planning, render_planning_figure, render_debug_figure,
        render_site_detail_figure, generate_site_report, plans_to_dict_list,
        ImplantPlan,
    )

    # ── Page header ───────────────────────────────────────────────
    st.markdown("""
<div class="title-card">
  <div style="display:flex;align-items:center;gap:10px;">
    <h1 class="title-main">🎯 2D Panoramic Tooth Detection &amp; Implant Planning Workstation</h1>
    <span class="title-badge">2D · Dual AI (YOLO + U-Net++)</span>
  </div>
  <p class="title-sub">
    Individual tooth detection &middot; Anatomical FDI numbering &middot; 32-Tooth Present/Missing chart &middot;
    Edentulous space measurement &middot; 2D Clearance analysis &middot; Preliminary virtual implant positioning.
  </p>
  <p style="color:#DC2626;font-size:11.5px;margin:6px 0 0 0;font-weight:600;">
    ⚠ PRELIMINARY ONLY — All measurements are approximate (panoramic magnification/distortion). CBCT required before surgery.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Model status check ─────────────────────────────────────────
    if not PANO_MODEL_PATH.exists():
        st.error(
            "⛔ **Panoramic model not found.** "
            f"Expected `models/panoramic_teeth_seg_best.pth`.\n\n"
            "Train first with: `python train_panoramic_seg.py`"
        )
        return

    yolo_model_file = Path("models/yolov8_teeth_fdi_best.pt")
    if yolo_model_file.exists():
        st.markdown("""
        <div style="background:#ECFDF5;border:1px solid #A7F3D0;border-radius:8px;
                    padding:8px 14px;display:flex;align-items:center;gap:10px;margin-bottom:16px;">
          <span style="font-size:16px;">✅</span>
          <div style="font-size:12px;font-weight:700;color:#047857;">
            Dual AI Engine Ready — YOLOv8 32-Tooth FDI Numbering (Val mAP 97.2%) + U-Net++ ResNet34 Segmentation
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 1: Upload + Segmentation ─────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">1</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Upload Panoramic X-Ray &amp; Segmentation Settings</div>
    </div>
    """, unsafe_allow_html=True)

    has_session_pano = ("pano_result" in st.session_state and st.session_state["pano_result"] is not None)
    if has_session_pano:
        pano_fname = st.session_state.get("pano_fname", "Current X-ray")
        st.markdown(
            f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;"
            f"padding:8px 12px;margin-bottom:10px;font-size:12px;color:#1E40AF;'>"
            f"⚡ <b>Active segmentation detected:</b> <code>{pano_fname}</code>. "
            f"Click <b>▶ Run Analysis</b> to process this image, or upload a new one below."
            f"</div>",
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Choose a panoramic dental X-ray (or use active segmentation)",
        type=["jpg", "jpeg", "png"],
        key="planner_upload",
        help="JPG or PNG panoramic (OPG) dental X-ray.",
    )

    if uploaded is None and not has_session_pano:
        st.markdown("""
        <div class="empty-upload-card">
          <div class="empty-icon">🦷</div>
          <div class="empty-title">Upload a Panoramic X-ray (OPG)</div>
          <p class="empty-sub">
            Drag and drop a <code>.jpg</code> or <code>.png</code> panoramic dental X-ray to initiate tooth instance detection and FDI analysis.
          </p>
          <div style="margin-top:14px;display:flex;justify-content:center;gap:16px;font-size:11px;color:var(--text-muted);">
            <span>✓ U-Net++ Teeth Segmentation</span>
            <span>✓ YOLOv8 FDI Numbering</span>
            <span>✓ 32-Tooth FDI Chart</span>
            <span>✓ Virtual Implant Planning</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Settings row ─────────────────────────────────────────────
    c_t, c_w, c_g, c_btn = st.columns([1, 1, 1, 1])
    with c_t:
        pl_thresh = st.slider("Seg. Threshold", 0.3, 0.9, 0.5, 0.05, key="pl_thresh")
    with c_w:
        pl_opg_mm = st.number_input("OPG Width (mm)", 80.0, 220.0, 150.0, 5.0, key="pl_opg_mm")
        pl_calibrated = pl_opg_mm != 150.0
    with c_g:
        pl_min_gap = st.slider("Min Gap (mm)", 3.0, 10.0, 4.5, 0.5, key="pl_min_gap")
    with c_btn:
        st.markdown("<div style='padding-top:22px;'></div>", unsafe_allow_html=True)
        pl_run_btn = st.button("▶ Run Analysis", type="primary", key="pl_run", use_container_width=True)

    # ── Run segmentation + planning ──────────────────────────────────
    pl_key = "planner_result"
    pl_raw_key = "planner_raw"

    if pl_run_btn:
        if uploaded is not None:
            img_bytes = uploaded.read()
            with st.spinner("🔍 Segmenting teeth in panoramic image..."):
                raw = run_panoramic_inference(img_bytes, pl_thresh)
            img_bgr, pred_mask, _, _ = raw
            fname_used = uploaded.name
        else:
            img_bgr, pred_mask, _, _ = st.session_state["pano_result"]
            fname_used = st.session_state.get("pano_fname", "panoramic_xray.png")

        if img_bgr is None:
            st.error("❌ Segmentation failed.")
            return

        with st.spinner("🦷 Performing tooth instance detection, FDI numbering & gap analysis..."):
            pl_result = run_full_planning(
                img_bgr, pred_mask,
                opg_width_mm=pl_opg_mm,
                is_calibrated=pl_calibrated,
                min_gap_mm=pl_min_gap,
            )
        st.session_state[pl_key]     = pl_result
        st.session_state[pl_raw_key] = (img_bgr, pred_mask, fname_used)
        for plan in pl_result.implant_plans:
            fdi = plan.site_fdi
            if f"pl_diam_{fdi}" not in st.session_state:
                st.session_state[f"pl_diam_{fdi}"] = plan.diameter_mm
                st.session_state[f"pl_len_{fdi}"]  = plan.length_mm
                st.session_state[f"pl_ang_{fdi}"]  = plan.angulation_deg
                st.session_state[f"pl_xoff_{fdi}"] = 0.0
                st.session_state[f"pl_yoff_{fdi}"] = 0.0

    pl_result = st.session_state.get(pl_key)
    pl_raw    = st.session_state.get(pl_raw_key)
    if pl_result is None or pl_raw is None:
        st.info("Configure settings above and click **▶ Run Analysis** to begin.")
        return

    img_bgr, pred_mask, fname = pl_raw
    stem = Path(fname).stem
    h, w = img_bgr.shape[:2]
    px_mm = pl_result.px_per_mm

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 2: 32 Permanent Teeth Status (FDI Chart) ────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">2</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">🦷 Complete 32 Permanent Teeth Status (FDI Odontogram Chart)</div>
    </div>
    """, unsafe_allow_html=True)

    present_count   = sum(1 for e in pl_result.tooth_status_table if e.status in ("Existing tooth", "Present", "Possibly non-restorable"))
    missing_count   = sum(1 for e in pl_result.tooth_status_table if e.status == "Missing")
    uncertain_count = sum(1 for e in pl_result.tooth_status_table if "Uncertain" in e.status)

    # Metrics strip
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🦷 Present Teeth",  f"{present_count} / 32")
    m2.metric("❌ Missing Teeth",  missing_count)
    m3.metric("⚠ Uncertain",       uncertain_count)
    m4.metric("🔩 Implant Sites",  len(pl_result.implant_plans))
    m5.metric("📏 Scale", f"{pl_opg_mm:.0f} mm" + (" (cal.)" if pl_calibrated else " (approx)"))

    # Status mapping & tooth anatomical names
    status_by_fdi = {e.fdi: e for e in pl_result.tooth_status_table}

    TOOTH_SHORT_NAMES = {
        18: "M3", 17: "M2", 16: "M1", 15: "P2", 14: "P1", 13: "C", 12: "I2", 11: "I1",
        21: "I1", 22: "I2", 23: "C", 24: "P1", 25: "P2", 26: "M1", 27: "M2", 28: "M3",
        48: "M3", 47: "M2", 46: "M1", 45: "P2", 44: "P1", 43: "C", 42: "I2", 41: "I1",
        31: "I1", 32: "I2", 33: "C", 34: "P1", 35: "P2", 36: "M1", 37: "M2", 38: "M3",
    }

    def render_tooth_cell(fdi):
        e = status_by_fdi.get(fdi)
        short_type = TOOTH_SHORT_NAMES.get(fdi, "")
        
        if e is None:
            bg_col = "#F8FAFC"
            border_col = "#CBD5E1"
            txt_col = "#64748B"
            badge_bg = "#E2E8F0"
            badge_txt = "#475569"
            status_text = "Unknown"
            icon = "⚪"
            width_str = "—"
        elif e.status == "Missing":
            bg_col = "#FEF2F2"
            border_col = "#EF4444"
            txt_col = "#991B1B"
            badge_bg = "#FEE2E2"
            badge_txt = "#DC2626"
            status_text = "Missing"
            icon = "❌"
            width_str = f"~{e.available_space_mm:.1f}mm" if e.available_space_mm else "Space"
        elif "non-restorable" in e.status or "Possibly" in e.status:
            bg_col = "#FFFBEB"
            border_col = "#F59E0B"
            txt_col = "#92400E"
            badge_bg = "#FEF3C7"
            badge_txt = "#D97706"
            status_text = "Compromised"
            icon = "⚠️"
            width_str = f"~{e.available_space_mm:.1f}mm" if e.available_space_mm else "Root"
        elif "Uncertain" in e.status:
            bg_col = "#F8FAFC"
            border_col = "#94A3B8"
            txt_col = "#475569"
            badge_bg = "#F1F5F9"
            badge_txt = "#64748B"
            status_text = "Uncertain"
            icon = "⚪"
            width_str = "N/A"
        else: # Existing tooth
            bg_col = "#F0FDF4"
            border_col = "#10B981"
            txt_col = "#166534"
            badge_bg = "#DCFCE7"
            badge_txt = "#15803D"
            status_text = "Present"
            icon = "✅"
            width_str = f"~{e.available_space_mm:.1f}mm" if e.available_space_mm else "Intact"

        is_implant_site = any(p.site_fdi == fdi for p in pl_result.implant_plans)
        site_ring = "box-shadow: 0 0 0 2px #2563EB, 0 2px 4px rgba(37,99,235,0.2);" if is_implant_site else ""
        candidate_tag = "<div style='font-size:8px;font-weight:800;color:#1E40AF;background:#DBEAFE;border-radius:2px;padding:1px 3px;margin-top:2px;'>PLAN SITE</div>" if is_implant_site else ""

        return (
            f"<div style='flex:1;min-width:42px;max-width:64px;background:{bg_col};border:1.5px solid {border_col};"
            f"border-radius:6px;padding:5px 2px;text-align:center;display:flex;flex-direction:column;"
            f"align-items:center;justify-content:space-between;min-height:84px;{site_ring}'>"
            f"<div style='font-size:9px;font-weight:700;color:#64748B;'>{short_type}</div>"
            f"<div style='font-size:13.5px;font-weight:800;color:{txt_col};margin:1px 0;'>{fdi}</div>"
            f"<div style='font-size:8px;background:{badge_bg};color:{badge_txt};padding:1px 3px;border-radius:4px;font-weight:700;white-space:nowrap;'>"
            f"{icon} {status_text}</div>"
            f"<div style='font-size:8px;color:#64748B;margin-top:2px;'>{width_str}</div>"
            f"{candidate_tag}</div>"
        )

    # Quadrant tooth sequences (Midline in center)
    q1_teeth = [18, 17, 16, 15, 14, 13, 12, 11]  # Right to Midline
    q2_teeth = [21, 22, 23, 24, 25, 26, 27, 28]  # Midline to Left
    q4_teeth = [48, 47, 46, 45, 44, 43, 42, 41]  # Right to Midline
    q3_teeth = [31, 32, 33, 34, 35, 36, 37, 38]  # Midline to Left

    q1_html = "".join([render_tooth_cell(t) for t in q1_teeth])
    q2_html = "".join([render_tooth_cell(t) for t in q2_teeth])
    q4_html = "".join([render_tooth_cell(t) for t in q4_teeth])
    q3_html = "".join([render_tooth_cell(t) for t in q3_teeth])

    odontogram_html = (
        f"<div style='background:#FFFFFF;border:1px solid #CBD5E1;border-radius:12px;padding:16px 18px;margin-bottom:16px;box-shadow:0 1px 4px rgba(15,23,42,0.04);'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #F1F5F9;'>"
        f"<div style='font-size:11px;font-weight:800;color:#0F3B7A;text-transform:uppercase;letter-spacing:0.5px;'>PATIENT RIGHT (VIEWER LEFT)</div>"
        f"<div style='font-size:12px;font-weight:800;color:#1E293B;background:#F1F5F9;padding:3px 12px;border-radius:9999px;'>FDI TWO-DIGIT ODONTOGRAM CHART</div>"
        f"<div style='font-size:11px;font-weight:800;color:#0F3B7A;text-transform:uppercase;letter-spacing:0.5px;'>PATIENT LEFT (VIEWER RIGHT)</div>"
        f"</div>"
        f"<div style='margin-bottom:8px;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:10.5px;font-weight:700;color:#475569;margin-bottom:6px;'>"
        f"<span>Q1: Maxillary Right (Upper Right)</span>"
        f"<span style='color:#0F3B7A;font-weight:800;'>▲ MAXILLARY ARCH (UPPER JAW) ▲</span>"
        f"<span>Q2: Maxillary Left (Upper Left)</span>"
        f"</div>"
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<div style='flex:1;display:flex;gap:3px;justify-content:flex-end;'>{q1_html}</div>"
        f"<div style='width:2px;height:90px;background:#2563EB;flex-shrink:0;position:relative;' title='Dental Midline'>"
        f"<div style='position:absolute;top:-10px;left:-10px;font-size:7px;font-weight:800;color:#2563EB;white-space:nowrap;'>MIDLINE</div>"
        f"</div>"
        f"<div style='flex:1;display:flex;gap:3px;justify-content:flex-start;'>{q2_html}</div>"
        f"</div>"
        f"</div>"
        f"<div style='display:flex;align-items:center;margin:12px 0 10px 0;'>"
        f"<div style='flex:1;height:1.5px;background:#E2E8F0;'></div>"
        f"<div style='font-size:9px;font-weight:800;color:#64748B;padding:0 12px;text-transform:uppercase;letter-spacing:1px;'>OCCLUSAL PLANE</div>"
        f"<div style='flex:1;height:1.5px;background:#E2E8F0;'></div>"
        f"</div>"
        f"<div>"
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<div style='flex:1;display:flex;gap:3px;justify-content:flex-end;'>{q4_html}</div>"
        f"<div style='width:2px;height:90px;background:#2563EB;flex-shrink:0;' title='Dental Midline'></div>"
        f"<div style='flex:1;display:flex;gap:3px;justify-content:flex-start;'>{q3_html}</div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:10.5px;font-weight:700;color:#475569;margin-top:6px;'>"
        f"<span>Q4: Mandibular Right (Lower Right)</span>"
        f"<span style='color:#0F3B7A;font-weight:800;'>▼ MANDIBULAR ARCH (LOWER JAW) ▼</span>"
        f"<span>Q3: Mandibular Left (Lower Left)</span>"
        f"</div>"
        f"</div>"
        f"<div style='display:flex;gap:18px;font-size:11px;color:#334155;flex-wrap:wrap;margin-top:14px;padding-top:10px;border-top:1px solid #F1F5F9;justify-content:center;'>"
        f"<span><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:#DCFCE7;border:1.5px solid #10B981;vertical-align:middle;margin-right:4px;'></span><b>Present</b> (Healthy)</span>"
        f"<span><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:#FEE2E2;border:1.5px solid #EF4444;vertical-align:middle;margin-right:4px;'></span><b>Missing</b> (Edentulous Space)</span>"
        f"<span><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:#FEF3C7;border:1.5px solid #F59E0B;vertical-align:middle;margin-right:4px;'></span><b>Compromised</b> (Non-Restorable)</span>"
        f"<span><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:#F1F5F9;border:1.5px dashed #94A3B8;vertical-align:middle;margin-right:4px;'></span><b>Uncertain</b> (Impacted/Absent)</span>"
        f"<span><span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:#DBEAFE;border:2px solid #2563EB;vertical-align:middle;margin-right:4px;'></span><b>Candidate Site</b> (Blue Border)</span>"
        f"</div>"
        f"</div>"
    )
    st.markdown(odontogram_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 3: Select Missing Tooth Site ────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">3</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Select Target Missing Tooth Site for Planning</div>
    </div>
    """, unsafe_allow_html=True)

    if not pl_result.implant_plans:
        st.success("✅ No edentulous missing spaces detected. All visible teeth appear present.")
        return

    site_options = [p.site_fdi for p in pl_result.implant_plans]
    col_site_sel, col_site_info = st.columns([1, 2])
    with col_site_sel:
        selected_site = st.selectbox(
            "Select Candidate Site (FDI)",
            options=site_options,
            format_func=lambda f: f"FDI {f} — {next((p.space.arch_region for p in pl_result.implant_plans if p.site_fdi==f), '')}",
            key="pl_site_sel",
        )
    with col_site_info:
        sel_plan_info = next((p for p in pl_result.implant_plans if p.site_fdi == selected_site), None)
        if sel_plan_info:
            is_upper_label = "Upper (Maxilla)" if sel_plan_info.is_upper else "Lower (Mandible)"
            adj_l = sel_plan_info.adjacent_left_fdi or "—"
            adj_r = sel_plan_info.adjacent_right_fdi or "—"
            st.markdown(
                f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;"
                f"padding:10px 14px;font-size:12px;color:#1E40AF;margin-top:4px;'>"
                f"<b>Target Site: FDI {selected_site}</b> · {is_upper_label} · {sel_plan_info.space.arch_region}<br>"
                f"<span style='color:#334155;'>Anatomical Sequence: <b>FDI {adj_l}</b> ←→ <b>GAP (Site {selected_site})</b> ←→ <b>FDI {adj_r}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    sel_plan = next((p for p in pl_result.implant_plans if p.site_fdi == selected_site), None)
    if not sel_plan:
        return

    fdi = sel_plan.site_fdi

    # Collect user params from session state for all candidate sites
    user_params = {}
    for plan in pl_result.implant_plans:
        _fdi = plan.site_fdi
        user_params[_fdi] = {
            "diameter_mm":    st.session_state.get(f"pl_diam_{_fdi}", plan.diameter_mm),
            "length_mm":      st.session_state.get(f"pl_len_{_fdi}",  plan.length_mm),
            "angulation_deg": st.session_state.get(f"pl_ang_{_fdi}",  plan.angulation_deg),
            "x_offset_px":    st.session_state.get(f"pl_xoff_{_fdi}", 0.0),
            "y_offset_px":    st.session_state.get(f"pl_yoff_{_fdi}", 0.0),
        }

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 4: Edentulous Space Measurement (43 ↔ 44 ↔ 45) ─────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">4</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Edentulous Space Measurement &amp; Neighboring Boundaries</div>
    </div>
    """, unsafe_allow_html=True)

    space     = sel_plan.space
    adj_l_fdi = sel_plan.adjacent_left_fdi
    adj_r_fdi = sel_plan.adjacent_right_fdi
    gap_mm    = sel_plan.mesiodistal_mm
    vert_h    = sel_plan.vertical_height_mm

    # Neighboring teeth widths
    adj_l_tooth = next((t for t in pl_result.detected_teeth if t.fdi == adj_l_fdi), None) if adj_l_fdi else None
    adj_r_tooth = next((t for t in pl_result.detected_teeth if t.fdi == adj_r_fdi), None) if adj_r_fdi else None
    adj_l_mm = round(adj_l_tooth.width_mm, 1) if adj_l_tooth else None
    adj_r_mm = round(adj_r_tooth.width_mm, 1) if adj_r_tooth else None

    # Gap visualization bar proportions
    total_span_mm = (adj_l_mm or 7.0) + gap_mm + (adj_r_mm or 7.0)
    pct_l   = round(((adj_l_mm or 7.0) / total_span_mm) * 100, 1)
    pct_gap = round((gap_mm / total_span_mm) * 100, 1)
    pct_r   = round(((adj_r_mm or 7.0) / total_span_mm) * 100, 1)

    gap_color = "#DC2626" if gap_mm < 6.0 else ("#F59E0B" if gap_mm < 8.0 else "#16A34A")

    adj_l_label = f"FDI {adj_l_fdi}" if adj_l_fdi else "Edge"
    adj_r_label = f"FDI {adj_r_fdi}" if adj_r_fdi else "Edge"
    adj_l_mm_str = f"{adj_l_mm:.1f} mm" if adj_l_mm else "?"
    adj_r_mm_str = f"{adj_r_mm:.1f} mm" if adj_r_mm else "?"

    col_space, col_vert = st.columns([3, 1])
    with col_space:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px 18px;">
          <div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:12px;">Neighboring Teeth Sequence &amp; Mesiodistal Gap</div>
          <!-- Neighbor label row -->
          <div style="display:flex;align-items:flex-end;gap:0;margin-bottom:4px;font-size:11px;font-weight:600;">
            <div style="width:{pct_l}%;text-align:center;color:#166534;">{adj_l_label}</div>
            <div style="width:{pct_gap}%;text-align:center;color:{gap_color};">FDI {fdi} (SITE)</div>
            <div style="width:{pct_r}%;text-align:center;color:#166534;">{adj_r_label}</div>
          </div>
          <!-- Bar -->
          <div style="display:flex;border-radius:6px;overflow:hidden;height:36px;border:1px solid #CBD5E1;">
            <div style="width:{pct_l}%;background:#DCFCE7;display:flex;align-items:center;
                         justify-content:center;font-size:10.5px;font-weight:700;color:#166534;">
              {adj_l_mm_str}
            </div>
            <div style="width:{pct_gap}%;background:{gap_color};display:flex;align-items:center;
                         justify-content:center;font-size:11.5px;font-weight:800;color:#FFFFFF;">
              ← {gap_mm:.1f} mm Space →
            </div>
            <div style="width:{pct_r}%;background:#DCFCE7;display:flex;align-items:center;
                         justify-content:center;font-size:10.5px;font-weight:700;color:#166534;">
              {adj_r_mm_str}
            </div>
          </div>
          <!-- Sub-label row -->
          <div style="display:flex;gap:0;margin-top:6px;font-size:10px;color:#94A3B8;">
            <div style="width:{pct_l}%;text-align:center;">Adjacent Left</div>
            <div style="width:{pct_gap}%;text-align:center;">Edentulous Candidate Site</div>
            <div style="width:{pct_r}%;text-align:center;">Adjacent Right</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_vert:
        vert_color = "#DC2626" if (vert_h is not None and vert_h < 8.0) else (
                      "#F59E0B" if (vert_h is not None and vert_h < 12.0) else "#16A34A")
        vert_label = f"{vert_h:.1f} mm" if vert_h is not None else "N/A"
        vert_note  = "Limited" if (vert_h is not None and vert_h < 8.0) else (
                      "Moderate" if (vert_h is not None and vert_h < 12.0) else (
                      "Sufficient" if vert_h is not None else "Not detected"))
        landmark_name = "Sinus Floor" if sel_plan.is_upper else "IAC"
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                    padding:16px 14px;height:100%;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:8px;">Vertical Bone Height</div>
          <div style="font-size:24px;font-weight:800;color:{vert_color};">{vert_label}</div>
          <div style="font-size:10px;font-weight:700;color:{vert_color};margin:2px 0;">{vert_note}</div>
          <div style="font-size:10px;color:#94A3B8;margin-top:4px;">
            Crest → {landmark_name}<br>
            <span style="font-style:italic;">(2D projection)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 5: Anatomical Landmarks Status (IAC / Sinus) ─────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">5</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Anatomical Landmarks (IAC / Maxillary Sinus)</div>
    </div>
    """, unsafe_allow_html=True)

    lmk_cols = st.columns(len(pl_result.landmarks) or 1)
    for idx, lmk in enumerate(pl_result.landmarks):
        icon_bg  = "#DCFCE7" if lmk.detected else "#FEE2E2"
        icon_brd = "#86EFAC" if lmk.detected else "#FCA5A5"
        icon_col = "#166534" if lmk.detected else "#991B1B"
        icon_sym = "✓" if lmk.detected else "✗"
        conf_col = "#047857" if lmk.detected else "#9A3412"
        with lmk_cols[idx % len(lmk_cols)]:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;
                        padding:12px 14px;margin-bottom:8px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="background:{icon_bg};border:1px solid {icon_brd};color:{icon_col};
                             border-radius:50%;width:22px;height:22px;display:flex;align-items:center;
                             justify-content:center;font-size:12px;font-weight:800;flex-shrink:0;">{icon_sym}</div>
                <div style="font-size:12px;font-weight:700;color:#1E293B;">{lmk.name}</div>
              </div>
              <div style="font-size:10.5px;color:{conf_col};font-weight:600;margin-bottom:3px;">{lmk.confidence}</div>
              <div style="font-size:10px;color:#64748B;line-height:1.4;">{lmk.note[:120]}{'…' if len(lmk.note) > 120 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 6: Virtual Implant Configuration (Diameter / Length / Angle) ────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">6</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Virtual Implant Overlay — Diameter · Length · Angle</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:11.5px;color:#1E40AF;background:#EFF6FF;border:1px solid #BFDBFE;"
        f"border-radius:6px;padding:6px 12px;margin-bottom:10px;'>"
        f"AI Suggested: <b>{sel_plan.suggested_diameter_range}</b> diameter · "
        f"<b>{sel_plan.suggested_length_range}</b> length · Site FDI <b>{fdi}</b> ({sel_plan.space.arch_region})"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_imp, col_fig = st.columns([1, 3])
    with col_imp:
        new_diam = st.slider(
            "🔵 Diameter (mm)", 3.0, 6.0,
            float(st.session_state.get(f"pl_diam_{fdi}", sel_plan.diameter_mm)),
            0.1, key=f"pl_diam_sl_{fdi}",
        )
        st.session_state[f"pl_diam_{fdi}"] = new_diam

        new_len = st.slider(
            "↕ Length (mm)", 6.0, 16.0,
            float(st.session_state.get(f"pl_len_{fdi}", sel_plan.length_mm)),
            0.5, key=f"pl_len_sl_{fdi}",
        )
        st.session_state[f"pl_len_{fdi}"] = new_len

        new_ang = st.slider(
            "↪ Angulation (°)", -30.0, 30.0,
            float(st.session_state.get(f"pl_ang_{fdi}", sel_plan.angulation_deg)),
            1.0, key=f"pl_ang_sl_{fdi}",
        )
        st.session_state[f"pl_ang_{fdi}"] = new_ang

        max_off = max(int(space.width_px * 0.45), 20)
        new_xoff = st.slider(
            "↔ H. Offset (px)", -max_off, max_off,
            int(st.session_state.get(f"pl_xoff_{fdi}", 0.0)),
            2, key=f"pl_xoff_sl_{fdi}",
        )
        st.session_state[f"pl_xoff_{fdi}"] = float(new_xoff)

        new_yoff = st.slider(
            "↕ V. Offset (px)", -40, 40,
            int(st.session_state.get(f"pl_yoff_{fdi}", 0.0)),
            2, key=f"pl_yoff_sl_{fdi}",
        )
        st.session_state[f"pl_yoff_{fdi}"] = float(new_yoff)

        user_params[fdi] = {
            "diameter_mm":    new_diam,
            "length_mm":      new_len,
            "angulation_deg": new_ang,
            "x_offset_px":    float(new_xoff),
            "y_offset_px":    float(new_yoff),
        }

        if st.button("🔄 Reset to Suggested", key=f"pl_reset_{fdi}", use_container_width=True):
            for k in (f"pl_diam_{fdi}", f"pl_len_{fdi}", f"pl_ang_{fdi}", f"pl_xoff_{fdi}", f"pl_yoff_{fdi}"):
                st.session_state.pop(k, None)
            st.rerun()

    with col_fig:
        show_debug = st.checkbox("🛠 Debug Mode (arch curves & coordinates)", False, key="pl_tog_debug_n")
        tog1, tog2, tog3, tog4, tog5 = st.columns(5)
        show_teeth = tog1.checkbox("🦷 Boxes",     True, key="pl_tog_teeth_v")
        show_miss  = tog2.checkbox("❌ Missing",   True, key="pl_tog_miss_v")
        show_meas  = tog3.checkbox("📏 Measures",  True, key="pl_tog_meas_v")
        show_lmk   = tog4.checkbox("🔬 Landmarks", True, key="pl_tog_lmk_v")
        show_imp   = tog5.checkbox("🔩 Implant",   True, key="pl_tog_imp_v")

        if show_debug:
            fig_debug = render_debug_figure(pl_result, img_bgr, figsize=(22, 9))
            st.pyplot(fig_debug, use_container_width=True)
            plt.close(fig_debug)
        else:
            fig_overview = render_planning_figure(
                pl_result, img_bgr,
                show_teeth_boxes=show_teeth,
                show_missing_boxes=show_miss,
                show_measurements=show_meas,
                show_implants=show_imp,
                show_landmarks=show_lmk,
                user_params=user_params,
                figsize=(22, 9),
            )
            st.pyplot(fig_overview, use_container_width=True)
            plt.close(fig_overview)

    # Site detail zoom
    with st.expander(f"🔍 Site Detail Inspection Zoom — FDI {fdi}", expanded=False):
        fig_det = render_site_detail_figure(pl_result, img_bgr, fdi, user_params=user_params)
        if fig_det:
            st.pyplot(fig_det, use_container_width=True)
            plt.close(fig_det)
            st.markdown("""
            <div style="font-size:11px;color:#64748B;margin-top:6px;">
              <b>Legend:</b>&nbsp;
              <span style="color:#2ECC71;">&#9633;</span> Adjacent Tooth (Intact) &nbsp;·&nbsp;
              <span style="color:#F39C12;">&#9633;</span> Possibly Non-Restorable &nbsp;·&nbsp;
              <span style="color:#E74C3C;">- -</span> Edentulous Space &nbsp;·&nbsp;
              <span style="color:#F39C12;">&#9135;</span> Mesiodistal Space &nbsp;·&nbsp;
              <span style="color:#00E5FF;">&#9711;</span> Candidate Implant Overlay
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 7: 2D Clearance Analysis ────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">7</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">2D Clearance Analysis (Horizontal &amp; Vertical Safety Margins)</div>
    </div>
    """, unsafe_allow_html=True)

    cur_diam    = float(user_params[fdi]["diameter_mm"])
    cur_len     = float(user_params[fdi]["length_mm"])
    clearance_l = max(0.0, round((gap_mm / 2.0) - (cur_diam / 2.0), 2))
    clearance_r = clearance_l

    def _clearance_color(c_mm):
        if c_mm < 0.5:   return "#DC2626", "#FEF2F2", "#FCA5A5"
        if c_mm < 1.5:   return "#D97706", "#FFFBEB", "#FDE68A"
        return "#16A34A", "#F0FDF4", "#86EFAC"

    cl_col, cl_bg, cl_brd = _clearance_color(clearance_l)
    cr_col, cr_bg, cr_brd = _clearance_color(clearance_r)

    col_horiz, col_vert2 = st.columns([3, 1])
    with col_horiz:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px 18px;">
          <div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:12px;">Horizontal (Mesiodistal) Interdental Clearance</div>

          <div style="display:flex;align-items:center;gap:0;border-radius:6px;overflow:hidden;
                      height:42px;border:1px solid #CBD5E1;margin-bottom:8px;">
            <div style="flex:1;background:{cl_bg};display:flex;align-items:center;justify-content:center;
                         font-size:11px;font-weight:700;color:{cl_col};">
              ← {clearance_l:.1f} mm
            </div>
            <div style="background:#1E40AF;color:#fff;padding:0 14px;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;white-space:nowrap;
                         flex-shrink:0;min-width:80px;">
              Ø {cur_diam:.1f} mm
            </div>
            <div style="flex:1;background:{cr_bg};display:flex;align-items:center;justify-content:center;
                         font-size:11px;font-weight:700;color:{cr_col};">
              {clearance_r:.1f} mm →
            </div>
          </div>

          <div style="display:flex;gap:0;font-size:10px;color:#94A3B8;text-align:center;">
            <div style="flex:1;">← FDI {adj_l_label} clearance</div>
            <div style="flex:0 0 80px;">Implant</div>
            <div style="flex:1;">FDI {adj_r_label} clearance →</div>
          </div>

          <div style="display:flex;gap:12px;margin-top:10px;font-size:11px;">
            <div style="background:{cl_bg};border:1px solid {cl_brd};color:{cl_col};
                         border-radius:6px;padding:4px 10px;font-weight:700;">
              Left: {clearance_l:.1f} mm {'✓' if clearance_l >= 1.5 else '⚠' if clearance_l >= 0.5 else '🚨'}
            </div>
            <div style="background:{cr_bg};border:1px solid {cr_brd};color:{cr_col};
                         border-radius:6px;padding:4px 10px;font-weight:700;">
              Right: {clearance_r:.1f} mm {'✓' if clearance_r >= 1.5 else '⚠' if clearance_r >= 0.5 else '🚨'}
            </div>
            <div style="background:#EFF6FF;border:1px solid #BFDBFE;color:#1E40AF;
                         border-radius:6px;padding:4px 10px;font-weight:700;">
              Gap: {gap_mm:.1f} mm total
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_vert2:
        vert_clearance = round((vert_h or 0.0) - cur_len, 1)
        vc_col, vc_bg, vc_brd = _clearance_color(vert_clearance if vert_h else 99)
        if vert_h is not None:
            vc_pct_implant = min(85, round((cur_len / vert_h) * 100, 1))
            vc_pct_clear   = max(5, 100 - vc_pct_implant)
            vc_label = f"{vert_clearance:.1f} mm"
            vc_sym   = "✓" if vert_clearance >= 2.0 else ("⚠" if vert_clearance >= 0.0 else "🚨")
        else:
            vc_pct_implant = 75
            vc_pct_clear   = 25
            vc_label = "N/A"
            vc_sym   = "—"
            vc_col   = "#64748B"
            vc_bg    = "#F8FAFC"
            vc_brd   = "#CBD5E1"

        landmark_label = "Sinus Floor" if sel_plan.is_upper else "IAC"
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:14px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:8px;">Vertical Clearance</div>

          <div style="display:flex;flex-direction:column;border-radius:6px;overflow:hidden;
                      height:120px;border:1px solid #CBD5E1;margin-bottom:8px;">
            <div style="flex:{vc_pct_implant};background:#1E40AF;display:flex;align-items:center;
                         justify-content:center;font-size:10px;font-weight:700;color:#fff;">
              {cur_len:.0f} mm implant
            </div>
            <div style="flex:{vc_pct_clear};background:{vc_bg};display:flex;align-items:center;
                         justify-content:center;font-size:10px;font-weight:700;color:{vc_col};">
              {vc_label}
            </div>
          </div>
          <div style="font-size:10px;color:#94A3B8;margin-top:2px;">→ {landmark_label}</div>
          <div style="font-size:13px;font-weight:800;color:{vc_col};margin-top:2px;">{vc_sym}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 8: Risk / Warning Assessment ────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">8</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Risk / Warning Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    has_red  = any("🚨" in n or "Limited" in n or clearance_l < 0.5 or clearance_r < 0.5
                   or (vert_h is not None and vert_clearance < 0) for n in sel_plan.clinical_reasons)
    has_warn = any("⚠" in n for n in sel_plan.clinical_reasons) or clearance_l < 1.5 or clearance_r < 1.5

    overall_risk = "HIGH" if has_red else ("MODERATE" if has_warn else "LOW")
    risk_colors  = {
        "HIGH":     ("#991B1B", "#FEF2F2", "#FCA5A5"),
        "MODERATE": ("#92400E", "#FFFBEB", "#FCD34D"),
        "LOW":      ("#166534", "#F0FDF4", "#86EFAC"),
    }
    risk_icons   = {"HIGH": "🚨", "MODERATE": "⚠️", "LOW": "✅"}
    rc, rbg, rbrd = risk_colors[overall_risk]

    st.markdown(f"""
    <div style="background:{rbg};border:2px solid {rbrd};border-radius:10px;padding:12px 16px;
                margin-bottom:14px;display:flex;align-items:center;gap:12px;">
      <div style="font-size:28px;">{risk_icons[overall_risk]}</div>
      <div>
        <div style="font-size:13px;font-weight:800;color:{rc};">
          {overall_risk} RISK — FDI {fdi} ({sel_plan.space.arch_region})
        </div>
        <div style="font-size:11.5px;color:{rc};margin-top:2px;">
          {'Significant concerns identified — CBCT mandatory before any surgical planning.' if overall_risk == 'HIGH'
           else 'Moderate concerns — clinical evaluation and CBCT recommended.'
           if overall_risk == 'MODERATE'
           else 'No critical concerns from 2D analysis — CBCT still required before surgery.'}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for note in sel_plan.clinical_reasons:
        if "🚨" in note or "Limited" in note:
            bg, brd, txt = "#FEF2F2", "#FCA5A5", "#991B1B"
            icon = "🚨"
        elif "⚠" in note:
            bg, brd, txt = "#FFFBEB", "#FCD34D", "#92400E"
            icon = "⚠️"
        elif "CLINICAL REQUIREMENT" in note or "CBCT" in note:
            bg, brd, txt = "#EFF6FF", "#BFDBFE", "#1E40AF"
            icon = "📋"
        else:
            bg, brd, txt = "#F0FDF4", "#86EFAC", "#166534"
            icon = "ℹ️"
        st.markdown(
            f"<div style='font-size:11.5px;padding:7px 12px;margin:4px 0;"
            f"background:{bg};border-left:4px solid {brd};color:{txt};"
            f"border-radius:4px;'>{icon} {note}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:1.2rem 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)

    # ── STEP 9: Dentist Review Dossier & Clinical Warning ─────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="background:#0F3B7A;color:#fff;font-size:11px;font-weight:800;
                  border-radius:50%;width:24px;height:24px;display:flex;align-items:center;
                  justify-content:center;flex-shrink:0;">9</div>
      <div style="font-size:14px;font-weight:700;color:#0F3B7A;">Dentist Review Dossier &amp; Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    adj_l_name = f"FDI {adj_l_fdi}" if adj_l_fdi else "N/A"
    adj_r_name = f"FDI {adj_r_fdi}" if adj_r_fdi else "N/A"
    arch_label = "Maxillary (Upper)" if sel_plan.is_upper else "Mandibular (Lower)"

    cur_diam = float(user_params[fdi]["diameter_mm"])
    cur_len  = float(user_params[fdi]["length_mm"])
    cur_ang  = float(user_params[fdi]["angulation_deg"])

    # Clinical status label without aggressive wording
    tier_label = "FAVORABLE 2D CANDIDATE" if overall_risk == "LOW" else ("EVALUATION REQUIRED" if overall_risk == "MODERATE" else "HIGH CAUTION / COMPROMISED")

    review_card_html = (
        f"<div style='background:#FFFFFF;border:2px solid #CBD5E1;border-radius:12px;padding:20px 22px;box-shadow:0 2px 6px rgba(15,23,42,0.04);'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #E2E8F0;'>"
        f"<div>"
        f"<div style='font-size:16px;font-weight:800;color:#0F3B7A;'>📋 Dentist Review &amp; Pre-Surgical Assessment — Site FDI {fdi}</div>"
        f"<div style='font-size:12px;color:#64748B;margin-top:3px;'>{arch_label} Arch &middot; {sel_plan.space.arch_region} &middot; <span style='color:{rc};font-weight:700;'>{tier_label}</span></div>"
        f"</div>"
        f"<div style='background:{rbg};border:1.5px solid {rbrd};border-radius:6px;padding:5px 12px;font-size:11px;font-weight:800;color:{rc};'>{risk_icons[overall_risk]} {tier_label}</div>"
        f"</div>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:18px;'>"
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px;'>"
        f"<div style='font-size:10px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>Anatomical Boundaries</div>"
        f"<table style='width:100%;font-size:11.5px;border-collapse:collapse;'>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Target Site</td><td style='font-weight:700;color:#0F3B7A;'>FDI {fdi}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Arch Sequence</td><td style='font-weight:600;'>{arch_label}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Region</td><td style='font-weight:600;'>{sel_plan.space.arch_region}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Adjacent Left</td><td style='font-weight:600;'>{adj_l_name}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Adjacent Right</td><td style='font-weight:600;'>{adj_r_name}</td></tr>"
        f"</table>"
        f"</div>"
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px;'>"
        f"<div style='font-size:10px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>2D Space &amp; Clearance</div>"
        f"<table style='width:100%;font-size:11.5px;border-collapse:collapse;'>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Mesiodistal Space</td><td style='font-weight:700;color:#1E40AF;'>~ {gap_mm:.1f} mm</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Vertical Height</td><td style='font-weight:700;color:{'#16A34A' if (vert_h and vert_h >= 12) else '#D97706' if vert_h else '#64748B'};'>{'~ ' + str(vert_h) + ' mm' if vert_h else 'Not detected'}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Clearance (Left)</td><td style='font-weight:700;color:{cl_col};'>~ {clearance_l:.1f} mm</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Clearance (Right)</td><td style='font-weight:700;color:{cr_col};'>~ {clearance_r:.1f} mm</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Clearance (Apex)</td><td style='font-weight:700;color:{vc_col};'>{'~ ' + str(vert_clearance) + ' mm' if vert_h else 'N/A'}</td></tr>"
        f"</table>"
        f"</div>"
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px;'>"
        f"<div style='font-size:10px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>Candidate Virtual Implant</div>"
        f"<table style='width:100%;font-size:11.5px;border-collapse:collapse;'>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Diameter</td><td style='font-weight:700;color:#0F3B7A;'>Ø {cur_diam:.1f} mm</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Length</td><td style='font-weight:700;color:#0F3B7A;'>{cur_len:.0f} mm</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Angulation</td><td style='font-weight:700;'>{cur_ang:+.0f}°</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Suggested Ø</td><td style='font-weight:600;color:#64748B;'>{sel_plan.suggested_diameter_range}</td></tr>"
        f"<tr><td style='color:#64748B;padding:3px 0;'>Suggested L</td><td style='font-weight:600;color:#64748B;'>{sel_plan.suggested_length_range}</td></tr>"
        f"</table>"
        f"</div>"
        f"</div>"
        f"<div style='background:#FFFBEB;border:2px solid #F59E0B;border-radius:10px;padding:18px 20px;margin-top:14px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
        f"<span style='font-size:18px;'>⚠️</span>"
        f"<span style='font-size:13.5px;font-weight:900;color:#92400E;letter-spacing:0.3px;'>CBCT REQUIRED FOR FINAL IMPLANT PLANNING</span>"
        f"</div>"
        f"<p style='font-size:12px;color:#78350F;line-height:1.55;font-weight:600;margin:0 0 10px 0;'>"
        f"These results are intended only as AI-assisted preliminary decision support from a 2D panoramic radiograph. "
        f"Final implant selection, positioning, and surgical planning must be performed by a qualified dental professional "
        f"using appropriate clinical examination and 3D imaging such as CBCT."
        f"</p>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:11px;color:#92400E;padding-top:8px;border-top:1px solid #FDE68A;'>"
        f"<div><b>• Buccolingual Ridge Width:</b> Cannot be determined from 2D panoramic radiograph.</div>"
        f"<div><b>• Nerve &amp; Sinus Boundaries:</b> Volumetric CBCT mapping required for true 3D canal proximity.</div>"
        f"<div><b>• Optical Distortion:</b> Variable panoramic magnification requires 3D calibrated verification.</div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(review_card_html, unsafe_allow_html=True)

    # ── Export Controls ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.markdown("#### ⬇ Export Planning Dossier")

    export_plans = []
    for plan in pl_result.implant_plans:
        p = user_params.get(plan.site_fdi, {})
        export_plans.append(ImplantPlan(
            site_fdi=plan.site_fdi,
            status_category=getattr(plan, "status_category", "Missing"),
            adjacent_left_fdi=plan.adjacent_left_fdi,
            adjacent_right_fdi=plan.adjacent_right_fdi,
            center_x_px=plan.center_x_px,
            center_y_px=plan.center_y_px,
            x_offset_px=float(p.get("x_offset_px", plan.x_offset_px)),
            y_offset_px=float(p.get("y_offset_px", plan.y_offset_px)),
            diameter_mm=float(p.get("diameter_mm", plan.diameter_mm)),
            length_mm=float(p.get("length_mm",   plan.length_mm)),
            angulation_deg=float(p.get("angulation_deg", plan.angulation_deg)),
            is_upper=plan.is_upper,
            suggested_diameter_range=plan.suggested_diameter_range,
            suggested_length_range=plan.suggested_length_range,
            mesiodistal_mm=plan.mesiodistal_mm,
            vertical_height_mm=plan.vertical_height_mm,
            planning_confidence=plan.planning_confidence,
            clinical_reasons=getattr(plan, "clinical_reasons", []),
            space=plan.space,
        ))

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        with st.spinner("Rendering export figure..."):
            fig_exp = render_planning_figure(
                pl_result, img_bgr,
                show_teeth_boxes=True, show_missing_boxes=True,
                show_measurements=True, show_landmarks=True, show_implants=True,
                user_params=user_params, figsize=(24, 10),
            )
            buf = _io.BytesIO()
            fig_exp.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#0d1117")
            buf.seek(0)
            plt.close(fig_exp)
        st.download_button(
            "⬇ Download Annotated Image (PNG)",
            data=buf.getvalue(),
            file_name=f"{stem}_implant_plan.png",
            mime="image/png",
            key="pl_dl_img",
            use_container_width=True,
        )

    with ex2:
        plan_json = json.dumps(
            {
                "image_file":        str(fname),
                "opg_width_mm":      float(pl_opg_mm),
                "is_calibrated":     bool(pl_calibrated),
                "px_per_mm":         round(float(pl_result.px_per_mm), 4),
                "selected_site_fdi": int(fdi),
                "tooth_status_table": [
                    {
                        "fdi": int(e.fdi),
                        "arch_region": str(e.arch_region),
                        "detected": bool(e.detected),
                        "status": str(e.status),
                        "confidence": float(e.confidence),
                        "bbox": [int(x) for x in e.bbox] if e.bbox else None,
                        "reason": str(e.reason),
                    }
                    for e in pl_result.tooth_status_table
                ],
                "implant_plans":  plans_to_dict_list(export_plans),
                "planning_status": "PRELIMINARY — DENTIST REVIEW REQUIRED",
                "disclaimer": (
                    "CBCT REQUIRED FOR FINAL IMPLANT PLANNING: "
                    "These results are intended only as AI-assisted preliminary decision support from a 2D panoramic radiograph. "
                    "Final implant selection, positioning, and surgical planning must be performed by a qualified dental professional "
                    "using appropriate clinical examination and 3D imaging such as CBCT."
                ),
            },
            indent=2,
            default=_json_default,
        )
        st.download_button(
            "⬇ Download Planning Report (JSON)",
            data=plan_json,
            file_name=f"{stem}_implant_report.json",
            mime="application/json",
            key="pl_dl_json",
            use_container_width=True,
        )

    with ex3:
        if export_plans:
            full_text_report = (
                "PANORAMIC TOOTH DETECTION & IMPLANT PLANNING REPORT\n"
                f"Image: {fname}\n"
                f"OPG Width: {pl_opg_mm:.0f} mm ({'calibrated' if pl_calibrated else 'uncalibrated/default'})\n"
                "\n"
            )
            for plan in export_plans:
                full_text_report += generate_site_report(plan, pl_result)
            st.download_button(
                "⬇ Download Text Report (TXT)",
                data=full_text_report,
                file_name=f"{stem}_implant_report.txt",
                mime="text/plain",
                key="pl_dl_txt",
                use_container_width=True,
            )

    # ── Advanced: 32-tooth status full data table ────────────────────────────
    with st.expander("📋 Detailed 32-Tooth Numerical Data Table (Advanced)", expanded=False):
        status_rows = []
        for entry in pl_result.tooth_status_table:
            if entry.status == "Existing tooth":
                det_icon = "✅"
                badge = "Existing tooth"
            elif entry.status == "Missing":
                det_icon = "❌"
                badge = "Missing (Edentulous space)"
            elif entry.status == "Possibly non-restorable":
                det_icon = "⚠"
                badge = "Possibly non-restorable (Compromised)"
            else:
                det_icon = "⚪"
                badge = "Uncertain (Clinical evaluation required)"

            bbox_str  = f"({entry.bbox[0]}, {entry.bbox[1]}, {entry.bbox[2]}, {entry.bbox[3]})" if entry.bbox else "—"
            space_str = f"~{entry.available_space_mm:.1f} mm (approx)" if entry.available_space_mm else "—"

            status_rows.append({
                "FDI": entry.fdi,
                "Arch Region": entry.arch_region,
                "Detection": det_icon,
                "Status": badge,
                "Confidence": f"{entry.confidence:.0%}",
                "Available Space": space_str,
                "Implant Consideration": entry.implant_consideration,
                "Clinical Considerations / Details": entry.reason,
            })

        df_status = pd.DataFrame(status_rows)
        st.dataframe(df_status, use_container_width=True, hide_index=True)
        st.caption(
            "💡 Status: Existing tooth (healthy/present), Missing (edentulous space), "
            "Possibly non-restorable (severely compromised), or Uncertain. "
            "Measurements are 2D projected approximations."
        )



# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_header()

    # ── Mode tabs ────────────────────────────────────────────────────────────
    tab_cbct, tab_pano, tab_planner = st.tabs([
        "🦴  CBCT 3D Segmentation",
        "🦷  Panoramic 2D Segmentation",
        "🎯  2D Implant Planner",
    ])

    with tab_cbct:
        # Check model exists before doing anything
        if not CHECKPOINT_PATH.exists():
            st.error(
                f"⛔ **OralSeg model checkpoint not found.**\n\n"
                f"Expected: `models/model_workstation39.pt`\n\n"
                f"Please place your checkpoint file at:\n```\n{CHECKPOINT_PATH}\n```"
            )
        else:
            # ── Upload section ───────────────────────────────────────────────
            uploaded_file = render_upload_section()

            if uploaded_file is None:
                st.markdown("""
                <div class="empty-upload-card">
                  <div class="empty-icon">🦷</div>
                  <div class="empty-title">
                    Upload CBCT Dataset (DICOM ZIP or NIfTI)
                  </div>
                  <p class="empty-sub">
                    Drag and drop a <code>.zip</code> (CBCT DICOM slices) or <code>.nii</code> / <code>.nii.gz</code> (3D NIfTI volume) to initiate anatomical segmentation.
                  </p>
                  <div style="margin-top:14px;display:flex;justify-content:center;gap:16px;font-size:11px;color:var(--text-muted);">
                    <span>✓ 35-Class Anatomy Engine</span>
                    <span>✓ Direct NIfTI Support</span>
                    <span>✓ High-Resolution 3D Meshes</span>
                    <span>✓ Safety Canal Measurement</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── Process upload ───────────────────────────────────────────
                upload_state = process_upload(uploaded_file)
                if upload_state is not None:
                    # ── Run segmentation ─────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 🚀 Run Segmentation")

                    col_mode, col_info_mode = st.columns([1, 2])
                    with col_mode:
                        speed_mode = st.radio(
                            "Inference Mode",
                            ["⚡ Fast Mode (overlap=0.25, ~1-2 min)", "🎯 High Precision (overlap=0.50, ~4-6 min)"],
                            index=0,
                            help="Fast Mode uses less patch overlap for much faster processing on laptop GPUs.",
                        )
                        overlap_val = 0.25 if "Fast" in speed_mode else 0.50

                    col_btn, col_info = st.columns([1, 3])
                    with col_btn:
                        run_btn = st.button("▶ Run OralSeg Segmentation", type="primary", key="run_seg")
                    with col_info:
                        st.markdown(
                            f"<p style='color:var(--text-secondary);font-size:0.85rem;padding-top:0.6rem;font-weight:500;'>"
                            f"Using <b>{'Fast Mode (~1-2 min)' if overlap_val == 0.25 else 'High Precision Mode (~4-6 min)'}</b> on GPU.</p>",
                            unsafe_allow_html=True,
                        )

                    if run_btn:
                        result = run_pipeline(upload_state, overlap=overlap_val)
                        if result is not None:
                            st.session_state["last_result"]  = result
                            st.session_state["upload_state"] = upload_state

                    # ── Show results if available ────────────────────────────
                    if "last_result" in st.session_state and st.session_state["last_result"] is not None:
                        render_results(st.session_state["last_result"])

    with tab_pano:
        render_panoramic_page()

    with tab_planner:
        render_2d_planner_page()

if __name__ == "__main__":
    main()
