"""
pdf_report.py
=============
Generates a professional clinical PDF report for the AI Dental Implant Planning System.

Uses reportlab (pure-Python, no system dependencies on Windows).
Install: pip install reportlab

Entry point:
    generate_pdf_report(plan_data: dict) -> bytes
        Returns raw PDF bytes ready to be written to a file or served via
        Streamlit's st.download_button.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

# ── Brand colours (matching app.py CSS variables) ────────────────────────────
NAVY_DARK    = colors.HexColor("#0F3B7A")
NAVY_BLUE    = colors.HexColor("#1E40AF")
NAVY_SOFT    = colors.HexColor("#2563EB")
SLATE_100    = colors.HexColor("#F1F5F9")
SLATE_200    = colors.HexColor("#E2E8F0")
SLATE_400    = colors.HexColor("#94A3B8")
SLATE_600    = colors.HexColor("#475569")
SLATE_800    = colors.HexColor("#1E293B")
GREEN_PASS   = colors.HexColor("#16A34A")
GREEN_LIGHT  = colors.HexColor("#DCFCE7")
YELLOW_WARN  = colors.HexColor("#D97706")
YELLOW_LIGHT = colors.HexColor("#FEF3C7")
RED_DANGER   = colors.HexColor("#DC2626")
RED_LIGHT    = colors.HexColor("#FEE2E2")
WHITE        = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _styles() -> Dict[str, ParagraphStyle]:
    return {
        "section": ParagraphStyle(
            "section", fontSize=10, fontName="Helvetica-Bold",
            textColor=NAVY_DARK, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontSize=8.5, fontName="Helvetica",
            textColor=SLATE_800, leading=13,
        ),
        "body_bold": ParagraphStyle(
            "body_bold", fontSize=8.5, fontName="Helvetica-Bold",
            textColor=SLATE_800, leading=13,
        ),
        "small": ParagraphStyle(
            "small", fontSize=7.5, fontName="Helvetica",
            textColor=SLATE_600, leading=11,
        ),
        "badge_text": ParagraphStyle(
            "badge_text", fontSize=12, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER, leading=16,
        ),
        "center": ParagraphStyle(
            "center", fontSize=8.5, fontName="Helvetica",
            alignment=TA_CENTER, textColor=SLATE_800,
        ),
        "footer": ParagraphStyle(
            "footer", fontSize=7, fontName="Helvetica",
            textColor=SLATE_400, alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontSize=7.5, fontName="Helvetica",
            textColor=colors.HexColor("#92400E"), leading=11,
        ),
    }


def _divider(clr=None, thickness=0.5):
    return HRFlowable(
        width="100%", thickness=thickness,
        color=clr or SLATE_200, spaceAfter=4, spaceBefore=4,
    )


def _section_header(text: str, s: Dict) -> List:
    return [Paragraph(text, s["section"]), _divider(NAVY_SOFT, 1.2)]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: key-value table
# ─────────────────────────────────────────────────────────────────────────────
def _kv_table(rows: List[tuple], col_widths=None) -> Table:
    s = _styles()
    data = [
        [Paragraph(f"<b>{k}</b>", s["body"]), Paragraph(str(v), s["body"])]
        for k, v in rows
    ]
    cw = col_widths or [65 * mm, 95 * mm]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, SLATE_100]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, SLATE_200),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Helper: status badge
# ─────────────────────────────────────────────────────────────────────────────
def _status_badge(overall_status: str, avail_w: float, s: Dict) -> Table:
    su = overall_status.upper()
    if "FEASIBLE" in su and "HIGH" not in su:
        bg, label = GREEN_PASS, f"OVERALL STATUS:  {overall_status}"
    elif "REVIEW" in su or "LIMITED" in su or "CAUTION" in su:
        bg, label = YELLOW_WARN, f"OVERALL STATUS:  {overall_status}"
    else:
        bg, label = RED_DANGER, f"OVERALL STATUS:  {overall_status}"

    t = Table([[Paragraph(label, s["badge_text"])]], colWidths=[avail_w], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bg),
        ("TOPPADDING", (0, 0), (0, 0), 9),
        ("BOTTOMPADDING", (0, 0), (0, 0), 9),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("RIGHTPADDING", (0, 0), (0, 0), 12),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safety checklist table
# ─────────────────────────────────────────────────────────────────────────────
def _checklist_table(checklist: List[Dict], s: Dict) -> Table:
    header = [
        Paragraph("<b>Anatomical Landmark</b>", s["body_bold"]),
        Paragraph("<b>Measured</b>", s["body_bold"]),
        Paragraph("<b>Threshold</b>", s["body_bold"]),
        Paragraph("<b>Status</b>", s["body_bold"]),
        Paragraph("<b>Clinical Note</b>", s["body_bold"]),
    ]
    rows = [header]
    row_styles = []

    for i, chk in enumerate(checklist):
        ri = i + 1
        meas = chk.get("measured_mm")
        meas_str = f"{meas:.1f} mm" if meas is not None else "N/A"
        thresh = chk.get("threshold_mm")
        thresh_str = f">= {thresh:.1f} mm" if thresh is not None else "-"
        status = chk.get("status", "UNKNOWN")
        icon = {"PASS": "PASS", "CAUTION": "CAUTION", "DANGER": "DANGER"}.get(status, status)

        rows.append([
            Paragraph(chk.get("item", ""), s["body"]),
            Paragraph(meas_str, s["center"]),
            Paragraph(thresh_str, s["center"]),
            Paragraph(icon, s["center"]),
            Paragraph(chk.get("note", ""), s["small"]),
        ])

        if status == "PASS":
            row_styles += [
                ("BACKGROUND", (3, ri), (3, ri), GREEN_LIGHT),
                ("TEXTCOLOR", (3, ri), (3, ri), GREEN_PASS),
                ("FONTNAME", (3, ri), (3, ri), "Helvetica-Bold"),
            ]
        elif status == "CAUTION":
            row_styles += [
                ("BACKGROUND", (3, ri), (3, ri), YELLOW_LIGHT),
                ("TEXTCOLOR", (3, ri), (3, ri), YELLOW_WARN),
                ("FONTNAME", (3, ri), (3, ri), "Helvetica-Bold"),
            ]
        elif status == "DANGER":
            row_styles += [
                ("BACKGROUND", (3, ri), (3, ri), RED_LIGHT),
                ("TEXTCOLOR", (3, ri), (3, ri), RED_DANGER),
                ("FONTNAME", (3, ri), (3, ri), "Helvetica-Bold"),
            ]

    col_w = [52*mm, 22*mm, 22*mm, 24*mm, 46*mm]
    t = Table(rows, colWidths=col_w, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_100]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, SLATE_200),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        *row_styles,
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(plan_data: Dict[str, Any]) -> bytes:
    """
    Generate a professional clinical PDF from a plan_data dict produced by
    implant_report.generate_implant_planning_data().

    Returns:
        Raw PDF bytes — pass directly to st.download_button or write to file.
    """
    buf = io.BytesIO()
    avail_w = PAGE_W - 2 * MARGIN

    fdi = plan_data.get("selected_implant_site", {}).get("fdi_tooth", "?")
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"AI Dental Implant Plan - FDI {fdi}",
        author="AI Dental Implant Planning System",
    )

    s = _styles()
    story: List = []

    # ── Extract data ──────────────────────────────────────────────────────────
    site      = plan_data.get("selected_implant_site", {})
    meas      = plan_data.get("anatomical_3d_measurements_mm", {})
    imp       = plan_data.get("virtual_implant_parameters", {})
    ang       = imp.get("angulation_degrees", {})
    offset    = imp.get("position_offset_mm", {})
    safety    = plan_data.get("safety_and_clearance_analysis", {})
    checklist = safety.get("checklist", [])
    teeth     = plan_data.get("dentition_status", {})
    timestamp = plan_data.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")
    case_id   = plan_data.get("case_id", "Unknown")
    model_arc = plan_data.get("model_architecture", "AI Model")
    overall   = safety.get("overall_status", "UNKNOWN")
    spacing   = plan_data.get("voxel_spacing_mm", [1.0, 1.0, 1.0])

    # ═════════════════════════════════════════════════════════════════════════
    # HEADER BAR
    # ═════════════════════════════════════════════════════════════════════════
    hdr_left = [
        Paragraph(
            "AI Dental Implant Planning System",
            ParagraphStyle("hl", fontSize=14, fontName="Helvetica-Bold",
                           textColor=WHITE, leading=18),
        ),
        Paragraph(
            "3D CBCT AI-Assisted Pre-Surgical Planning Dossier",
            ParagraphStyle("hl2", fontSize=8.5, fontName="Helvetica",
                           textColor=colors.HexColor("#BFDBFE"), leading=12),
        ),
    ]
    hdr_right = [
        Paragraph(
            f"Case: {case_id}",
            ParagraphStyle("hr", fontSize=8, fontName="Helvetica-Bold",
                           textColor=WHITE, alignment=TA_RIGHT),
        ),
        Paragraph(
            f"Date: {timestamp}",
            ParagraphStyle("hr2", fontSize=8, fontName="Helvetica",
                           textColor=colors.HexColor("#BFDBFE"), alignment=TA_RIGHT),
        ),
        Paragraph(
            f"Model: {model_arc}",
            ParagraphStyle("hr3", fontSize=7.5, fontName="Helvetica",
                           textColor=colors.HexColor("#93C5FD"), alignment=TA_RIGHT),
        ),
    ]
    hdr_tbl = Table(
        [[hdr_left, hdr_right]],
        colWidths=[avail_w * 0.60, avail_w * 0.40],
        hAlign="LEFT",
    )
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY_DARK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 14),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 6 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # OVERALL STATUS BADGE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_status_badge(overall, avail_w, s))
    story.append(Spacer(1, 5 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Implant Site & Dentition
    # ═════════════════════════════════════════════════════════════════════════
    story += _section_header("1.  Implant Site & Dentition Status", s)

    arch_label = "Mandibular" if int(fdi) >= 31 else "Maxillary"
    detected_str = ", ".join(str(t) for t in teeth.get("detected_teeth", [])) or "None"
    missing_str  = ", ".join(str(t) for t in teeth.get("potential_missing_teeth", [])) or "None"

    story.append(_kv_table([
        ("Target FDI Tooth",      f"FDI {fdi}  ({arch_label} arch)"),
        ("Site Status",           site.get("status", "Missing / Edentulous Site")),
        ("Mesial Adjacent (FDI)", str(site.get("adjacent_mesial_fdi") or "None")),
        ("Distal Adjacent (FDI)", str(site.get("adjacent_distal_fdi") or "None")),
        ("Detected Teeth",        f"{teeth.get('detected_count', 0)} teeth  [{detected_str}]"),
        ("Potentially Missing",   f"{teeth.get('potential_missing_count', 0)} teeth  [{missing_str}]"),
        ("Voxel Spacing (mm)",    f"[{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}]"),
    ]))
    story.append(Spacer(1, 4 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — 3D Anatomical Measurements
    # ═════════════════════════════════════════════════════════════════════════
    story += _section_header("2.  3D Anatomical Measurements (Physical mm)", s)

    def _mm(val):
        return f"~ {val:.1f} mm" if val is not None else "N/A"

    story.append(_kv_table([
        ("Mesiodistal Available Space",    _mm(meas.get("mesiodistal_available_space"))),
        ("Buccolingual Ridge Width",        _mm(meas.get("buccolingual_ridge_width"))),
        ("Available Vertical Bone Height",  _mm(meas.get("available_vertical_bone_height"))),
    ]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<i>Mesiodistal: min surface-to-surface distance between adjacent teeth. "
        "Ridge width: PCA-based buccolingual cross-section at 3 mm crest depth. "
        "Bone height: alveolar crest to mandibular canal (crest-to-canal method).</i>",
        s["small"],
    ))
    story.append(Spacer(1, 4 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Virtual Implant Parameters
    # ═════════════════════════════════════════════════════════════════════════
    story += _section_header("3.  Virtual Candidate Implant Parameters", s)

    story.append(_kv_table([
        ("Implant Diameter",         f"O {imp.get('diameter_mm', 0):.1f} mm"),
        ("Implant Length",           f"{imp.get('length_mm', 0):.0f} mm"),
        ("Buccolingual Angle (Rx)",  f"{ang.get('buccolingual_rx', 0):+.1f} deg"),
        ("Mesiodistal Angle (Ry)",   f"{ang.get('mesiodistal_ry', 0):+.1f} deg"),
        ("Axial Rotation (Rz)",      f"{ang.get('axial_rotation_rz', 0):+.1f} deg"),
        ("Position Offset X/Y/Z",    f"{offset.get('offset_x_mm',0):.1f} / "
                                      f"{offset.get('offset_y_mm',0):.1f} / "
                                      f"{offset.get('offset_z_mm',0):.1f}  mm"),
    ]))
    story.append(Spacer(1, 4 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Safety & Clearance Checklist
    # ═════════════════════════════════════════════════════════════════════════
    story += _section_header("4.  Safety & Clearance Verification Checklist", s)

    canal_mm   = safety.get("mandibular_canal_clearance_mm")
    mesial_mm  = safety.get("mesial_root_clearance_mm")
    distal_mm  = safety.get("distal_root_clearance_mm")
    buccal_mm  = safety.get("buccal_cortical_plate_mm")
    lingual_mm = safety.get("lingual_cortical_plate_mm")
    contained  = safety.get("bone_containment_preserved", False)

    story.append(_kv_table([
        ("Mandibular Canal Clearance",  _mm(canal_mm)   + "  (threshold >= 2.0 mm)"),
        ("Mesial Root Clearance",       _mm(mesial_mm)  + "  (threshold >= 1.5 mm)"),
        ("Distal Root Clearance",       _mm(distal_mm)  + "  (threshold >= 1.5 mm)"),
        ("Buccal Cortical Plate",       _mm(buccal_mm)  + "  (threshold >= 1.0 mm)"),
        ("Lingual Cortical Plate",      _mm(lingual_mm) + "  (threshold >= 1.0 mm)"),
        ("Bone Containment (>=70%)",    "Preserved" if contained else "BREACH DETECTED"),
    ]))

    story.append(Spacer(1, 4 * mm))

    if checklist:
        story.append(Paragraph("Detailed Checklist:", s["body_bold"]))
        story.append(Spacer(1, 2 * mm))
        story.append(KeepTogether(_checklist_table(checklist, s)))

    story.append(Spacer(1, 5 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Clinical Thresholds Reference
    # ═════════════════════════════════════════════════════════════════════════
    story += _section_header("5.  Clinical Threshold Reference", s)

    thresh_data = [
        [
            Paragraph("<b>Parameter</b>", s["body_bold"]),
            Paragraph("<b>Safe Threshold</b>", s["body_bold"]),
            Paragraph("<b>Caution Zone</b>", s["body_bold"]),
            Paragraph("<b>Danger Zone</b>", s["body_bold"]),
        ],
        ["Mandibular Canal",  ">= 2.0 mm",   "1.0 - 2.0 mm",  "< 1.0 mm"],
        ["Adjacent Root",     ">= 1.5 mm",   "0.8 - 1.5 mm",  "< 0.8 mm"],
        ["Cortical Plate",    ">= 1.0 mm",   "< 1.0 mm",      "-"],
        ["Bone Containment",  ">= 70%",      "50 - 70%",      "< 50%"],
    ]
    tw = [52 * mm, 36 * mm, 36 * mm, 36 * mm]
    thresh_tbl = Table(thresh_data, colWidths=tw, hAlign="LEFT")
    thresh_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_100]),
        ("GRID", (0, 0), (-1, -1), 0.4, SLATE_200),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (1, 1), (1, -1), GREEN_LIGHT),
        ("BACKGROUND", (2, 1), (2, -1), YELLOW_LIGHT),
        ("BACKGROUND", (3, 1), (3, -1), RED_LIGHT),
    ]))
    story.append(thresh_tbl)
    story.append(Spacer(1, 5 * mm))

    # ═════════════════════════════════════════════════════════════════════════
    # CLINICAL DISCLAIMER
    # ═════════════════════════════════════════════════════════════════════════
    disclaimer_text = plan_data.get(
        "disclaimer",
        "AI-assisted preliminary planning only. Final implant selection, positioning and "
        "surgical planning must be performed by a qualified dental professional based on "
        "clinical examination and appropriate 3D imaging.",
    )
    disc_inner = [
        Paragraph(
            "<b>CLINICAL DISCLAIMER</b>",
            ParagraphStyle("disc_hdr", fontSize=9, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#92400E"), spaceAfter=4),
        ),
        Paragraph(disclaimer_text, s["disclaimer"]),
    ]
    disc_tbl = Table([[disc_inner]], colWidths=[avail_w], hAlign="LEFT")
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (0, 0), 1.5, colors.HexColor("#F59E0B")),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("RIGHTPADDING", (0, 0), (0, 0), 12),
    ]))
    story.append(disc_tbl)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * mm))
    story.append(_divider())
    story.append(Paragraph(
        f"Generated by AI Dental Implant Planning System  |  {timestamp}  |  "
        f"Case: {case_id}  |  All processing performed locally - no patient data transmitted.",
        s["footer"],
    ))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: save to file
# ─────────────────────────────────────────────────────────────────────────────
def save_pdf_report(plan_data: Dict[str, Any], output_path: str) -> str:
    """Generate and save PDF to output_path. Returns the path."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf_bytes = generate_pdf_report(plan_data)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path


# Re-export extra specialized PDF generators
from utils.pdf_generators_extra import (
    generate_ds_pdf_report,
    generate_2d_seg_pdf_report,
    generate_panoramic_pdf_report,
)
