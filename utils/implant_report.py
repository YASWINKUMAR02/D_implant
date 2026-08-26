"""
implant_report.py
Generates exportable JSON, printable HTML Implant Planning Reports,
and 3D Virtual Implant STL surface meshes with structured clinical decision support metrics.
"""

from __future__ import annotations
import json
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from utils.implant_geometry import create_virtual_implant_mesh
from utils.visualization import export_binary_stl


CLINICAL_DISCLAIMER_3D = (
    "AI-assisted preliminary planning only. Final implant selection, positioning and surgical "
    "planning must be performed by a qualified dental professional based on clinical examination "
    "and appropriate 3D imaging."
)


def generate_implant_planning_data(
    case_name: str,
    selected_fdi_tooth: int,
    detected_teeth: list,
    potential_missing_teeth: list,
    bone_height_mm: float,
    ridge_width_mm: float,
    mesiodistal_mm: float,
    implant_diameter_mm: float,
    implant_length_mm: float,
    implant_angulation_deg: tuple,
    implant_position_offset_mm: tuple,
    canal_distance_mm: Optional[float],
    mesial_root_clearance_mm: Optional[float],
    distal_root_clearance_mm: Optional[float],
    mesial_fdi: Optional[int],
    distal_fdi: Optional[int],
    buccal_plate_mm: float,
    lingual_plate_mm: float,
    is_contained: bool,
    overall_status: str,
    checklist: List[Dict[str, Any]],
    planning_threshold_mm: float,
    voxel_spacing_mm: list,
    model_name: str = "OralSeg (model_workstation39.pt)",
) -> Dict[str, Any]:
    """
    Assemble the complete structured 3D implant planning dictionary.
    """
    return {
        "report_type": "3D CBCT AI-Assisted Implant Planning Dossier",
        "timestamp": datetime.now().isoformat(),
        "case_id": case_name,
        "model_architecture": model_name,
        "voxel_spacing_mm": voxel_spacing_mm,
        "selected_implant_site": {
            "fdi_tooth": selected_fdi_tooth,
            "status": "Missing / Edentulous Site",
            "adjacent_mesial_fdi": mesial_fdi,
            "adjacent_distal_fdi": distal_fdi,
        },
        "dentition_status": {
            "detected_teeth": detected_teeth,
            "detected_count": len(detected_teeth),
            "potential_missing_teeth": potential_missing_teeth,
            "potential_missing_count": len(potential_missing_teeth),
        },
        "anatomical_3d_measurements_mm": {
            "mesiodistal_available_space": mesiodistal_mm,
            "buccolingual_ridge_width": ridge_width_mm,
            "available_vertical_bone_height": bone_height_mm,
        },
        "virtual_implant_parameters": {
            "diameter_mm": implant_diameter_mm,
            "length_mm": implant_length_mm,
            "angulation_degrees": {
                "buccolingual_rx": implant_angulation_deg[0],
                "mesiodistal_ry": implant_angulation_deg[1],
                "axial_rotation_rz": implant_angulation_deg[2],
            },
            "position_offset_mm": {
                "offset_x_mm": implant_position_offset_mm[0],
                "offset_y_mm": implant_position_offset_mm[1],
                "offset_z_mm": implant_position_offset_mm[2],
            },
        },
        "safety_and_clearance_analysis": {
            "overall_status": overall_status,
            "mandibular_canal_clearance_mm": canal_distance_mm,
            "mesial_root_clearance_mm": mesial_root_clearance_mm,
            "distal_root_clearance_mm": distal_root_clearance_mm,
            "buccal_cortical_plate_mm": buccal_plate_mm,
            "lingual_cortical_plate_mm": lingual_plate_mm,
            "bone_containment_preserved": is_contained,
            "configured_canal_threshold_mm": planning_threshold_mm,
            "checklist": checklist,
        },
        "disclaimer": CLINICAL_DISCLAIMER_3D,
    }


def export_virtual_implant_stl(
    center_voxel: np.ndarray,
    diameter_mm: float,
    length_mm: float,
    angulation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    output_path: str = "implant.stl",
) -> Optional[str]:
    """
    Export the virtual implant 3D geometry as a standard binary STL file in physical mm.
    """
    try:
        verts, faces = create_virtual_implant_mesh(
            center_voxel, diameter_mm, length_mm,
            angulation_deg=angulation_deg, voxel_spacing_mm=voxel_spacing_mm,
        )
        if verts is not None and len(verts) > 0:
            # Convert to physical mm
            verts_mm = verts * np.array(voxel_spacing_mm)
            export_binary_stl(verts_mm, faces, output_path)
            return output_path
    except Exception as e:
        pass
    return None


def _json_default_helper(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalar types and arrays to standard Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def plan_data_to_json_str(plan_data: Dict[str, Any], indent: int = 2) -> str:
    """Serialize plan_data safely to a JSON string without numpy serialization errors."""
    clean_data = sanitize_for_json(plan_data)
    return json.dumps(clean_data, indent=indent, default=_json_default_helper)


def save_implant_planning_json(output_path: str, plan_data: Dict[str, Any]) -> str:
    """Save the planning data to a JSON file safely."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_data = sanitize_for_json(plan_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, default=_json_default_helper)
    return output_path


def generate_printable_html_report(plan_data: Dict[str, Any]) -> str:
    """
    Generate a clean, printable HTML clinical review dossier.
    """
    site = plan_data["selected_implant_site"]
    fdi = site["fdi_tooth"]
    meas = plan_data["anatomical_3d_measurements_mm"]
    imp = plan_data["virtual_implant_parameters"]
    safety = plan_data["safety_and_clearance_analysis"]
    teeth = plan_data["dentition_status"]

    overall = safety.get("overall_status", "PRELIMINARY FEASIBLE")
    is_safe = "FEASIBLE" in overall
    is_caution = "REVIEW" in overall

    status_bg = "#DCFCE7" if is_safe else ("#FEF3C7" if is_caution else "#FEE2E2")
    status_border = "#16A34A" if is_safe else ("#D97706" if is_caution else "#DC2626")
    status_text = "#166534" if is_safe else ("#92400E" if is_caution else "#991B1B")

    chk_rows = ""
    for chk in safety.get("checklist", []):
        meas_str = f"{chk['measured_mm']:.1f} mm" if chk.get("measured_mm") is not None else "N/A"
        icon_str = "✓" if chk.get("status") == "PASS" else ("⚠" if chk.get("status") == "CAUTION" else "✕")
        chk_rows += f"""
        <tr>
          <td style='padding:6px 4px;font-weight:600;'>{chk['item']}</td>
          <td style='padding:6px 4px;text-align:right;font-family:monospace;'>{meas_str}</td>
          <td style='padding:6px 4px;text-align:center;'>{icon_str}</td>
          <td style='padding:6px 4px;font-size:11px;color:#64748B;'>{chk.get('note', '')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D CBCT Implant Planning Dossier — FDI {fdi}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2rem; color: #1E293B; background: #F8FAFC; line-height: 1.5; }}
  .container {{ max-width: 860px; margin: 0 auto; background: white; border: 1px solid #CBD5E1; border-radius: 12px; padding: 2.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
  .header {{ border-bottom: 2px solid #0F3B7A; padding-bottom: 1.2rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: flex-end; }}
  .header h1 {{ margin: 0; font-size: 1.5rem; color: #0F3B7A; }}
  .header p {{ margin: 0.3rem 0 0; font-size: 0.85rem; color: #64748B; }}
  .status-badge {{ background: {status_bg}; border: 1.5px solid {status_border}; color: {status_text}; padding: 0.8rem 1.2rem; border-radius: 8px; font-weight: 700; font-size: 1.05rem; margin-bottom: 1.5rem; text-align: center; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.5rem; }}
  .card {{ border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.1rem; background: #F8FAFC; }}
  .card h3 {{ margin: 0 0 0.8rem 0; font-size: 0.95rem; color: #0F3B7A; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.4rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  td {{ padding: 0.35rem 0; }}
  td.label {{ color: #64748B; width: 55%; }}
  td.val {{ font-weight: 600; color: #0F172A; }}
  .chk-tbl th {{ text-align: left; padding: 6px 4px; font-size: 0.75rem; color: #64748B; text-transform: uppercase; border-bottom: 1px solid #CBD5E1; }}
  .disclaimer-card {{ background: #FFFBEB; border: 2px solid #F59E0B; border-radius: 10px; padding: 1.2rem; margin-top: 2rem; color: #92400E; font-size: 0.82rem; line-height: 1.5; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>3D CBCT AI-Assisted Implant Planning Dossier</h1>
      <p>Case: <strong>{plan_data['case_id']}</strong> &nbsp;|&nbsp; Target Site: <strong>FDI {fdi}</strong> &nbsp;|&nbsp; Date: {plan_data['timestamp'][:10]}</p>
    </div>
    <div style="font-size:0.8rem;color:#64748B;text-align:right;">OralSeg 35-Class<br>3D CBCT Workstation</div>
  </div>

  <div class="status-badge">
    {overall}
  </div>

  <div class="grid">
    <div class="card">
      <h3>📏 3D Anatomical Measurements</h3>
      <table>
        <tr><td class="label">Mesiodistal Space:</td><td class="val">~ {meas['mesiodistal_available_space']:.1f} mm</td></tr>
        <tr><td class="label">Buccolingual Ridge Width:</td><td class="val">~ {meas['buccolingual_ridge_width']:.1f} mm</td></tr>
        <tr><td class="label">Available Vertical Height:</td><td class="val">~ {meas['available_vertical_bone_height']:.1f} mm</td></tr>
        <tr><td class="label">Mesial Adjacent FDI:</td><td class="val">{site['adjacent_mesial_fdi'] or 'None'}</td></tr>
        <tr><td class="label">Distal Adjacent FDI:</td><td class="val">{site['adjacent_distal_fdi'] or 'None'}</td></tr>
      </table>
    </div>

    <div class="card">
      <h3>🔩 Virtual Candidate Implant</h3>
      <table>
        <tr><td class="label">Target Site:</td><td class="val">FDI {fdi}</td></tr>
        <tr><td class="label">Implant Diameter:</td><td class="val">Ø {imp['diameter_mm']:.1f} mm</td></tr>
        <tr><td class="label">Implant Length:</td><td class="val">{imp['length_mm']:.0f} mm</td></tr>
        <tr><td class="label">Buccolingual Angle:</td><td class="val">{imp['angulation_degrees']['buccolingual_rx']:+.0f}°</td></tr>
        <tr><td class="label">Mesiodistal Angle:</td><td class="val">{imp['angulation_degrees']['mesiodistal_ry']:+.0f}°</td></tr>
      </table>
    </div>
  </div>

  <div class="card" style="margin-bottom: 1.5rem;">
    <h3>🛡️ Safety &amp; Clearance Verification Checklist</h3>
    <table class="chk-tbl">
      <thead>
        <tr><th>Anatomical Landmark</th><th style="text-align:right;">Measured</th><th style="text-align:center;">Check</th><th>Clinical Note</th></tr>
      </thead>
      <tbody>
        {chk_rows}
      </tbody>
    </table>
  </div>

  <div class="disclaimer-card">
    <strong style="font-size:0.9rem;display:block;margin-bottom:0.4rem;">⚠️ CLINICAL DISCLAIMER</strong>
    {plan_data['disclaimer']}
  </div>
</div>
</body>
</html>
"""
    return html
