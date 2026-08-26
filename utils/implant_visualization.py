"""
implant_visualization.py
Renders 2D Multi-Planar cross-sections with virtual implant overlays,
local cropped 3D implant site models, and full 3D interactive Plotly scenes with layer controls
and camera viewpoint presets (Buccal, Lingual, Occlusal, Mesial, Distal).
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Dict, Any, List, Tuple, Optional

from utils.visualization import extract_mesh_for_label, _apply_overlay
from utils.implant_geometry import create_virtual_implant_mesh, get_adjacent_teeth_fdi, FDI_TO_LABEL


def render_2d_mpr_implant_views(
    cbct_vol: np.ndarray,
    seg_vol: np.ndarray,
    implant_center_vox: np.ndarray,
    implant_diameter_mm: float,
    implant_length_mm: float,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    canal_distance_mm: Optional[float] = None,
    bone_height_mm: Optional[float] = None,
    mesiodistal_mm: Optional[float] = None,
    figsize=(16, 5.2),
) -> plt.Figure:
    """
    Render 2D Multi-Planar Reconstruction (MPR) slice views (Axial, Coronal, Sagittal)
    centered precisely at the virtual implant site with the implant cylinder overlaid.
    """
    cx = int(np.clip(round(implant_center_vox[0]), 0, cbct_vol.shape[0] - 1))
    cy = int(np.clip(round(implant_center_vox[1]), 0, cbct_vol.shape[1] - 1))
    cz = int(np.clip(round(implant_center_vox[2]), 0, cbct_vol.shape[2] - 1))

    rad_vox_x = (implant_diameter_mm / 2.0) / max(voxel_spacing_mm[0], 0.01)
    rad_vox_y = (implant_diameter_mm / 2.0) / max(voxel_spacing_mm[1], 0.01)
    len_vox_z = implant_length_mm / max(voxel_spacing_mm[2], 0.01)

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0F172A")

    # ── 1. Axial View (Z slice at center of implant) ─────────────────────────
    ax_slice = cbct_vol[:, :, cz].T
    ax_seg = seg_vol[:, :, cz].T
    _apply_overlay(axes[0], ax_slice, ax_seg, f"Axial Cross-Section (z={cz})")

    imp_circle = Circle(
        (cx, cy), radius=(rad_vox_x + rad_vox_y) / 2.0,
        facecolor="cyan", alpha=0.35, edgecolor="#00E5FF", linewidth=2.2, linestyle="--"
    )
    axes[0].add_patch(imp_circle)
    axes[0].plot(cx, cy, marker="+", color="#00E5FF", markersize=10, markeredgewidth=2.0)

    if mesiodistal_mm is not None:
        axes[0].text(
            0.04, 0.93, f"MD Space: ~{mesiodistal_mm:.1f} mm",
            transform=axes[0].transAxes, color="#38BDF8", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#334155", alpha=0.9)
        )

    # ── 2. Coronal View (Y slice at center of implant) ───────────────────────
    cor_slice = cbct_vol[:, cy, :].T
    cor_seg = seg_vol[:, cy, :].T
    _apply_overlay(axes[1], cor_slice, cor_seg, f"Coronal Cross-Section (y={cy})")

    imp_rect_cor = Rectangle(
        (cx - rad_vox_x, cz - len_vox_z), width=2 * rad_vox_x, height=len_vox_z,
        facecolor="#00E5FF", alpha=0.30, edgecolor="#00E5FF", linewidth=2.2
    )
    axes[1].add_patch(imp_rect_cor)
    axes[1].plot([cx, cx], [cz, cz - len_vox_z], color="#00E5FF", linestyle=":", linewidth=1.8)

    if bone_height_mm is not None:
        axes[1].text(
            0.04, 0.93, f"Bone Height: ~{bone_height_mm:.1f} mm",
            transform=axes[1].transAxes, color="#FDE047", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#334155", alpha=0.9)
        )

    # ── 3. Sagittal View (X slice at center of implant) ──────────────────────
    sag_slice = cbct_vol[cx, :, :].T
    sag_seg = seg_vol[cx, :, :].T
    _apply_overlay(axes[2], sag_slice, sag_seg, f"Sagittal Cross-Section (x={cx})")

    imp_rect_sag = Rectangle(
        (cy - rad_vox_y, cz - len_vox_z), width=2 * rad_vox_y, height=len_vox_z,
        facecolor="#00E5FF", alpha=0.30, edgecolor="#00E5FF", linewidth=2.2
    )
    axes[2].add_patch(imp_rect_sag)
    axes[2].plot([cy, cy], [cz, cz - len_vox_z], color="#00E5FF", linestyle=":", linewidth=1.8)

    if canal_distance_mm is not None:
        c_color = "#4ADE80" if canal_distance_mm >= 2.0 else ("#FACC15" if canal_distance_mm >= 1.0 else "#F87171")
        axes[2].text(
            0.04, 0.93, f"Canal Dist: {canal_distance_mm:.1f} mm",
            transform=axes[2].transAxes, color=c_color, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#334155", alpha=0.9)
        )

    plt.tight_layout(pad=0.6)
    return fig


def build_3d_implant_scene(
    seg_vol: np.ndarray,
    implant_center_vox: np.ndarray,
    implant_diameter_mm: float,
    implant_length_mm: float,
    angulation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    selected_fdi_tooth: int = 46,
    detected_teeth: Optional[List[int]] = None,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    show_maxilla: bool = True,
    show_mandible: bool = True,
    show_teeth: bool = True,
    show_adjacent_only: bool = False,
    show_canal: bool = True,
    show_implant: bool = True,
    camera_view: str = "perspective",
    step_size: int = 2,
):
    """
    Build a comprehensive 3D interactive Plotly scene with layer controls
    and camera perspective presets.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    is_mandibular = (selected_fdi_tooth >= 31 and selected_fdi_tooth <= 48)

    # 1. Maxilla
    if show_maxilla:
        v, f = extract_mesh_for_label(
            seg_vol, [1], step_size=step_size,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            smoothing_iterations=10, decimate_target=0.75,
        )
        if v is not None:
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#93C5FD", opacity=0.25,
                name="Maxilla Bone",
                lighting=dict(ambient=0.45, diffuse=0.65, specular=0.25),
                hoverinfo="name",
            ))

    # 2. Mandible
    if show_mandible:
        v, f = extract_mesh_for_label(
            seg_vol, [2], step_size=step_size,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            smoothing_iterations=10, decimate_target=0.75,
        )
        if v is not None:
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#86EFAC", opacity=0.28,
                name="Mandible Bone",
                lighting=dict(ambient=0.45, diffuse=0.65, specular=0.25),
                hoverinfo="name",
            ))

    # 3. Teeth (All or Adjacent Only)
    if show_teeth:
        if show_adjacent_only and detected_teeth:
            mesial_fdi, distal_fdi = get_adjacent_teeth_fdi(selected_fdi_tooth, detected_teeth)
            tooth_labels = []
            if mesial_fdi and FDI_TO_LABEL.get(mesial_fdi):
                tooth_labels.append(FDI_TO_LABEL[mesial_fdi])
            if distal_fdi and FDI_TO_LABEL.get(distal_fdi):
                tooth_labels.append(FDI_TO_LABEL[distal_fdi])
            name_teeth = "Adjacent Teeth"
        else:
            tooth_labels = list(range(3, 35))
            name_teeth = "Dentition (Teeth)"

        if tooth_labels:
            v, f = extract_mesh_for_label(
                seg_vol, tooth_labels, step_size=max(1, step_size - 1),
                voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
                keep_largest_component=False, smoothing_iterations=10,
            )
            if v is not None:
                fig.add_trace(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color="#FEF08A", opacity=0.92,
                    name=name_teeth,
                    lighting=dict(ambient=0.55, diffuse=0.8, specular=0.4, roughness=0.2),
                    hoverinfo="name",
                ))

    # 4. Mandibular Canal
    if show_canal:
        v, f = extract_mesh_for_label(
            seg_vol, [35], step_size=1,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            keep_largest_component=False, smoothing_iterations=10,
        )
        if v is not None:
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#EF4444", opacity=1.0,
                name="Mandibular Canal (Nerve)",
                lighting=dict(ambient=0.65, diffuse=0.85, specular=0.55),
                hoverinfo="name",
            ))

    # 5. Virtual Implant Cylinder
    if show_implant:
        v_imp, f_imp = create_virtual_implant_mesh(
            center_voxel=implant_center_vox,
            diameter_mm=implant_diameter_mm,
            length_mm=implant_length_mm,
            angulation_deg=angulation_deg,
            voxel_spacing_mm=voxel_spacing_mm,
            num_segments=32,
        )
        if v_imp is not None:
            # Convert voxel vertices to physical mm for scene consistency
            v_imp_mm = v_imp * np.array(voxel_spacing_mm)
            fig.add_trace(go.Mesh3d(
                x=v_imp_mm[:, 0], y=v_imp_mm[:, 1], z=v_imp_mm[:, 2],
                i=f_imp[:, 0], j=f_imp[:, 1], k=f_imp[:, 2],
                color="#06B6D4", opacity=1.0,
                name=f"Virtual Implant (Ø{implant_diameter_mm}×{implant_length_mm}mm)",
                lighting=dict(ambient=0.65, diffuse=0.9, specular=0.85, roughness=0.1),
                hoverinfo="name",
            ))

    # Camera presets
    camera_presets = {
        "perspective": dict(eye=dict(x=1.6, y=-1.6, z=1.2), up=dict(x=0, y=0, z=1)),
        "buccal":      dict(eye=dict(x=0.0, y=-2.3, z=0.2), up=dict(x=0, y=0, z=1)),
        "lingual":     dict(eye=dict(x=0.0, y=2.3, z=0.2), up=dict(x=0, y=0, z=1)),
        "occlusal":    dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0)),
        "mesial":      dict(eye=dict(x=-2.2, y=0.0, z=0.3), up=dict(x=0, y=0, z=1)),
        "distal":      dict(eye=dict(x=2.2, y=0.0, z=0.3), up=dict(x=0, y=0, z=1)),
    }
    selected_camera = camera_presets.get(camera_view.lower(), camera_presets["perspective"])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="#0d1117"),
            yaxis=dict(visible=False, backgroundcolor="#0d1117"),
            zaxis=dict(visible=False, backgroundcolor="#0d1117"),
            bgcolor="#0d1117",
            aspectmode="data",
            camera=selected_camera,
        ),
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            yanchor="top", y=0.98,
            xanchor="left", x=0.02
        ),
        height=620,
    )

    return fig


def build_local_roi_3d_scene(
    seg_vol: np.ndarray,
    implant_center_vox: np.ndarray,
    implant_diameter_mm: float,
    implant_length_mm: float,
    angulation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    roi_radius_mm: float = 22.0,
):
    """
    Build a focused 3D viewport cropped locally around the selected implant site.
    """
    import plotly.graph_objects as go

    cx, cy, cz = int(round(implant_center_vox[0])), int(round(implant_center_vox[1])), int(round(implant_center_vox[2]))
    rx_vox = int(round(roi_radius_mm / max(voxel_spacing_mm[0], 0.01)))
    ry_vox = int(round(roi_radius_mm / max(voxel_spacing_mm[1], 0.01)))
    rz_vox = int(round(roi_radius_mm / max(voxel_spacing_mm[2], 0.01)))

    x0 = max(0, cx - rx_vox)
    x1 = min(seg_vol.shape[0], cx + rx_vox)
    y0 = max(0, cy - ry_vox)
    y1 = min(seg_vol.shape[1], cy + ry_vox)
    z0 = max(0, cz - rz_vox)
    z1 = min(seg_vol.shape[2], cz + rz_vox)

    cropped_vol = np.zeros_like(seg_vol)
    cropped_vol[x0:x1, y0:y1, z0:z1] = seg_vol[x0:x1, y0:y1, z0:z1]

    fig = go.Figure()

    # Bone in local ROI
    v_bone, f_bone = extract_mesh_for_label(
        cropped_vol, [1, 2], step_size=1,
        voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
        smoothing_iterations=8,
    )
    if v_bone is not None:
        fig.add_trace(go.Mesh3d(
            x=v_bone[:, 0], y=v_bone[:, 1], z=v_bone[:, 2],
            i=f_bone[:, 0], j=f_bone[:, 1], k=f_bone[:, 2],
            color="#86EFAC", opacity=0.32, name="Alveolar Bone (ROI)",
            lighting=dict(ambient=0.4, diffuse=0.7, specular=0.2),
        ))

    # Adjacent teeth in local ROI
    v_teeth, f_teeth = extract_mesh_for_label(
        cropped_vol, list(range(3, 35)), step_size=1,
        voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
        keep_largest_component=False, smoothing_iterations=8,
    )
    if v_teeth is not None:
        fig.add_trace(go.Mesh3d(
            x=v_teeth[:, 0], y=v_teeth[:, 1], z=v_teeth[:, 2],
            i=f_teeth[:, 0], j=f_teeth[:, 1], k=f_teeth[:, 2],
            color="#FEF08A", opacity=0.95, name="Adjacent Teeth (ROI)",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4),
        ))

    # Mandibular canal in local ROI
    v_canal, f_canal = extract_mesh_for_label(
        cropped_vol, [35], step_size=1,
        voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
        keep_largest_component=False, smoothing_iterations=8,
    )
    if v_canal is not None:
        fig.add_trace(go.Mesh3d(
            x=v_canal[:, 0], y=v_canal[:, 1], z=v_canal[:, 2],
            i=f_canal[:, 0], j=f_canal[:, 1], k=f_canal[:, 2],
            color="#EF4444", opacity=1.0, name="Mandibular Canal (ROI)",
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5),
        ))

    # Virtual Implant Cylinder
    v_imp, f_imp = create_virtual_implant_mesh(
        center_voxel=implant_center_vox,
        diameter_mm=implant_diameter_mm,
        length_mm=implant_length_mm,
        angulation_deg=angulation_deg,
        voxel_spacing_mm=voxel_spacing_mm,
        num_segments=32,
    )
    if v_imp is not None:
        v_imp_mm = v_imp * np.array(voxel_spacing_mm)
        fig.add_trace(go.Mesh3d(
            x=v_imp_mm[:, 0], y=v_imp_mm[:, 1], z=v_imp_mm[:, 2],
            i=f_imp[:, 0], j=f_imp[:, 1], k=f_imp[:, 2],
            color="#06B6D4", opacity=1.0,
            name=f"Virtual Implant (Ø{implant_diameter_mm}×{implant_length_mm}mm)",
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.8),
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0d1117",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.3, y=-1.3, z=1.0)
            )
        ),
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, b=0, t=20),
        legend=dict(font=dict(color="white", size=10), bgcolor="rgba(15, 23, 42, 0.8)"),
        height=480,
    )

    return fig


def build_diagnocat_panoramic_3d_scene(
    seg_vol: np.ndarray,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    show_perio: bool = True,
    show_restorative: bool = True,
    show_endo: bool = True,
    show_anatomy: bool = True,
    selected_fdi: Optional[int] = None,
    step_size: int = 2,
):
    """
    Builds the Diagnocat-style 3D panoramic reconstructed viewport with
    layer pills (Perio, Restorative, Endo, Anatomy) and high-contrast clinical materials.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # 1. Alveolar Bone (Mandible + Maxilla) in translucent slate/blue
    if show_anatomy:
        v_bone, f_bone = extract_mesh_for_label(
            seg_vol, [1, 2], step_size=step_size,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            smoothing_iterations=10, decimate_target=0.70,
        )
        if v_bone is not None:
            fig.add_trace(go.Mesh3d(
                x=v_bone[:, 0], y=v_bone[:, 1], z=v_bone[:, 2],
                i=f_bone[:, 0], j=f_bone[:, 1], k=f_bone[:, 2],
                color="#64748B", opacity=0.35,
                name="Anatomy (Jaw Bone)",
                lighting=dict(ambient=0.45, diffuse=0.65, specular=0.25),
                hoverinfo="name",
            ))

    # 2. Dentition (Teeth) in realistic enamel ivory
    v_teeth, f_teeth = extract_mesh_for_label(
        seg_vol, list(range(3, 35)), step_size=max(1, step_size - 1),
        voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
        keep_largest_component=False, smoothing_iterations=10,
    )
    if v_teeth is not None:
        fig.add_trace(go.Mesh3d(
            x=v_teeth[:, 0], y=v_teeth[:, 1], z=v_teeth[:, 2],
            i=f_teeth[:, 0], j=f_teeth[:, 1], k=f_teeth[:, 2],
            color="#F8FAFC", opacity=0.96,
            name="Dentition",
            lighting=dict(ambient=0.6, diffuse=0.85, specular=0.45, roughness=0.15),
            hoverinfo="name",
        ))

    # 3. Mandibular Nerve Canal in warm amber/brown
    if show_anatomy:
        v_canal, f_canal = extract_mesh_for_label(
            seg_vol, [35], step_size=1,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            keep_largest_component=False, smoothing_iterations=10,
        )
        if v_canal is not None:
            fig.add_trace(go.Mesh3d(
                x=v_canal[:, 0], y=v_canal[:, 1], z=v_canal[:, 2],
                i=f_canal[:, 0], j=f_canal[:, 1], k=f_canal[:, 2],
                color="#F59E0B", opacity=1.0,
                name="Mandibular Canal (Nerve)",
                lighting=dict(ambient=0.65, diffuse=0.85, specular=0.55),
                hoverinfo="name",
            ))

    # 4. Highlight Selected FDI Tooth if specified
    if selected_fdi and FDI_TO_LABEL.get(selected_fdi):
        lbl = FDI_TO_LABEL[selected_fdi]
        v_sel, f_sel = extract_mesh_for_label(
            seg_vol, [lbl], step_size=1,
            voxel_spacing=tuple(voxel_spacing_mm) if voxel_spacing_mm else None,
            smoothing_iterations=8,
        )
        if v_sel is not None:
            fig.add_trace(go.Mesh3d(
                x=v_sel[:, 0], y=v_sel[:, 1], z=v_sel[:, 2],
                i=f_sel[:, 0], j=f_sel[:, 1], k=f_sel[:, 2],
                color="#EC4899", opacity=1.0,
                name=f"Selected Tooth {selected_fdi}",
                lighting=dict(ambient=0.7, diffuse=0.9, specular=0.8),
                hoverinfo="name",
            ))

    # Diagnocat Panoramic Viewing Angle
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="#090D16"),
            yaxis=dict(visible=False, backgroundcolor="#090D16"),
            zaxis=dict(visible=False, backgroundcolor="#090D16"),
            bgcolor="#090D16",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.0, y=-2.2, z=0.3),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        paper_bgcolor="#090D16",
        margin=dict(l=0, r=0, b=0, t=10),
        showlegend=False,
        height=380,
    )

    return fig

