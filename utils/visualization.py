"""
visualization.py
Generates 2D slice views (axial, coronal, sagittal) with segmentation overlays,
FDI tooth charts, and high-fidelity 3D mesh reconstruction with Plotly interactive visualization.

Key Features:
- Exact clinical report multiplanar layout with slice indicators, crosshairs, and orientation tags.
- Connected-component noise reduction per anatomical class.
- Conservative morphological cleaning (binary closing & hole filling).
- Marching cubes isosurface extraction preserving true physical NIfTI voxel spacing.
- Taubin / Laplacian surface smoothing and quadric decimation for clean clinical surfaces.
- Individual mesh traces for each detected FDI tooth with distinct quadrant-based color coding.
- Dynamic transparency: semi-transparent bone (maxilla/mandible) and highlighted mandibular canal.
- Auto-centered scene framing and camera angle presets.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────────────────────
# 2D & 3D Harmonized Clinical Color Palette (0=transparent, 1–35=distinct colors)
# ─────────────────────────────────────────────────────────────────────────────
_COLORS = [
    (0.0,  0.0,  0.0,  0.0),   # 0  = background (transparent)
    (0.15, 0.45, 0.95, 0.70),  # 1  = maxilla     (royal blue #2563EB)
    (0.41, 0.62, 0.22, 0.70),  # 2  = mandible    (olive/sage green #689F38)
    # Teeth 11-18 (upper right)
    (1.00, 0.95, 0.46, 0.85),  # 3  = tooth 11 (#FFF176 yellow)
    (1.00, 0.84, 0.00, 0.85),  # 4  = tooth 12 (#FFD600 gold)
    (1.00, 0.63, 0.00, 0.85),  # 5  = tooth 13 (#FFA000 amber)
    (0.90, 0.32, 0.00, 0.85),  # 6  = tooth 14 (#E65100 burnt orange)
    (0.55, 0.43, 0.39, 0.85),  # 7  = tooth 15 (#8D6E63 warm brown/tan)
    (0.56, 0.14, 0.67, 0.85),  # 8  = tooth 16 (#8E24AA deep purple)
    (0.25, 0.32, 0.71, 0.85),  # 9  = tooth 17 (#3F51B5 indigo)
    (0.73, 0.41, 0.78, 0.85),  # 10 = tooth 18 (#BA68C8 pink/magenta)
    # Teeth 21-28 (upper left)
    (0.30, 0.69, 0.31, 0.85),  # 11 = tooth 21 (#4CAF50 mint green)
    (0.18, 0.49, 0.20, 0.85),  # 12 = tooth 22 (#2E7D32 green)
    (0.11, 0.37, 0.13, 0.85),  # 13 = tooth 23 (#1B5E20 forest green)
    (0.00, 0.54, 0.48, 0.85),  # 14 = tooth 24 (#00897B teal)
    (0.00, 0.67, 0.76, 0.85),  # 15 = tooth 25 (#00ACC1 cyan)
    (0.01, 0.61, 0.90, 0.85),  # 16 = tooth 26 (#039BE5 sky blue)
    (0.10, 0.46, 0.82, 0.85),  # 17 = tooth 27 (#1976D2 blue)
    (0.16, 0.21, 0.58, 0.85),  # 18 = tooth 28 (#283593 royal indigo)
    # Teeth 31-38 (lower left)
    (0.88, 0.75, 0.91, 0.85),  # 19 = tooth 31 (#E1BEE7 lavender)
    (0.67, 0.28, 0.74, 0.85),  # 20 = tooth 32 (#AB47BC violet)
    (0.48, 0.12, 0.64, 0.85),  # 21 = tooth 33 (#7B1FA2 purple)
    (0.29, 0.08, 0.55, 0.85),  # 22 = tooth 34 (#4A148C dark violet)
    (0.36, 0.42, 0.75, 0.85),  # 23 = tooth 35 (#5C6BC0 periwinkle)
    (0.12, 0.53, 0.90, 0.85),  # 24 = tooth 36 (#1E88E5 blue)
    (0.05, 0.28, 0.63, 0.85),  # 25 = tooth 37 (#0D47A1 deep blue)
    (0.10, 0.14, 0.49, 0.85),  # 26 = tooth 38 (#1A237E midnight blue)
    # Teeth 41-48 (lower right)
    (0.97, 0.73, 0.82, 0.85),  # 27 = tooth 41 (#F8BBD0 light pink)
    (0.94, 0.38, 0.57, 0.85),  # 28 = tooth 42 (#F06292 rose)
    (0.26, 0.63, 0.28, 0.85),  # 29 = tooth 43 (#43A047 vivid green)
    (0.98, 0.55, 0.00, 0.85),  # 30 = tooth 44 (#FB8C00 orange)
    (0.63, 0.53, 0.50, 0.85),  # 31 = tooth 45 (#A1887F warm tan)
    (0.91, 0.12, 0.39, 0.85),  # 32 = tooth 46 (#E91E63 magenta)
    (1.00, 0.96, 0.62, 0.85),  # 33 = tooth 47 (#FFF59D cream/ivory)
    (0.53, 0.05, 0.31, 0.85),  # 34 = tooth 48 (#880E4F deep ruby)
    (1.00, 0.09, 0.27, 0.95),  # 35 = mandibular canal (bright red #FF1744)
]

_SEG_CMAP = ListedColormap([c[:3] for c in _COLORS])
_SEG_ALPHA = [c[3] for c in _COLORS]
_BOUNDS = np.arange(-0.5, 36.5, 1)
_SEG_NORM = BoundaryNorm(_BOUNDS, _SEG_CMAP.N)

_LABEL_NAMES = {
    0: "Background", 1: "Maxilla", 2: "Mandible",
    3: "Tooth 11", 4: "Tooth 12", 5: "Tooth 13", 6: "Tooth 14", 7: "Tooth 15", 8: "Tooth 16", 9: "Tooth 17", 10: "Tooth 18",
    11: "Tooth 21", 12: "Tooth 22", 13: "Tooth 23", 14: "Tooth 24", 15: "Tooth 25", 16: "Tooth 26", 17: "Tooth 27", 18: "Tooth 28",
    19: "Tooth 31", 20: "Tooth 32", 21: "Tooth 33", 22: "Tooth 34", 23: "Tooth 35", 24: "Tooth 36", 25: "Tooth 37", 26: "Tooth 38",
    27: "Tooth 41", 28: "Tooth 42", 29: "Tooth 43", 30: "Tooth 44", 31: "Tooth 45", 32: "Tooth 46", 33: "Tooth 47", 34: "Tooth 48",
    35: "Mandibular Canal",
}

# ─────────────────────────────────────────────────────────────────────────────
# Anatomical FDI Tooth Definitions & 3D Color System
# ─────────────────────────────────────────────────────────────────────────────
TOOTH_LABEL_TO_FDI = {
    3: 11,  4: 12,  5: 13,  6: 14,  7: 15,  8: 16,  9: 17,  10: 18,
    11: 21, 12: 22, 13: 23, 14: 24, 15: 25, 16: 26, 17: 27, 18: 28,
    19: 31, 20: 32, 21: 33, 22: 34, 23: 35, 24: 36, 25: 37, 26: 38,
    27: 41, 28: 42, 29: 43, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48,
}
FDI_TO_TOOTH_LABEL = {v: k for k, v in TOOTH_LABEL_TO_FDI.items()}

TOOTH_NAMES = {
    11: "Maxillary Right Central Incisor",
    12: "Maxillary Right Lateral Incisor",
    13: "Maxillary Right Canine",
    14: "Maxillary Right 1st Premolar",
    15: "Maxillary Right 2nd Premolar",
    16: "Maxillary Right 1st Molar",
    17: "Maxillary Right 2nd Molar",
    18: "Maxillary Right 3rd Molar",
    21: "Maxillary Left Central Incisor",
    22: "Maxillary Left Lateral Incisor",
    23: "Maxillary Left Canine",
    24: "Maxillary Left 1st Premolar",
    25: "Maxillary Left 2nd Premolar",
    26: "Maxillary Left 1st Molar",
    27: "Maxillary Left 2nd Molar",
    28: "Maxillary Left 3rd Molar",
    31: "Mandibular Left Central Incisor",
    32: "Mandibular Left Lateral Incisor",
    33: "Mandibular Left Canine",
    34: "Mandibular Left 1st Premolar",
    35: "Mandibular Left 2nd Premolar",
    36: "Mandibular Left 1st Molar",
    37: "Mandibular Left 2nd Molar",
    38: "Mandibular Left 3rd Molar",
    41: "Mandibular Right Central Incisor",
    42: "Mandibular Right Lateral Incisor",
    43: "Mandibular Right Canine",
    44: "Mandibular Right 1st Premolar",
    45: "Mandibular Right 2nd Premolar",
    46: "Mandibular Right 1st Molar",
    47: "Mandibular Right 2nd Molar",
    48: "Mandibular Right 3rd Molar",
}

# Hex colors for 3D mesh rendering exactly matching _COLORS
FDI_3D_COLORS = {
    # Quadrant 1: Upper Right
    11: "#FFF176", 12: "#FFD600", 13: "#FFA000", 14: "#E65100",
    15: "#8D6E63", 16: "#8E24AA", 17: "#3F51B5", 18: "#BA68C8",
    # Quadrant 2: Upper Left
    21: "#4CAF50", 22: "#2E7D32", 23: "#1B5E20", 24: "#00897B",
    25: "#00ACC1", 26: "#039BE5", 27: "#1976D2", 28: "#283593",
    # Quadrant 3: Lower Left
    31: "#E1BEE7", 32: "#AB47BC", 33: "#7B1FA2", 34: "#4A148C",
    35: "#5C6BC0", 36: "#1E88E5", 37: "#0D47A1", 38: "#1A237E",
    # Quadrant 4: Lower Right
    41: "#F8BBD0", 42: "#F06292", 43: "#43A047", 44: "#FB8C00",
    45: "#A1887F", 46: "#E91E63", 47: "#FFF59D", 48: "#880E4F",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2D Multiplanar Slice Rendering (Clinical Layout with Crosshairs & Slice Tags)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overlay(
    ax,
    cbct_slice: np.ndarray,
    seg_slice: np.ndarray,
    title: str,
    slice_text: str = "",
    left_label: str = "",
    right_label: str = "",
    vline_pos: Optional[float] = None,
    vline_color: str = "#EF4444",
    hline_pos: Optional[float] = None,
    hline_color: str = "#22C55E",
    cmap_cbct: str = "gray",
):
    """Render CBCT slice with segmentation overlay, crosshairs, orientation markers, and slice indicators."""
    ax.set_facecolor("#000000")
    
    # Background CBCT
    vmin, vmax = np.percentile(cbct_slice, 1), np.percentile(cbct_slice, 99)
    ax.imshow(cbct_slice, cmap=cmap_cbct, aspect="equal", interpolation="bilinear", vmin=vmin, vmax=vmax)

    # Per-pixel RGBA segmentation overlay
    h, w = seg_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for label_idx, (r, g, b, a) in enumerate(_COLORS):
        if label_idx == 0:
            continue
        mask = (seg_slice == label_idx)
        overlay[mask] = [r, g, b, a]

    ax.imshow(overlay, aspect="equal", interpolation="none")

    # Orthogonal Crosshair Reference Lines
    if vline_pos is not None and 0 <= vline_pos < w:
        ax.axvline(x=vline_pos, color=vline_color, linewidth=0.9, alpha=0.85)
    if hline_pos is not None and 0 <= hline_pos < h:
        ax.axhline(y=hline_pos, color=hline_color, linewidth=0.9, alpha=0.85)

    # Orientation Letters (e.g., R / L, A / P)
    if left_label:
        ax.text(0.03, 0.5, left_label, transform=ax.transAxes, color="#E2E8F0",
                fontsize=11, fontweight="bold", va="center", ha="left")
    if right_label:
        ax.text(0.97, 0.5, right_label, transform=ax.transAxes, color="#E2E8F0",
                fontsize=11, fontweight="bold", va="center", ha="right")

    # Slice Index Tag (Bottom Left)
    if slice_text:
        ax.text(0.04, 0.05, slice_text, transform=ax.transAxes, color="#F8FAFC",
                fontsize=8.5, fontfamily="monospace", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"))

    ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=6)
    ax.axis("off")


def render_slice_views(
    cbct_vol: np.ndarray,
    seg_vol: np.ndarray,
    axial_idx: int,
    coronal_idx: int,
    sagittal_idx: int,
    figsize=(14.5, 4.8),
) -> plt.Figure:
    """
    Render axial, coronal, sagittal views with segmentation overlay,
    matching the exact Multiplanar View in the clinical report.
    """
    x_max, y_max, z_max = cbct_vol.shape
    axial_idx    = max(0, min(axial_idx,    z_max - 1))
    coronal_idx  = max(0, min(coronal_idx,  y_max - 1))
    sagittal_idx = max(0, min(sagittal_idx, x_max - 1))

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#000000")

    # 1. Axial: slice along Z (x=coronal/sagittal plane)
    ax_slice  = cbct_vol[:, :, axial_idx].T
    ax_seg    = seg_vol[:, :, axial_idx].T
    _apply_overlay(
        axes[0], ax_slice, ax_seg,
        title="AXIAL VIEW",
        slice_text=f"Slice {axial_idx + 1} / {z_max}",
        left_label="R", right_label="L",
        vline_pos=sagittal_idx, vline_color="#22C55E",
        hline_pos=coronal_idx, hline_color="#EF4444",
    )

    # 2. Coronal: slice along Y
    cor_slice = cbct_vol[:, coronal_idx, :].T
    cor_seg   = seg_vol[:, coronal_idx, :].T
    _apply_overlay(
        axes[1], cor_slice, cor_seg,
        title="CORONAL VIEW",
        slice_text=f"Slice {coronal_idx + 1} / {y_max}",
        left_label="R", right_label="L",
        vline_pos=sagittal_idx, vline_color="#22C55E",
        hline_pos=axial_idx, hline_color="#3B82F6",
    )

    # 3. Sagittal: slice along X
    sag_slice = cbct_vol[sagittal_idx, :, :].T
    sag_seg   = seg_vol[sagittal_idx, :, :].T
    _apply_overlay(
        axes[2], sag_slice, sag_seg,
        title="SAGITTAL VIEW",
        slice_text=f"Slice {sagittal_idx + 1} / {x_max}",
        left_label="A", right_label="P",
        vline_pos=coronal_idx, vline_color="#EF4444",
        hline_pos=axial_idx, hline_color="#3B82F6",
    )

    plt.tight_layout(pad=0.6)
    return fig


def render_legend(detected_labels: List[int]) -> plt.Figure:
    """Return a compact legend figure for detected structures."""
    patches = []
    for lbl in sorted(detected_labels):
        if lbl == 0:
            continue
        color = _COLORS[lbl][:3]
        name = _LABEL_NAMES.get(lbl, f"Label {lbl}")
        patches.append(mpatches.Patch(color=color, label=name))

    n = len(patches)
    ncols = min(6, max(1, n))
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, ax = plt.subplots(figsize=(ncols * 1.6, nrows * 0.55), facecolor="#000000")
    ax.set_facecolor("#000000")
    ax.legend(handles=patches, loc="center", ncol=ncols,
              fontsize=8, frameon=False,
              labelcolor="white", handlelength=1.2, handleheight=0.8)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def render_tooth_chart(seg_info: Dict[str, Any]) -> plt.Figure:
    """
    Render a dental chart grid showing detected / missing teeth.
    """
    upper_right = [18, 17, 16, 15, 14, 13, 12, 11]
    upper_left  = [21, 22, 23, 24, 25, 26, 27, 28]
    lower_left  = [31, 32, 33, 34, 35, 36, 37, 38]
    lower_right = [41, 42, 43, 44, 45, 46, 47, 48]

    from utils.oralseg_inference import TOOTH_FDI_MAP
    fdi_to_label = {v: k for k, v in TOOTH_FDI_MAP.items()}
    structures = seg_info.get("structures", {})

    def tooth_detected(fdi_num: int) -> bool:
        label_idx = fdi_to_label.get(fdi_num)
        if label_idx is None:
            return False
        return structures.get(label_idx, {}).get("detected", False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), facecolor="#000000")

    for row_idx, (row_right, row_left, row_label) in enumerate([
        (upper_right, upper_left, "Upper"),
        (lower_right, lower_left, "Lower"),
    ]):
        ax = axes[row_idx]
        ax.set_facecolor("#000000")
        row_teeth = row_right + row_left
        for col_idx, tooth in enumerate(row_teeth):
            det = tooth_detected(tooth)
            color = "#22C55E" if det else "#334155"
            edge  = "#4ADE80" if det else "#475569"
            rect = mpatches.FancyBboxPatch(
                (col_idx + 0.05, 0.1), 0.85, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor=edge, linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, 0.5, str(tooth),
                    ha="center", va="center",
                    color="white" if det else "#94A3B8",
                    fontsize=8, fontweight="bold" if det else "normal")

        ax.set_xlim(0, 16)
        ax.set_ylim(0, 1)
        ax.set_ylabel(row_label, color="white", fontsize=9, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.axvline(x=8, color="#475569", linewidth=1, linestyle="--")

    plt.suptitle("FDI Tooth Chart", color="white", fontsize=11, y=1.02)
    plt.tight_layout(pad=0.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3D Morphological Cleaning & Connected-Component Filtering
# ─────────────────────────────────────────────────────────────────────────────

def clean_binary_mask(
    mask: np.ndarray,
    min_voxels: int = 30,
    keep_largest_n: int = 1,
    apply_morphology: bool = True,
    closing_radius: int = 1,
) -> np.ndarray:
    """
    Remove small isolated voxel noise and floating fragments from a binary mask.
    Applies connected component analysis and conservative binary morphology.
    """
    if mask.sum() < min_voxels:
        return np.zeros_like(mask, dtype=bool)

    from scipy import ndimage

    labeled_array, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    if num_features == 1:
        cleaned = (labeled_array > 0)
    else:
        component_sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
        if keep_largest_n == 1:
            largest_idx = int(np.argmax(component_sizes)) + 1
            if component_sizes[largest_idx - 1] < min_voxels:
                return np.zeros_like(mask, dtype=bool)
            cleaned = (labeled_array == largest_idx)
        else:
            valid_indices = np.where(component_sizes >= min_voxels)[0] + 1
            if len(valid_indices) == 0:
                return np.zeros_like(mask, dtype=bool)
            cleaned = np.isin(labeled_array, valid_indices)

    if apply_morphology and cleaned.sum() > 0:
        try:
            if closing_radius > 0:
                cleaned = ndimage.binary_closing(cleaned, iterations=closing_radius)
            cleaned = ndimage.binary_fill_holes(cleaned)
        except Exception:
            pass

    return cleaned


def _laplacian_smooth_numpy(verts: np.ndarray, faces: np.ndarray, iterations: int = 5, factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy/SciPy fallback for mesh smoothing."""
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts, faces

    v = verts.copy().astype(np.float64)
    adj = [set() for _ in range(len(verts))]
    for f in faces:
        adj[f[0]].add(f[1]); adj[f[0]].add(f[2])
        adj[f[1]].add(f[0]); adj[f[1]].add(f[2])
        adj[f[2]].add(f[0]); adj[f[2]].add(f[1])

    for _ in range(iterations):
        v_next = v.copy()
        for i, neighbors in enumerate(adj):
            if neighbors:
                nbr_arr = np.array(list(neighbors), dtype=int)
                centroid = np.mean(v[nbr_arr], axis=0)
                v_next[i] = v[i] + factor * (centroid - v[i])
        v = v_next

    return v.astype(verts.dtype), faces


def smooth_and_simplify_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    smoothing_iterations: int = 15,
    smoothing_method: str = "taubin",
    decimate_target: Optional[float] = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth mesh surface using Taubin / Laplacian filters and optionally decimate.
    Taubin smoothing eliminates voxel stepping without volume shrinkage.
    """
    if len(verts) < 10 or len(faces) < 10:
        return verts, faces

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        if decimate_target is not None and 0.1 <= decimate_target < 1.0 and len(mesh.faces) > 3000:
            target_faces = max(1000, int(len(mesh.faces) * decimate_target))
            try:
                mesh = mesh.simplify_quadratic_decimation(target_faces)
            except Exception:
                pass

        if smoothing_iterations > 0:
            if smoothing_method == "taubin":
                trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)
            else:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

        return np.array(mesh.vertices), np.array(mesh.faces)
    except Exception:
        return _laplacian_smooth_numpy(verts, faces, iterations=min(smoothing_iterations, 8))


# ─────────────────────────────────────────────────────────────────────────────
# 3D Surface Mesh Extraction (Preserving Physical Voxel Spacing)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mesh_for_label(
    seg_vol: np.ndarray,
    label_indices: List[int],
    step_size: int = 1,
    voxel_spacing: Optional[Tuple[float, float, float]] = None,
    min_voxels: int = 35,
    keep_largest_component: bool = True,
    apply_morphology: bool = True,
    smoothing_iterations: int = 12,
    decimate_target: Optional[float] = 0.85,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract a clean 3D polygon surface mesh for specified label(s) using Marching Cubes.
    """
    try:
        from skimage import measure

        raw_mask = np.isin(seg_vol, label_indices).astype(np.uint8)
        if raw_mask.sum() < min_voxels:
            return None, None

        keep_n = 1 if keep_largest_component else 9999
        clean_mask = clean_binary_mask(
            raw_mask,
            min_voxels=min_voxels,
            keep_largest_n=keep_n,
            apply_morphology=apply_morphology,
        )
        if clean_mask.sum() < min_voxels:
            return None, None

        mc_spacing = voxel_spacing if voxel_spacing is not None else (1.0, 1.0, 1.0)
        verts, faces, normals, values = measure.marching_cubes(
            clean_mask,
            level=0.5,
            spacing=mc_spacing,
            step_size=step_size,
        )

        if len(verts) < 4 or len(faces) < 4:
            return None, None

        if smoothing_iterations > 0 or decimate_target is not None:
            verts, faces = smooth_and_simplify_mesh(
                verts,
                faces,
                smoothing_iterations=smoothing_iterations,
                smoothing_method="taubin",
                decimate_target=decimate_target,
            )

        return verts, faces
    except Exception:
        return None, None


def export_binary_stl(verts: np.ndarray, faces: np.ndarray, output_path: str):
    """
    Export vertices and triangular faces to a standard binary STL file in physical coordinates.
    """
    import struct
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lens[norm_lens == 0] = 1.0
    normals = normals / norm_lens

    num_triangles = len(faces)

    with open(output_path, "wb") as f:
        header = b"OralSeg 3D Dental Model Binary STL Export" + b"\0" * (80 - len("OralSeg 3D Dental Model Binary STL Export"))
        f.write(header)
        f.write(struct.pack("<I", num_triangles))

        for i in range(num_triangles):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *v0[i]))
            f.write(struct.pack("<3f", *v1[i]))
            f.write(struct.pack("<3f", *v2[i]))
            f.write(struct.pack("<H", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly 3D Dental Viewer (Exact Medical Report Style)
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_interactive_figure(
    seg_vol: np.ndarray,
    voxel_spacing: Optional[Tuple[float, float, float]] = None,
    show_maxilla: bool = True,
    show_mandible: bool = True,
    show_teeth: bool = True,
    show_canal: bool = True,
    selected_teeth: Optional[List[int]] = None,
    maxilla_opacity: float = 0.45,
    mandible_opacity: float = 0.45,
    teeth_opacity: float = 1.0,
    canal_opacity: float = 1.0,
    smoothing_iterations: int = 15,
    camera_preset: str = "oblique",
    height: int = 590,
):
    """
    Build an interactive 3D dental anatomy model matching the report visualization:
    - Translucent Blue Maxilla (label 1)
    - Translucent Olive Green Mandible (label 2)
    - Highlighted Red Mandibular Canal (label 35) running inside mandible
    - Individual FDI teeth with distinct vivid clinical colors
    - Clean legend with exact labels (Maxilla, Mandible, Tooth 12, ..., Mandibular Canal)
    - Dark black background with crisp lighting and anatomical proportions
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    all_verts_list = []

    # Lighting configurations for clinical realism
    bone_lighting = dict(ambient=0.45, diffuse=0.70, specular=0.30, roughness=0.45)
    tooth_lighting = dict(ambient=0.55, diffuse=0.85, specular=0.50, roughness=0.18, fresnel=0.1)
    canal_lighting = dict(ambient=0.75, diffuse=0.90, specular=0.60, roughness=0.10)

    # ── 1. Maxilla (Upper Jaw Bone - Translucent Blue) ───────────────────────
    if show_maxilla:
        v, f = extract_mesh_for_label(
            seg_vol,
            [1],
            step_size=2,
            voxel_spacing=voxel_spacing,
            min_voxels=100,
            keep_largest_component=True,
            smoothing_iterations=smoothing_iterations,
            decimate_target=0.75,
        )
        if v is not None:
            all_verts_list.append(v)
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#2563EB",
                opacity=maxilla_opacity,
                name="Maxilla",
                lighting=bone_lighting,
                hoverinfo="name",
                flatshading=False,
            ))

    # ── 2. Mandible (Lower Jaw Bone - Translucent Green) ─────────────────────
    if show_mandible:
        v, f = extract_mesh_for_label(
            seg_vol,
            [2],
            step_size=2,
            voxel_spacing=voxel_spacing,
            min_voxels=100,
            keep_largest_component=True,
            smoothing_iterations=smoothing_iterations,
            decimate_target=0.75,
        )
        if v is not None:
            all_verts_list.append(v)
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#689F38",
                opacity=mandible_opacity,
                name="Mandible",
                lighting=bone_lighting,
                hoverinfo="name",
                flatshading=False,
            ))

    # ── 3. Individual FDI Teeth (Labels 3 to 34) ──────────────────────────────
    if show_teeth:
        unique_labels = set(np.unique(seg_vol))

        for label_idx in range(3, 35):
            if label_idx not in unique_labels:
                continue

            fdi_num = TOOTH_LABEL_TO_FDI.get(label_idx)
            if selected_teeth is not None and fdi_num not in selected_teeth:
                continue

            tooth_name = TOOTH_NAMES.get(fdi_num, f"Tooth {fdi_num}")
            tooth_color = FDI_3D_COLORS.get(fdi_num, "#FFF59D")

            v, f = extract_mesh_for_label(
                seg_vol,
                [label_idx],
                step_size=1,
                voxel_spacing=voxel_spacing,
                min_voxels=30,
                keep_largest_component=True,
                smoothing_iterations=smoothing_iterations,
                decimate_target=0.90,
            )
            if v is not None:
                all_verts_list.append(v)
                fig.add_trace(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color=tooth_color,
                    opacity=teeth_opacity,
                    name=f"Tooth {fdi_num}",
                    lighting=tooth_lighting,
                    hovertemplate=f"<b>Tooth {fdi_num}</b><br>{tooth_name}<extra></extra>",
                    flatshading=False,
                ))

    # ── 4. Mandibular Canal (Nerve Tube - Vibrant Red) ───────────────────────
    if show_canal:
        v, f = extract_mesh_for_label(
            seg_vol,
            [35],
            step_size=1,
            voxel_spacing=voxel_spacing,
            min_voxels=25,
            keep_largest_component=False,
            smoothing_iterations=smoothing_iterations,
            decimate_target=0.90,
        )
        if v is not None:
            all_verts_list.append(v)
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#FF1744",
                opacity=canal_opacity,
                name="Mandibular Canal",
                lighting=canal_lighting,
                hoverinfo="name",
                flatshading=False,
            ))

    # ── Camera Presets & Medical Viewing Perspective ─────────────────────────
    camera_presets = {
        "oblique": dict(eye=dict(x=1.7, y=-1.6, z=0.7), up=dict(x=0, y=0, z=1)),
        "anterior": dict(eye=dict(x=0.0, y=-2.2, z=0.1), up=dict(x=0, y=0, z=1)),
        "upper_occlusal": dict(eye=dict(x=0.0, y=0.0, z=2.2), up=dict(x=0, y=1, z=0)),
        "lower_occlusal": dict(eye=dict(x=0.0, y=0.0, z=-2.2), up=dict(x=0, y=-1, z=0)),
        "right_lateral": dict(eye=dict(x=2.2, y=0.0, z=0.1), up=dict(x=0, y=0, z=1)),
        "left_lateral": dict(eye=dict(x=-2.2, y=0.0, z=0.1), up=dict(x=0, y=0, z=1)),
    }
    cam_dict = camera_presets.get(camera_preset, camera_presets["oblique"])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="#000000", showgrid=False),
            yaxis=dict(visible=False, backgroundcolor="#000000", showgrid=False),
            zaxis=dict(visible=False, backgroundcolor="#000000", showgrid=False),
            bgcolor="#000000",
            aspectmode="data",
            camera=cam_dict,
        ),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(
            font=dict(color="#FFFFFF", size=10, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            bgcolor="rgba(0, 0, 0, 0.75)",
            bordercolor="#262626",
            borderwidth=1,
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            itemsizing="constant",
        ),
        height=height,
    )

    return fig
