"""
make_architecture_diagram.py

Master architecture figure: BusterNet + VQC fusion (amplitude encoding).

Run  make_diagram_assets.py  first -- it produces every embedded sub-image from
the real code / real checkpoint / real dataset.

Then:   python make_architecture_diagram.py
Outputs: architecture_diagram.png (raster) and architecture_diagram.pdf (vector)

Canvas is 140 x 100 data units on a 28 x 20 inch figure, so both axes are exactly
5 units/inch -- embedded images keep their true aspect ratio automatically.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "diagram_assets")
os.chdir(HERE)

C_MANI, C_MANI_E = "#F7D9DC", "#B5626C"
C_SIMI, C_SIMI_E = "#D3E3F6", "#4A72A8"
C_FUSE, C_FUSE_E = "#D9EFD9", "#4E8A4E"
C_VQC, C_VQC_E = "#E7DCF5", "#6F4FA8"
C_IO, C_IO_E = "#ECECEC", "#666666"
C_SIDE, C_SIDE_E = "#FDF3D6", "#B08B26"
C_W = "#FFFFFF"
FS_T, FS_B, FS_L, FS_N = 12.0, 8.6, 7.8, 7.4


def box(ax, x, y, w, h, text, fc=C_W, ec="#333333", fs=FS_B, dashed=False, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                linewidth=1.2, facecolor=fc, edgecolor=ec, zorder=3,
                                linestyle=(0, (4, 2)) if dashed else "solid"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=4, weight=weight, linespacing=1.55)


def container(ax, x, y, w, h, title, fc, ec, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.0",
                                linewidth=1.8, facecolor=fc, edgecolor=ec, zorder=z, alpha=0.5))
    ax.text(x + 1.1, y + h - 1.6, title, ha="left", va="center", fontsize=FS_T,
            weight="bold", color=ec, zorder=z + 2)


def arrow(ax, x1, y1, x2, y2, label=None, lx=None, ly=None, fs=FS_L, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color="#222222", zorder=6))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2,
                label, ha="center", va="center", fontsize=fs, zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))


def elbow(ax, pts, label=None, lx=None, ly=None, fs=FS_L, lw=1.5):
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color="#222222", lw=lw, zorder=6)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color="#222222", zorder=6))
    if label:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fs, zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))


def img(ax, name, x, y, w, border=True, trim=False, max_h=None, cx=None, crop_r=None):
    """Place an asset image at its true aspect ratio.
    trim   -- strip white margins first
    crop_r -- keep only this left fraction of the width (applied BEFORE trim;
              used to cut the paper's own "(b)" sub-caption off Fig. 3(b))
    max_h -- cap the height, shrinking the width to match (keeps aspect)
    cx    -- if given, centre the (possibly shrunk) image on this x instead of x
    """
    im = mpimg.imread(os.path.join(A, name))
    if crop_r is not None:
        im = im[:, :int(im.shape[1] * crop_r)]
    if trim:
        g = im[..., :3].mean(axis=2) if im.ndim == 3 else im
        m = g < 0.97
        r, c = np.any(m, axis=1), np.any(m, axis=0)
        if r.any() and c.any():
            r0, r1 = np.where(r)[0][[0, -1]]
            c0, c1 = np.where(c)[0][[0, -1]]
            im = im[r0:r1 + 1, c0:c1 + 1]
    ar = im.shape[0] / im.shape[1]
    h = w * ar
    if max_h is not None and h > max_h:
        h = max_h
        w = h / ar
    if cx is not None:
        x = cx - w / 2
    ax.imshow(im, extent=[x, x + w, y, y + h], aspect="auto", zorder=5,
              interpolation="antialiased")
    if border:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#555555", lw=0.9, zorder=6))
    return h


def volume(ax, x, y, w, h, n, fc, ec, label):
    for i in range(n):
        o = i * 0.45
        ax.add_patch(Rectangle((x + o, y + o), w, h, facecolor=fc, edgecolor=ec,
                               lw=1.0, zorder=3 + i, alpha=0.95))
    ax.text(x + w / 2 + n * 0.22, y - 1.4, label, ha="center", va="top", fontsize=FS_N, zorder=6)


fig, ax = plt.subplots(figsize=(28, 20))
ax.set_xlim(0, 140)
ax.set_ylim(0, 100)
ax.axis("off")

for sx in (48.5, 97.0):
    ax.plot([sx, sx], [1.5, 95.5], linestyle=(0, (6, 4)), color="#AAAAAA", lw=1.4, zorder=0)

ax.text(70, 98.8, "BusterNet + VQC Fusion   —   Quantum-Hybrid Copy-Move Forgery Detection",
        ha="center", fontsize=19, weight="bold", color="#111111")
for tx, tt in ((24, "STAGE 1  |  Input & Two-Branch Feature Extraction"),
               (72, "STAGE 2  |  Fusion & Quantum Decision Layer"),
               (118, "STAGE 3  |  Prediction & Output")):
    ax.text(tx, 96.3, tt, ha="center", fontsize=13.5, weight="bold", color="#222222")

# =====================================================================
# STAGE 1
# =====================================================================
img(ax, "sample_input.png", 3, 85.0, 10.5)
ax.text(8.25, 83.6, "$X$ :  3 x 256 x 256", ha="center", va="top", fontsize=FS_N)
box(ax, 18, 87.0, 17, 5.4, "BusterNetPreprocess\nBGR convert + mean subtract", fc=C_IO, ec=C_IO_E)
arrow(ax, 13.8, 89.7, 17.8, 89.7)
elbow(ax, [(26.5, 86.8), (26.5, 84.5), (1.3, 84.5), (1.3, 73.5), (4.3, 73.5)])
elbow(ax, [(1.3, 73.5), (1.3, 49.2), (4.3, 49.2)])

# ---- Mani-Det ----
container(ax, 2, 58.5, 45.5, 22.5, "Mani-Det branch", C_MANI, C_MANI_E)
box(ax, 4.5, 70.8, 13, 5.4, "VGG16\nfirst 4 blocks\n(random init)", ec=C_MANI_E, fs=FS_N)
arrow(ax, 17.7, 73.5, 20.3, 73.5)
volume(ax, 20.5, 70.6, 3.2, 5.6, 4, "#F2C6CB", C_MANI_E, "$f_m$ : 512x16x16")
arrow(ax, 25.7, 73.5, 28.0, 73.5)
box(ax, 28.2, 68.6, 18, 9.0, "", ec=C_MANI_E)
ax.text(37.2, 76.9, "ManiDecoder", ha="center", fontsize=FS_N, weight="bold", color=C_MANI_E)
img(ax, "busternet_fig3a_decoder.png", 29.0, 69.2, 16.4, border=False, trim=True)
arrow(ax, 37.2, 68.4, 37.2, 66.0, label="$d_m$ : 6 x 256 x 256", lx=37.2, ly=67.2)
box(ax, 27.0, 60.4, 20.0, 5.0, "Aux head  (training only)\nConv2d(6→1) + sigmoid", ec="#777777",
    dashed=True, fs=FS_N)
ax.text(14.0, 62.9, "$M_m$ :  manipulation mask\n→  auxiliary BCE loss", ha="center", va="center",
        fontsize=FS_N, style="italic", color="#555555")

# ---- Simi-Det ----
container(ax, 2, 13.2, 45.5, 42.8, "Simi-Det branch", C_SIMI, C_SIMI_E)
box(ax, 4.5, 46.5, 13, 5.4, "VGG16\nfirst 4 blocks\n(ImageNet-pretrained)", ec=C_SIMI_E, fs=FS_N)
arrow(ax, 17.7, 49.2, 20.3, 49.2)
volume(ax, 20.5, 46.4, 3.2, 5.6, 4, "#C3D8F2", C_SIMI_E, "$f_s$ : 512x16x16")
arrow(ax, 25.7, 49.2, 28.0, 49.2)
box(ax, 28.2, 46.5, 18, 5.4, "std_norm_along_chs\n$\\tilde f[i]=(f[i]-\\mu)/\\sigma$",
    ec=C_SIMI_E, fs=FS_N)
arrow(ax, 37.2, 46.3, 37.2, 44.0)

container(ax, 3.5, 24.5, 42.5, 18.5, "Self-Correlation + Percentile Pooling   (paper Eq. 1–6)",
          "#E8F1FC", C_SIMI_E, z=2)
ax.text(24.5, 39.6, "$\\rho(i,j)=\\tilde f[i]^{T}\\tilde f[j]\\,/\\,512$", ha="center",
        fontsize=10.5, zorder=6)
img(ax, "selfcorr_heatmap.png", 5.5, 28.4, 9.6)
ax.text(10.3, 27.6, "real corr. map, centre patch\n$S$ : 16x16x256", ha="center", va="top",
        fontsize=FS_N)
arrow(ax, 15.6, 33.2, 18.0, 33.2)
img(ax, "selfcorr_curve.png", 18.2, 29.2, 15.0, border=False, trim=True)
ax.text(25.7, 27.6, "sorted scores;  red = the K=256 percentile taps", ha="center", va="top",
        fontsize=FS_N)
arrow(ax, 34.0, 33.2, 36.6, 33.2)
volume(ax, 36.9, 30.6, 2.8, 5.2, 4, "#C3D8F2", C_SIMI_E, "$P$ : 16x16x256")

arrow(ax, 24.5, 24.3, 24.5, 22.0, label="BatchNorm", lx=24.5, ly=23.1)
box(ax, 4.5, 16.6, 19.5, 5.2, "SimilarityDecoder\nBN-Inception + bilinear up x4\n(separate weights)",
    ec=C_SIMI_E, fs=FS_N)
arrow(ax, 24.2, 19.2, 26.5, 19.2)
box(ax, 26.7, 16.6, 20.0, 5.2, "Aux head  (training only)\nsigmoid", ec="#777777", dashed=True,
    fs=FS_N)
ax.text(25.5, 15.8, "$d_s$ : 6 x 256 x 256          $M_s$ :  similarity mask  →  auxiliary BCE loss",
        ha="center", va="top", fontsize=FS_N, style="italic", color="#555555")

elbow(ax, [(46.6, 73.5), (50.4, 73.5), (50.4, 92.2), (52.3, 92.2)], label="$d_m$", lx=50.4, ly=80.0)
elbow(ax, [(14.0, 16.5), (14.0, 11.2), (49.5, 11.2), (49.5, 90.2), (52.3, 90.2)],
      label="$d_s$", lx=49.5, ly=52.0)

# =====================================================================
# STAGE 2
# =====================================================================
box(ax, 52.5, 89.0, 41, 5.0, "Concatenate   $d_s \\Vert d_m$", fc=C_FUSE, ec=C_FUSE_E, weight="bold")
volume(ax, 88.0, 89.4, 2.6, 4.0, 5, "#BFE3BF", C_FUSE_E, "12 x 256 x 256")
arrow(ax, 70, 88.8, 70, 86.4)

box(ax, 57.0, 74.5, 26, 11.6, "", fc=C_FUSE, ec=C_FUSE_E)
ax.text(70.0, 84.6, "BnInception(12 → 9)      [ KEPT — classical, unchanged ]", ha="center",
        fontsize=FS_B, weight="bold", color="#14421A")
img(ax, "busternet_fig3b_bninception.png", 0, 75.4, 22.0, border=False, trim=True,
    max_h=8.2, cx=70.0, crop_r=0.54)
arrow(ax, 70, 74.3, 70, 71.6, label="9 x 256 x 256", lx=70, ly=72.9)
ax.text(70, 70.8, "per-pixel reshape   [B, 9, H, W] → [B·H·W, 9]   ·   65,536 circuit calls / image",
        ha="center", va="top", fontsize=FS_N, color=C_VQC_E, style="italic")
arrow(ax, 70, 68.2, 70, 66.2)

# ---- VQC (amplitude arm only) ----
container(ax, 50.5, 29.5, 45.0, 35.5,
          "VQC  —  Amplitude encoding, 4 qubits   ·   replaces Conv2d(9→3)", C_VQC, C_VQC_E)
box(ax, 53.0, 56.5, 40.0, 4.6,
    "BatchNorm1d(9)   +   $\\epsilon$ = 1e-6\n(avoids AmplitudeEmbedding's 0/0 on all-zero pixels)",
    ec=C_VQC_E, fs=FS_N)
arrow(ax, 66.0, 56.3, 66.0, 54.6)
ax.text(66.0, 53.9,
        "$|\\psi_x\\rangle=\\sum_i x_i|i\\rangle$      ·      L2-normalize,  pad 9 → 16 = $2^4$",
        ha="center", va="top", fontsize=9.2)
img(ax, "circuit_amplitude.png", 53.0, 44.0, 33.0, border=False, trim=True)
ax.text(69.5, 43.2, "the actual committed circuit:   state prep $|\\Psi\\rangle$   →   "
                    "4 x [ $R_Y(\\theta)$ on 4 qubits  +  linear CNOT chain ]   →   "
                    "$\\langle Z\\rangle$ readout x4", ha="center", va="top", fontsize=FS_N)
arrow(ax, 66.0, 42.0, 66.0, 39.6)
box(ax, 53.0, 34.0, 17.0, 5.4, "$\\langle Z\\rangle$  readout\nx 4 qubits", ec=C_VQC_E, fs=FS_N)
box(ax, 86.4, 48.0, 7.0, 7.0, "$\\Vert x\\Vert_2$\nside-\nchannel", fc=C_SIDE, ec=C_SIDE_E, fs=FS_N)
elbow(ax, [(89.9, 56.3), (89.9, 55.6), (89.9, 55.2)])
elbow(ax, [(89.9, 47.8), (89.9, 39.9), (82.0, 39.9), (82.0, 39.1)])
box(ax, 74.0, 34.5, 16.0, 4.4, "concat  →  5 values", fc=C_SIDE, ec=C_SIDE_E, fs=FS_N)
arrow(ax, 70.2, 36.7, 73.8, 36.7)
ax.text(71.5, 32.6, "4 expectation values  +  1 magnitude value", ha="center", va="top",
        fontsize=FS_N, style="italic", color="#4A3A6B")

elbow(ax, [(90.1, 36.7), (96.0, 36.7), (96.0, 91.3), (99.8, 91.3)],
      label="5 values", lx=96.0, ly=44.0)

# =====================================================================
# STAGE 3
# =====================================================================
box(ax, 100, 88.8, 37, 5.0, "Reshape back   [B·H·W, 5] → [B, 5, H, W]", fc=C_IO, ec=C_IO_E)
arrow(ax, 118.5, 88.6, 118.5, 85.8)
box(ax, 100, 79.0, 37, 6.6,
    "Conv2d(5 → 3, kernel = 1)     [ NEW — classical post-layer ]\n"
    "the post-processing layer of the dressed circuit", fc=C_VQC, ec=C_VQC_E)
arrow(ax, 118.5, 78.8, 118.5, 76.0)
box(ax, 100, 70.6, 37, 5.0, "Softmax over 3 classes, per pixel", fc=C_FUSE, ec=C_FUSE_E)
arrow(ax, 118.5, 70.4, 118.5, 67.6)
box(ax, 100, 61.0, 37, 6.4,
    "Output mask  $M_c$ :  3 x 256 x 256\npristine / source copy / target copy",
    fc=C_IO, ec=C_IO_E, weight="bold")

img(ax, "sample_input.png", 101.5, 42.0, 16.0)
img(ax, "sample_gt.png", 120.0, 42.0, 16.0)
ax.text(109.5, 41.0, "input", ha="center", va="top", fontsize=FS_N)
ax.text(128.0, 41.0, "target $M_c$", ha="center", va="top", fontsize=FS_N)
ax.text(118.5, 37.6, "blue = pristine        green = source copy        red = target copy",
        ha="center", va="top", fontsize=FS_N, style="italic", color="#555555")

container(ax, 100, 6.0, 37, 26.0, "Legend", "#FAFAFA", "#777777")
rows = [(C_MANI, C_MANI_E, "Mani-Det branch  (manipulation artifacts)"),
        (C_SIMI, C_SIMI_E, "Simi-Det branch  (copy-move similarity)"),
        (C_FUSE, C_FUSE_E, "Classical fusion / softmax  (unchanged)"),
        (C_VQC, C_VQC_E, "Quantum contribution  (this project)"),
        (C_SIDE, C_SIDE_E, "Magnitude side-channel")]
for i, (fc, ec, lab) in enumerate(rows):
    yy = 26.6 - i * 3.5
    ax.add_patch(FancyBboxPatch((101.5, yy), 3.2, 2.0, boxstyle="round,pad=0.02,rounding_size=0.35",
                                facecolor=fc, edgecolor=ec, lw=1.2, zorder=4))
    ax.text(106.2, yy + 1.0, lab, ha="left", va="center", fontsize=FS_N, zorder=4)
ax.add_patch(FancyBboxPatch((101.5, 9.1), 3.2, 2.0, boxstyle="round,pad=0.02,rounding_size=0.35",
                            facecolor=C_W, edgecolor="#777777", lw=1.2, linestyle=(0, (4, 2)),
                            zorder=4))
ax.text(106.2, 10.1, "Dashed = active during branch training only", ha="left", va="center",
        fontsize=FS_N, zorder=4)
ax.text(118.5, 7.4, "Decoder / BN-Inception panels: BusterNet, Wu et al., ECCV 2018, Fig. 3",
        ha="center", fontsize=6.8, style="italic", color="#777777")

fig.savefig("architecture_diagram.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig("architecture_diagram.pdf", bbox_inches="tight", facecolor="white")
print("saved architecture_diagram.png and architecture_diagram.pdf")
