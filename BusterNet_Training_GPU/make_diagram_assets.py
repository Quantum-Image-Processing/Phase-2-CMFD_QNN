"""
make_diagram_assets.py

Generates every sub-image the master architecture figure embeds -- all from the
REAL code, the REAL pretrained checkpoint, and the REAL dataset. Nothing here is
mocked or hand-drawn.

Outputs into  diagram_assets/ :
    circuit_amplitude.png      qml.draw_mpl of the actual AmplitudeVQC qnode
    circuit_reupload.png       qml.draw_mpl of the actual ReuploadVQC qnode
    sample_input.png           a real forged image from coco_subset
    sample_gt.png              its real 3-class ground-truth mask (blue/green/red)
    sample_pred.png            busternet_converted.pth's actual prediction on it
    selfcorr_heatmap.png       real 16x16 self-correlation map from that image
    selfcorr_curve.png         its sorted score curve + the K percentile taps
    feat9_values.txt           one real pixel's 9-channel fused vector
    busternet_fig3.png         Fig. 3 (decoder module) lifted from the paper PDF
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagram_assets")
os.makedirs(OUT, exist_ok=True)
os.chdir(HERE)

# ---------------------------------------------------------------- circuits
from vqc_fusion import (_amplitude_circuit, _reupload_circuit, AMP_QUBITS,
                        AMP_LAYERS, REUP_QUBITS, REUP_ROUNDS, REUP_LAYERS_PER_ROUND)

fig, _ = qml.draw_mpl(_amplitude_circuit, style="black_white")(
    torch.zeros(9), torch.zeros(AMP_LAYERS, AMP_QUBITS))
fig.savefig(f"{OUT}/circuit_amplitude.png", dpi=210, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

fig, _ = qml.draw_mpl(_reupload_circuit, style="black_white")(
    torch.zeros(9), torch.zeros(REUP_ROUNDS * REUP_LAYERS_PER_ROUND, REUP_QUBITS))
fig.savefig(f"{OUT}/circuit_reupload.png", dpi=210, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("[ok] circuit diagrams")

# ------------------------------------------------- real data + real model
from dataset import SyntheticCMFDataset
from model_busternet import build_busternet_full
from model_utils import std_norm_along_chs, self_corr_and_percentile

ds = SyntheticCMFDataset(manifest_csv="coco_subset/forgery_manifest_with_masks.csv",
                         root="coco_subset", img_size=256)

model = build_busternet_full(target_size=256)
sd = torch.load("busternet_converted.pth", map_location="cpu")
model.load_state_dict(sd if "state_dict" not in sd else sd["state_dict"], strict=False)
model.eval()

# pick a sample whose mask actually contains both source and target classes,
# otherwise the figure shows a mask with nothing interesting in it
CMAP = np.array([[47, 79, 168], [47, 168, 79], [192, 57, 43]], dtype=np.uint8)
chosen = None
for idx in range(0, 400, 7):
    s = ds[idx]
    u = torch.unique(s["mc"])
    if 1 in u and 2 in u:
        chosen = s
        break
if chosen is None:
    chosen = ds[0]

img = chosen["image"]                                   # [3,256,256] in [0,1]
plt.imsave(f"{OUT}/sample_input.png", img.permute(1, 2, 0).numpy())
plt.imsave(f"{OUT}/sample_gt.png", CMAP[chosen["mc"][0].numpy()])

with torch.no_grad():
    out = model(img.unsqueeze(0))
pred = out["mc_pred"][0].argmax(0).numpy()
plt.imsave(f"{OUT}/sample_pred.png", CMAP[pred])
print("[ok] real input / ground-truth / prediction images")

# ------------------------------------ real self-correlation + percentile taps
with torch.no_grad():
    x = model.preprocess(img.unsqueeze(0))
    f = std_norm_along_chs(model.simidet.backbone(x))    # [1,512,16,16]
    S, P = self_corr_and_percentile(f, K=256)

heat = S[0, 8, 8].reshape(16, 16).numpy()                # centre patch vs all others
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.imshow(heat, cmap="viridis")
ax.set_xticks([]); ax.set_yticks([])
fig.savefig(f"{OUT}/selfcorr_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

sorted_scores = np.sort(S[0, 8, 8].numpy())[::-1]
taps = np.round(np.linspace(1, len(sorted_scores) - 1, 16)).astype(int)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.plot(sorted_scores, color="#4A72A8", lw=2.0)
ax.plot(taps, sorted_scores[taps], "o", ms=4.0, color="#C0392B")
ax.set_xlabel("rank", fontsize=8); ax.set_ylabel("corr. score", fontsize=8)
ax.tick_params(labelsize=7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(f"{OUT}/selfcorr_curve.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[ok] self-correlation heatmap + percentile curve")

# ------------------------------------------- a real 9-channel fused vector
with torch.no_grad():
    fused = torch.cat([out["simi_feat"], out["mani_feat"]], dim=1)   # [1,12,256,256]
    nine = model.fusion(fused)[0, :, 128, 128]                        # centre pixel
np.savetxt(f"{OUT}/feat9_values.txt", nine.numpy(), fmt="%.2f")
print("[ok] real 9-channel vector:", np.round(nine.numpy(), 2))

# ------------------------------------------- BusterNet Fig. 3 from the PDF
try:
    import fitz
    pdf = None
    for cand in [os.path.expanduser("~/Downloads/Rex_Yue_Wu_BusterNet_Detecting_Copy-Move_ECCV_2018_paper.pdf"),
                 os.path.expanduser("~/Downloads/Rex_Yue_Wu_BusterNet_Detecting_Copy-Move_ECCV_2018_paper (1).pdf")]:
        if os.path.exists(cand):
            pdf = cand
            break
    doc = fitz.open(pdf)
    page = doc[4]                       # page 5 holds Fig. 3 (decoder module)
    r = page.rect
    top, bot = r.y0 + 0.487 * r.height, r.y0 + 0.610 * r.height
    # (a) the decoder chain  and  (b) the BN-Inception block, cropped separately
    page.get_pixmap(clip=fitz.Rect(r.x0 + 0.105 * r.width, top,
                                   r.x0 + 0.575 * r.width, bot),
                    dpi=340).save(f"{OUT}/busternet_fig3a_decoder.png")
    page.get_pixmap(clip=fitz.Rect(r.x0 + 0.575 * r.width, top,
                                   r.x1 - 0.085 * r.width, bot),
                    dpi=340).save(f"{OUT}/busternet_fig3b_bninception.png")
    print("[ok] BusterNet Fig. 3(a) and 3(b) extracted from", os.path.basename(pdf))
except Exception as e:
    print("[warn] could not extract Fig. 3:", e)

print("\nassets written to", OUT)
