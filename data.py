"""Dataset adapters. Each adapter returns a list of
    (image_id: str, path: str, label: int, source_id: str)
label: 1 = forged, 0 = authentic.
source_id: the *base* image identity used for leakage-safe grouping. For CoMoFoD
this MUST be the base-image number so all attack variants group together; for MICC
each image is its own source.

Dataset directory layouts vary by download, so the parsing below is intentionally
thin. Fill in the TODOs for your local copies; the (id, path, label, source_id)
contract is all the rest of the pipeline depends on.
"""
import os, glob, re

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _is_img(p):
    return p.lower().endswith(IMG_EXTS)


def adapter_labelled_folders(root):
    """Generic adapter: root/authentic/*, root/forged/*. source_id = filename stem."""
    items = []
    for label, sub in ((0, "authentic"), (1, "forged")):
        for p in sorted(glob.glob(os.path.join(root, sub, "*"))):
            if _is_img(p):
                stem = os.path.splitext(os.path.basename(p))[0]
                items.append((f"{sub}/{stem}", p, label, stem))
    return items


def adapter_comofod(root):
    """CoMoFoD: filenames like 001_F.png (forged) / 001_O.png (original) plus many
    attack variants 001_F_BC1.png etc. source_id = the leading base number ('001')
    so every variant of one base image stays on the same side of the split.
    TODO: confirm the exact suffix convention of your CoMoFoD copy."""
    items = []
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if not _is_img(p):
            continue
        name = os.path.basename(p)
        base = name.split("_")[0]                 # '001'
        label = 0 if "_O" in name else 1          # TODO: verify O/F convention
        items.append((name, p, label, base))
    return items


def adapter_micc_f2000(root):
    """MICC-F2000 flat folder adapter.
    Authentic images look like 00007_scale.jpg or 00007_2_scale.jpg.
    Forged images look like 00007tamp1.jpg.
    source_id is the leading digits extracted from the filename.
    """
    items = []
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if not _is_img(p):
            continue
        name = os.path.basename(p)
        stem, _ = os.path.splitext(name)
        src_match = re.match(r"^(\d+)", stem)
        source_id = src_match.group(1) if src_match else stem
        label = 1 if "tamp" in stem else 0
        items.append((name, p, label, source_id))
    return items


# TODO: micc_f600 / micc_f2000 ship tampered+original lists; map each to
# (id, path, label, source_id=id). casia_cmfd: use the copy-move subset only.
ADAPTERS = {
    "micc_f600": adapter_labelled_folders,    # TODO: replace with real MICC parser
    "micc_f2000": adapter_micc_f2000,
    "comofod": adapter_comofod,
    "casia_cmfd": adapter_labelled_folders,   # TODO: copy-move subset only
}


def load_all(datasets_cfg):
    items = []
    seen = set()
    for d in datasets_cfg:
        name, root = d["name"], d["root"]
        if name not in ADAPTERS:
            raise KeyError(f"No adapter for dataset '{name}'. Add one to ADAPTERS.")
        for iid, path, label, src in ADAPTERS[name](root):
            gid = f"{name}:{iid}"                       # globally unique image id
            gsrc = f"{name}:{src}"                      # namespace source ids per dataset
            if gid in seen:
                continue
            seen.add(gid)
            items.append((gid, path, label, gsrc))
    if not items:
        raise RuntimeError("No images loaded. Check dataset roots in config.yaml.")
    return items
