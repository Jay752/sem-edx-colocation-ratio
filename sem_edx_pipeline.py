#!/usr/bin/env python3
"""
SEM–EDX two-step workflow (co-location + ratio gating).

Features
- Load Bruker .bcf hypermaps (HyperSpy) OR pre-exported 2D element maps.
- Preprocess (percentile clip + optional denoise).
- Build per-element masks M_E (threshold rules).
- Co-location via AND on include list; optional NOT via exclude list.
- Pixel-wise ratio gating (windows) on co-located pixels.
- Outputs: overlay PNG, final mask NPY, JSON summary (element sums, total ratios).

Limitations
- .spx (Bruker) = single spectrum; no pixel grid ⇒ not supported for mapping.

CLI
    python sem_edx_pipeline.py --config config.yaml

License: MIT
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np
from skimage import io as skio, color, util
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

# ----------------------------- Config model ------------------------------ #
@dataclass
class PreprocessCfg:
    clip_percent: Tuple[float, float] = (1, 99)
    denoise: Optional[str] = "bilateral"   # 'bilateral' | 'tv' | None
    bilateral_sigma_color: float = 0.05
    bilateral_sigma_spatial: float = 2.0
    tv_weight: float = 0.05

@dataclass
class MaskRule:
    method: str = "percentile"             # 'percentile' | 'otsu' | 'absolute'
    p: Optional[float] = 90.0              # for percentile
    value: Optional[float] = None          # for absolute

@dataclass
class CoLocationCfg:
    include: List[str] = field(default_factory=lambda: ["Ca", "O"])
    exclude: List[str] = field(default_factory=list)

@dataclass
class OverlayCfg:
    color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    alpha: float = 0.35

@dataclass
class PipelineCfg:
    # One of:
    bruker_path: Optional[str] = None      # expects .bcf
    maps_dir: Optional[str] = None         # directory with per-element maps
    sem_path: Optional[str] = None         # required if using maps_dir

    elements: List[str] = field(default_factory=lambda: ["C", "O", "Ca", "Si"])
    preprocess: PreprocessCfg = PreprocessCfg()
    mask_rules: Dict[str, MaskRule] = field(default_factory=dict)
    co_location: CoLocationCfg = CoLocationCfg()
    ratio_windows: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    overlay: OverlayCfg = OverlayCfg()
    save_dir: str = "./edx_outputs"
    save_prefix: str = "phase_detect"
    sums_chart: bool = True

# ----------------------------- Utils ------------------------------------ #
def robust_normalize(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    finite = np.isfinite(img)
    lo, hi = np.percentile(img[finite], [p_low, p_high])
    img = np.clip(img, lo, hi)
    return (img - lo) / max(hi - lo, 1e-12)

def preprocess_image(img: np.ndarray, cfg: PreprocessCfg) -> np.ndarray:
    out = robust_normalize(img, *cfg.clip_percent)
    if cfg.denoise == "bilateral":
        out = denoise_bilateral(out, sigma_color=cfg.bilateral_sigma_color,
                                sigma_spatial=cfg.bilateral_sigma_spatial, channel_axis=None)
    elif cfg.denoise == "tv":
        out = denoise_tv_chambolle(out, weight=cfg.tv_weight)
    return out

def build_mask(channel: np.ndarray, rule: MaskRule) -> np.ndarray:
    method = (rule.method or "percentile").lower()
    arr = channel[np.isfinite(channel)]
    if method == "percentile":
        thr = np.percentile(arr, rule.p if rule.p is not None else 90.0)
    elif method == "otsu":
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(arr)
    elif method == "absolute":
        thr = float(rule.value if rule.value is not None else 0.0)
    else:
        raise ValueError(f"Unknown threshold method: {rule.method}")
    return channel >= thr

def logical_and(masks: List[np.ndarray]) -> np.ndarray:
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out

def logical_or(masks: List[np.ndarray]) -> np.ndarray:
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out

def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / (den + eps)

def parse_ratio_key(key: str) -> Tuple[str, str]:
    if "/" not in key:
        raise ValueError(f"Ratio key must be 'A/B', got '{key}'")
    a, b = key.split("/")
    return a.strip(), b.strip()

def overlay_mask_on_sem(sem_gray: np.ndarray, mask: np.ndarray,
                        color_rgb: Tuple[float, float, float], alpha: float) -> np.ndarray:
    if sem_gray.dtype.kind in "ui":
        sem = sem_gray.astype(np.float32) / np.iinfo(sem_gray.dtype).max
    else:
        sem = sem_gray.astype(np.float32)
        if sem.max() > 1: sem /= (sem.max() + 1e-12)
    sem_rgb = color.gray2rgb(sem)
    overlay = np.zeros_like(sem_rgb)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_rgb
    out = sem_rgb.copy()
    out[mask] = (1 - alpha) * sem_rgb[mask] + alpha * overlay[mask]
    return out

# ----------------------------- Loaders ----------------------------------- #
def load_maps_from_dir(maps_dir: str, elements: List[str]) -> Dict[str, np.ndarray]:
    """
    Load per-element maps from files in maps_dir. Supports .npy and image files.
    Filenames must be like 'EDX_Ca.npy' or 'Ca.png' etc. Case-sensitive element keys.
    """
    maps: Dict[str, np.ndarray] = {}
    for el in elements:
        candidates = [
            os.path.join(maps_dir, f"{el}.npy"),
            os.path.join(maps_dir, f"EDX_{el}.npy"),
            os.path.join(maps_dir, f"{el}.png"),
            os.path.join(maps_dir, f"{el}.tif"),
            os.path.join(maps_dir, f"EDX_{el}.png"),
            os.path.join(maps_dir, f"EDX_{el}.tif"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"Missing map for element '{el}' in {maps_dir}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
        else:
            arr = util.img_as_float(skio.imread(path))
            if arr.ndim == 3:
                arr = color.rgb2gray(arr)
        maps[el] = arr.astype(np.float32)
    return maps

def load_bruker_bcf(bcf_path: str, elements: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load .bcf hypermap via HyperSpy and return (SEM_image, {element: 2D map}).
    """
    try:
        import hyperspy.api as hs
    except Exception as e:
        raise RuntimeError("HyperSpy is required to read .bcf. Install 'hyperspy[all]'") from e

    s = hs.load(bcf_path)
    signals = s if isinstance(s, list) else [s]
    eds = None
    sem_img = None

    for sig in signals:
        st = getattr(sig, "signal_type", "").lower()
        if "eds" in st and sig.axes_manager.navigation_dimension == 2:
            eds = sig
        if hasattr(sig, "metadata"):
            try:
                arr = np.squeeze(sig.data)
                if arr.ndim == 2 and arr.size > 0:
                    sem_img = arr.astype(np.float32)
            except Exception:
                pass

    if eds is None:
        raise RuntimeError("No 2D EDX hypermap found in .bcf.")

    eds.add_elements(list(set(elements)))
    try:
        be = eds.metadata.Acquisition_instrument.SEM.beam_energy
        eds.set_microscope_parameters(beam_energy=be)
    except Exception:
        pass
    li = eds.get_lines_intensity(elements=elements)
    maps = {el: np.array(li[el], dtype=np.float32) for el in elements}

    if sem_img is None:
        sem_img = np.array(eds.sum(axis=2), dtype=np.float32)

    return sem_img, maps

def load_inputs(cfg: PipelineCfg) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if cfg.bruker_path:
        ext = os.path.splitext(cfg.bruker_path)[1].lower()
        if ext == ".spx":
            raise RuntimeError(".spx is single-spectrum; re-export a .bcf hypermap for mapping.")
        if ext != ".bcf":
            raise RuntimeError(f"Unsupported Bruker format '{ext}'. Use .bcf.")
        return load_bruker_bcf(cfg.bruker_path, cfg.elements)

    if cfg.maps_dir:
        if not cfg.sem_path:
            raise ValueError("When using maps_dir, you must also provide sem_path.")
        sem = util.img_as_float(skio.imread(cfg.sem_path))
        if sem.ndim == 3:
            sem = color.rgb2gray(sem)
        maps = load_maps_from_dir(cfg.maps_dir, cfg.elements)
        H, W = sem.shape
        for el, arr in maps.items():
            if arr.shape != (H, W):
                raise ValueError(f"Shape mismatch for {el}: {arr.shape} != {(H,W)}")
        return sem.astype(np.float32), maps

    raise ValueError("Provide either 'bruker_path' (.bcf) OR 'maps_dir' + 'sem_path'.")

# ----------------------------- Pipeline ---------------------------------- #
def run_pipeline(cfg: PipelineCfg) -> Dict:
    os.makedirs(cfg.save_dir, exist_ok=True)

    sem, edx_maps = load_inputs(cfg)
    H, W = sem.shape

    edx_pre = {el: preprocess_image(edx_maps[el], cfg.preprocess) for el in cfg.elements}

    masks: Dict[str, np.ndarray] = {}
    for el in cfg.elements:
        rule = cfg.mask_rules.get(el, MaskRule(method="percentile", p=90.0))
        masks[el] = build_mask(edx_pre[el], rule)

    inc = cfg.co_location.include
    and_mask = logical_and([masks[e] for e in inc])
    if cfg.co_location.exclude:
        not_mask = ~logical_or([masks[e] for e in cfg.co_location.exclude])
        gated = and_mask & not_mask
    else:
        gated = and_mask

    ratio_gate = np.ones((H, W), dtype=bool)
    total_ratios: Dict[str, float] = {}
    ratio_maps: Dict[str, np.ndarray] = {}

    for rk, window in cfg.ratio_windows.items():
        a, b = parse_ratio_key(rk)
        if a not in edx_pre or b not in edx_pre:
            raise KeyError(f"Ratio {rk} requires {a} and {b} maps.")
        R = safe_ratio(edx_pre[a], edx_pre[b])
        ratio_maps[rk] = R
        rmin, rmax = float(window[0]), float(window[1])
        ratio_gate &= (R >= rmin) & (R <= rmax)

    final_mask = gated & ratio_gate

    overlay_img = overlay_mask_on_sem(
        sem_gray=sem,
        mask=final_mask,
        color_rgb=cfg.overlay.color_rgb,
        alpha=cfg.overlay.alpha,
    )

    sums = {el: float(np.nansum(edx_maps[el][final_mask])) for el in cfg.elements}

    for rk in cfg.ratio_windows:
        a, b = parse_ratio_key(rk)
        total_ratios[rk] = float(sums[a] / (sums[b] + 1e-12))

    prefix = os.path.join(cfg.save_dir, cfg.save_prefix)
    skio.imsave(prefix + "_overlay.png", (np.clip(overlay_img, 0, 1) * 255).astype(np.uint8))
    np.save(prefix + "_mask.npy", final_mask.astype(np.uint8))

    summary = {
        "elements": cfg.elements,
        "sums": sums,
        "total_ratios": total_ratios,
        "pixels_selected": int(final_mask.sum()),
        "image_shape": [int(H), int(W)],
        "co_location": {"include": cfg.co_location.include, "exclude": cfg.co_location.exclude},
        "ratio_windows": cfg.ratio_windows,
    }
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if cfg.sums_chart:
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 4))
            keys = list(sums.keys()); vals = [sums[k] for k in keys]
            plt.bar(keys, vals)
            plt.ylabel("Summed intensity (a.u.)")
            plt.title("EDX intensity sums over detected pixels")
            plt.tight_layout()
            fig.savefig(prefix + "_sums.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            print("Sums chart skipped:", e, file=sys.stderr)

    return summary

# ----------------------------- CLI --------------------------------------- #
def load_yaml_config(path: str) -> PipelineCfg:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load YAML configs. pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def to_dc(dc_cls, src, defaults=None):
        defaults = defaults or {}
        merged = {**defaults, **(src or {})}
        return dc_cls(**merged)

    preprocess = to_dc(PreprocessCfg, data.get("preprocess"))
    mask_rules = {k: to_dc(MaskRule, v) for k, v in (data.get("mask_rules", {}) or {}).items()}
    co_loc = to_dc(CoLocationCfg, data.get("co_location"))
    overlay = to_dc(OverlayCfg, data.get("overlay"))

    cfg = PipelineCfg(
        bruker_path=data.get("bruker_path"),
        maps_dir=data.get("maps_dir"),
        sem_path=data.get("sem_path"),
        elements=data.get("elements", ["C", "O", "Ca", "Si"]),
        preprocess=preprocess,
        mask_rules=mask_rules,
        co_location=co_loc,
        ratio_windows=data.get("ratio_windows", {}),
        overlay=overlay,
        save_dir=data.get("save_dir", "./edx_outputs"),
        save_prefix=data.get("save_prefix", "phase_detect"),
        sums_chart=bool(data.get("sums_chart", True)),
    )
    return cfg

def main():
    ap = argparse.ArgumentParser(description="SEM–EDX co-location + ratio gating pipeline")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
