"""Display rendering for the MVP (P5): stretched RGB / FCC PNGs and fixed-scale overlays.

Display stretch (docs/mvp/api_contract.md): per-band 2-98 percentile computed on
the LR input and applied identically to LR, bicubic, SR and HR. Band order is
cfg.dataset.bands (B04, B03, B02, B08): RGB = 0,1,2; FCC (NIR, R, G) = 3,0,1.

Overlays use a FIXED scale 0..vmax (never per-image min-max), so one colour
means one value on every image. Colour maps are built-in 256-entry LUTs
(matplotlib's magma / viridis, sampled once and embedded) so matplotlib is not
needed at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

try:
    from drishtisr.eval.baselines import nearest_upsample
except ImportError:
    from src.eval.baselines import nearest_upsample  # type: ignore

BANDS = {"rgb": (0, 1, 2), "fcc": (3, 0, 1)}
ALPHA_MAX = 0.85
ALPHA_FLOOR_FRAC = 0.05

_LUT_HEX = {
    "magma": "00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf",
    "viridis": "44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725",
}
LUTS = {name: np.frombuffer(bytes.fromhex(h), dtype=np.uint8).reshape(256, 3)
        for name, h in _LUT_HEX.items()}


def stretch_params(lr: np.ndarray) -> np.ndarray:
    """Per-band ``(p2, p98)`` of the LR ``(C, h, w)``; returns ``(C, 2)`` float32."""
    lr = np.asarray(lr, dtype=np.float32)
    return np.stack([np.percentile(lr, 2, axis=(1, 2)),
                     np.percentile(lr, 98, axis=(1, 2))], axis=1).astype(np.float32)


def to_png_array(img: np.ndarray, params: np.ndarray, mode: str = "rgb") -> np.ndarray:
    """``(C, H, W)`` reflectance -> uint8 ``(H, W, 3)`` with the LR stretch ``params``."""
    if mode not in BANDS:
        raise ValueError(f"mode must be one of {sorted(BANDS)}; got {mode!r}.")
    idx = list(BANDS[mode])
    x = np.asarray(img, dtype=np.float32)[idx]
    lo = params[idx, 0][:, None, None]
    span = np.maximum(params[idx, 1] - params[idx, 0], 1e-6)[:, None, None]
    y = np.clip((x - lo) / span, 0.0, 1.0)
    return np.ascontiguousarray((y * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0))


def lr_display(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    """Nearest-neighbour upsample of the LR to the SR grid (project nearest baseline)."""
    return np.asarray(nearest_upsample(np.ascontiguousarray(lr, dtype=np.float32), scale),
                      dtype=np.float32)


def upsample_map(map2d: np.ndarray, scale: int = 4) -> np.ndarray:
    """Nearest-upsample a 2-D map (used for the 10 m consistency map before rendering)."""
    return np.repeat(np.repeat(np.asarray(map2d), scale, axis=0), scale, axis=1)


def overlay_rgba(map2d: np.ndarray, vmax: float, cmap: str = "magma") -> np.ndarray:
    """Fixed-scale RGBA overlay, uint8 ``(H, W, 4)``.

    Colour = LUT at ``clip(map / vmax, 0, 1)``. Alpha is 0 below 5% of vmax and
    rises linearly to 0.85 at vmax (held above). Non-finite pixels are
    transparent.
    """
    if cmap not in LUTS:
        raise ValueError(f"cmap must be one of {sorted(LUTS)}; got {cmap!r}.")
    if not vmax > 0:
        raise ValueError(f"vmax must be positive; got {vmax!r}.")
    m = np.asarray(map2d, dtype=np.float64)
    finite = np.isfinite(m)
    t = np.clip(np.where(finite, m, 0.0) / float(vmax), 0.0, 1.0)
    rgb = LUTS[cmap][np.rint(t * 255).astype(np.int64)]
    alpha = np.clip((t - ALPHA_FLOOR_FRAC) / (1.0 - ALPHA_FLOOR_FRAC), 0.0, 1.0) * ALPHA_MAX
    alpha = np.where(finite, alpha, 0.0)
    a8 = np.rint(alpha * 255).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, a8[..., None]], axis=-1))


def legend_stops(vmax: float, n: int = 5) -> List[float]:
    """``n`` evenly spaced tick values from 0 to vmax."""
    return [float(v) for v in np.linspace(0.0, float(vmax), int(n))]


def save_png(array: np.ndarray, path: Union[str, Path]) -> Path:
    """Write a uint8 ``(H, W, 3|4)`` or ``(H, W)`` array as PNG; parents created."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8)).save(p, format="PNG")
    return p
