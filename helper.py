"""
helper.py — CUDA Graph acceleration for XFeat & LightGlue
=========================================================

Drop-in CUDA Graph wrappers that capture the native PyTorch forward
pass of XFeat CNN and LightGlue matcher **once** in GPU memory,
then replay with zero kernel-launch overhead.

Key results (measured on RTX 4090 Laptop, Windows WDDM):
  - LightGlue B=1/M=512/N=512:  eager 16ms → graph 1.75ms  (9.1x)
  - XFeat CNN @704×640:         eager  7ms → graph 1.4ms   (5.0x)
  - Output is **bit-identical** to the eager torch path.

Quick start
-----------
.. code-block:: python

    from helper import (
        patch_kornia_capture_safe,
        CGLightGlue, capture_cg_lightglue,
        CGXFeatNet, capture_cg_xfeat, xfeat_cg_extract,
    )

    # 1. LightGlue CUDA Graph (B=1 example)
    patch_kornia_capture_safe()
    model = CGLightGlue().cuda().eval()
    cg = capture_cg_lightglue(model, B=1, M=512, N=512)
    cg["graph"].replay()                    # ~1.75ms vs ~16ms eager

    # 2. XFeat CNN CUDA Graph
    from accelerated_features.modules.xfeat import XFeatModel
    xfeat_net = XFeatModel().cuda().eval()
    cg_net = CGXFeatNet(xfeat_net, H=704, W=640).cuda().eval()
    xcg = capture_cg_xfeat(cg_net, H=704, W=640)
    kpts, desc, scores = xfeat_cg_extract(xcg, rgb_tensor, top_k=512)

Dependencies
------------
- torch >= 2.0 (CUDAGraph requires CUDA 11+)
- kornia >= 0.7 (for LightGlue integration)
- numpy
- accelerated_features  (bundled as ``accelerated_features/`` subdirectory)
"""

from __future__ import annotations

import os
import sys as _sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Locate the bundled accelerated_features subdirectory ──────────────
# This lets users import from accelerated_features.modules.xfeat etc.
# without any pip install or absolute path configuration.
_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
_ACCEL_DIR = os.path.join(_HELPER_DIR, "accelerated_features")
if os.path.isdir(_ACCEL_DIR) and _ACCEL_DIR not in _sys.path:
    _sys.path.insert(0, _ACCEL_DIR)


# ====================================================================
#  kornia capture-safe patches
# ====================================================================

def patch_kornia_capture_safe() -> None:
    """Apply patches to kornia LightGlue for CUDA Graph safety and correct

    mask broadcasting (idempotent).

    The native kornia LightGlue forward uses operations that are illegal
    inside a ``torch.cuda.CUDAGraph`` capture region:
      1. ``torch.nn.functional.new_full`` in the dustbin allocation
         (replaced by ``F.pad``).
      2. ``torch.tensor(0)`` (CPU scalar) in ``filter_matches``
         (replaced by GPU ``torch.zeros``).

    Additionally, ``CrossBlock.forward`` and ``Attention.forward`` (manual
    path) have a **mask broadcasting bug** when the batch size > 1: the
    mask has shape ``(B, M, N)`` but ``sim`` has shape ``(B, 1, M, N)``, so
    ``masked_fill`` broadcasts to ``(B, B, M, N)`` instead of ``(B, 1, M,
    N)``.  The fix is ``mask.unsqueeze(1)`` before ``masked_fill``.
    """
    import kornia.feature.lightglue as _klg

    if getattr(_klg, "_cg_patched", False):
        return

    def _slds(sim, z0, z1):
        b, m, n = sim.shape
        cert = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        s0 = F.log_softmax(sim, 2)
        s1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        sc = F.pad(sim, (0, 1, 0, 1))
        sc[:, :m, :n] = s0 + s1 + cert
        sc[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
        sc[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
        return sc

    _klg.sigmoid_log_double_softmax = _slds
    _klg._cg_patched = True

    # --- Fix mask broadcasting in CrossBlock.forward ---
    _orig_cross_fwd = _klg.CrossBlock.forward

    def _cross_fwd(self, x0, x1, mask=None):
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = (
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
            for t in (qk0, qk1, v0, v1)
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0,
                            mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale ** 0.5, qk1 * self.scale ** 0.5
            sim = _klg.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                # unsqueeze(1) prevents broadcasting (B, M, N) -> (B, B, M, N)
                sim = sim.masked_fill(~mask.unsqueeze(1), -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = _klg.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = _klg.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(_klg.concatenate([x0, m0], -1))
        x1 = x1 + self.ffn(_klg.concatenate([x1, m1], -1))
        return x0, x1

    _klg.CrossBlock.forward = _cross_fwd

    # --- Fix mask broadcasting in Attention.forward (manual path) ---
    _orig_attn_fwd = _klg.Attention.forward

    def _attn_fwd(self, q, k, v, mask=None):
        if self.enable_flash and q.device.type == "cuda":
            if self.has_sdp:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args,
                                                    attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
            else:
                _klg.KORNIA_CHECK(mask is None)
                q, k, v = (x.transpose(-2, -3).contiguous() for x in [q, k, v])
                from kornia.core import stack
                m = self.flash_(q.half(), stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = _klg.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                # unsqueeze(1) to match the head dimension of sim
                sim = sim.masked_fill(~mask.unsqueeze(1), -float("inf"))
            attn = F.softmax(sim, -1)
            return _klg.einsum("...ij,...jd->...id", attn, v)

    _klg.Attention.forward = _attn_fwd


def filter_matches_graph(scores: torch.Tensor, th: float):
    """Capture-safe ``filter_matches`` (GPU scalar zero; identical math).

    The original kornia implementation uses ``torch.tensor(0)`` which
    creates a CPU scalar — illegal inside a CUDA Graph capture.  This
    version uses a GPU zero tensor created via ``torch.zeros``.
    """
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = torch.zeros((), device=max0_exp.device, dtype=max0_exp.dtype)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class CGXFeatNet(torch.nn.Module):
    """XFeat CNN wrapper capturable by CUDA Graph at a fixed resolution.

    The forward pass does:
      1. Bilinear resize input to ``(H, W)``.
      2. Run ``XFeatModel`` → normalized features + keypoint heatmap.
      3. Compute full-resolution keypoint heatmap (``get_kpts_heatmap``
         math, fused into the graph).

    The sparse tail (NMS, top-k, descriptor interpolation) is **not**
    captured — it runs eagerly after replay (see ``xfeat_cg_extract``).

    Parameters
    ----------
    net : XFeatModel
        The underlying XFeat CNN (``XFeatModel`` from accelerated_features).
    H, W : int
        Fixed capture resolution (must be the same as the input after
        preprocessing).
    """

    def __init__(self, net, H: int, W: int):
        super().__init__()
        self.net = net
        self._H, self._W = H, W

    def forward(self, x: torch.Tensor):
        """(1, 3, H, W) → (feats, k1, h1, k1h)"""
        x = F.interpolate(x, (self._H, self._W), mode="bilinear",
                           align_corners=False)
        feats, k1, h1 = self.net(x)
        feats = F.normalize(feats, dim=1)

        # get_kpts_heatmap (softmax_temp=1.0) fused into graph
        s = F.softmax(k1, 1)[:, :64]
        B, _, Hh, Ww = s.shape
        k1h = s.permute(0, 2, 3, 1).reshape(B, Hh, Ww, 8, 8)
        k1h = k1h.permute(0, 1, 3, 2, 4).reshape(B, 1, Hh * 8, Ww * 8)
        return feats, k1, h1, k1h


def capture_cg_xfeat(
    model: CGXFeatNet,
    H: int,
    W: int,
    n_warmup: int = 3,
    device: Optional[torch.device] = None,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Capture a CUDA Graph for the XFeat CNN at resolution ``(H, W)``.

    Parameters
    ----------
    model : CGXFeatNet
        Already on device and in ``eval()`` mode.
    H, W : int
        Capture resolution (must match ``model._H, model._W``).
    n_warmup : int
        Number of warmup passes before capture.
    device : torch.device or None
    stream : torch.cuda.Stream or None

    Returns
    -------
    dict with keys ``"model"``, ``"graph"``, ``"buf"``, ``"outs"``, ``"norm"``:
        - ``graph`` : ``torch.cuda.CUDAGraph``.
        - ``buf`` : static input buffer ``(1, 3, H, W)``.
        - ``outs`` : tuple of output tensors from the last replay.
        - ``norm`` : ``(W-1, H-1)`` for coordinate scaling.
    """
    if device is None:
        device = next(model.parameters()).device
    dev = device

    buf = torch.zeros(1, 3, H, W, device=dev)

    s = stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(n_warmup):
                model(buf)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize(dev)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            outs = model(buf)

    norm = torch.tensor([W - 1, H - 1], device=dev, dtype=torch.float32)
    return {"model": model, "graph": g, "buf": buf, "outs": outs, "norm": norm}


@torch.inference_mode()
def xfeat_cg_extract(
    cg: dict,
    rgb: torch.Tensor,
    top_k: int = 512,
    detection_threshold: float = 0.05,
    original_size: Optional[tuple] = None,
):
    """CUDA-Graph accelerated XFeat extraction for one frame.

    The CNN forward is replayed via CUDA Graph; the sparse tail (NMS,
    top-k, descriptor interpolation, score computation) runs eagerly.

    Parameters
    ----------
    cg : dict
        Returned by ``capture_cg_xfeat``.
    rgb : torch.Tensor (H, W, 3) or (1, 3, H, W)
        Input image.  If ``(H, W, 3)``, conversion to ``(1, 3, H, W)``
        and ``/255`` normalisation is applied automatically.
    top_k : int
        Maximum number of keypoints to keep.
    detection_threshold : float
        NMS score threshold.
    original_size : (int, int) or None
        Original ``(H, W)`` before any resize.  If ``None``, inferred
        from ``rgb.shape``.

    Returns
    -------
    keypoints : (K, 2)  —  (x, y) in original image coordinates
    descriptors : (K, 64)
    scores : (K,)
    """
    dev = cg["buf"].device
    H, W = cg["buf"].shape[2:]

    # Normalise input (accept both torch.Tensor and np.ndarray)
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    if rgb.ndim == 3:
        t = rgb.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        oh, ow = rgb.shape[:2]
    else:
        t = rgb.float()
        if t.shape[1] != 3:
            t = t.permute(0, 3, 1, 2)
        t = t / 255.0
        oh, ow = original_size or t.shape[2:]

    t = F.interpolate(t.to(dev), (H, W), mode="bilinear", align_corners=False)
    cg["buf"].copy_(t)

    # Replay
    cg["graph"].replay()
    feats, _k1, h1, k1h = cg["outs"]

    # ── Fused eager sparse tail (bit-identical to XFeat.detectAndCompute) ──
    norm = cg["norm"]
    local_max = F.max_pool2d(k1h, 5, 1, 2)
    pos = (k1h == local_max) & (k1h > detection_threshold)
    mkpts = pos[0, 0].nonzero()[:, [1, 0]].unsqueeze(0).to(torch.float32)

    if mkpts.shape[1] == 0:
        z = torch.zeros(0, device=dev)
        return z, z.clone(), z.clone()

    grid = 2.0 * (mkpts / norm) - 1.0
    s_nearest = F.grid_sample(
        k1h, grid.unsqueeze(-2), mode="nearest", align_corners=False
    ).squeeze(-1).squeeze(1)
    s_bilin = F.grid_sample(
        h1, grid.unsqueeze(-2), mode="bilinear", align_corners=False
    ).squeeze(-1).squeeze(1)
    scores = s_nearest * s_bilin
    scores[torch.all(mkpts == 0, dim=-1)] = -1

    idxs = torch.argsort(-scores)[:, :top_k]
    sel = idxs[..., None].expand(-1, -1, 2)
    mkpts = torch.gather(mkpts, 1, sel)
    scores = torch.gather(scores, 1, idxs)

    gridk = 2.0 * (mkpts / norm) - 1.0
    feats_s = F.grid_sample(
        feats, gridk.unsqueeze(-2), mode="bicubic", align_corners=False
    ).permute(0, 2, 3, 1).squeeze(-2)
    feats_s = F.normalize(feats_s, dim=-1)

    # Scale keypoints back to original frame size
    mkpts = mkpts * torch.tensor([ow / W, oh / H], device=dev).view(1, 1, -1)
    valid = scores > 0
    return (mkpts[0][valid[0]], feats_s[0][valid[0]], scores[0][valid[0]])


# ====================================================================
#  LightGlue CUDA Graph wrapper (unified, B>=1)
# ====================================================================

class CGLightGlue(torch.nn.Module):
    """Static LightGlue wrapper for CUDA Graph capture/replay.

    Supports arbitrary batch sizes (B>=1). Pre-allocates fixed size
    buffers for B image pairs, each with at most M keypoints on the
    left side and N keypoints on the right side.  Keypoints are sorted
    by score before padding, so only the highest-score points are kept
    and padded with zeros.  Attention masks hide padding.

    This enables a **true batched CUDA Graph** — entire batch matches in a
    single GPU kernel launch with zero kernel launch overhead after capture.

    Parameters
    ----------
    weights_path : str or None
        Path to ``xfeat-lighterglue.pt``.  If ``None``, the default
        path bundled with accelerated_features is used.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        from kornia.feature.lightglue import normalize_keypoints
        self._norm_kpts = normalize_keypoints

        from modules.lighterglue import LighterGlue
        self.lg = LighterGlue(weights=weights_path) if weights_path else LighterGlue()
        self.net = self.lg.net
        self.net.conf.flash = False
        self.net.conf.filter_threshold = 0.1  # default; overridden at call

        # Disable SDPA backend — the einsum fallback is graph-stable
        for t in self.net.transformers:
            sa = getattr(t.self_attn, "inner_attn", None)
            if sa is not None:
                sa.has_sdp = False
                sa.enable_flash = False
            if getattr(t.cross_attn, "flash", None) is not None:
                t.cross_attn.flash = None

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    def forward(self, kpts0, desc0, size0, kpts1, desc1, size1,
                mask0, mask1, th: float = 0.1):
        """Batched LightGlue forward with fixed shape.

        Parameters
        ----------
        kpts0, kpts1 : (B, M, 2) / (B, N, 2)
        desc0, desc1 : (B, M, 64) / (B, N, 64)
        size0, size1 : (B, 2)  —  image (W, H)
        mask0, mask1 : (B, M, 1) / (B, N, 1)  —  bool, True = real keypoint
        th : float  —  match confidence threshold

        Returns
        -------
        m0, m1 : (B, M) / (B, N) int64  —  -1 = no match
        ms0, ms1 : (B, M) / (B, N) float32  —  matching scores
        """
        net = self.net
        kpts0 = self._norm_kpts(kpts0, size0)
        kpts1 = self._norm_kpts(kpts1, size1)
        desc0 = net.input_proj(desc0)
        desc1 = net.input_proj(desc1)
        enc0 = net.posenc(kpts0)
        enc1 = net.posenc(kpts1)
        x0, x1 = desc0, desc1
        for i in range(net.conf.n_layers):
            x0, x1 = net.transformers[i](x0, x1, enc0, enc1,
                                          mask0=mask0, mask1=mask1)
        x0 = x0[..., :desc0.shape[-2], :]
        x1 = x1[..., :desc1.shape[-2], :]
        scores, _ = net.log_assignment[net.conf.n_layers - 1](x0, x1)
        m0o, m1o, ms0, ms1 = filter_matches_graph(scores, th)
        return (m0o.to(torch.int64), m1o.to(torch.int64),
                ms0.to(torch.float32), ms1.to(torch.float32))


def capture_cg_lightglue(
    model: CGLightGlue,
    B: int,
    M: int = 512,
    N: int = 512,
    n_warmup: int = 3,
    device: Optional[torch.device] = None,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Capture a CUDA Graph for batched LightGlue matching at size (B, M, N).

    Parameters
    ----------
    model : CGLightGlue
        Already on the target device and in ``eval()`` mode.
    B, M, N : int
        B = batch size (number of pairs, B=1 works for single pair),
        M = max points per left image in batch,
        N = max points per right image in batch.
    n_warmup : int
        Number of warmup forward passes before capture.
    device : torch.device or None
        Defaults to ``model.device``.
    stream : torch.cuda.Stream or None
        Optional stream for capture isolation.

    Returns
    -------
    dict with keys ``"model"``, ``"graph"``, ``"bufs"``, ``"outs"``:
        - ``"graph"`` : ``torch.cuda.CUDAGraph`` — call ``.replay()``.
        - ``"bufs"`` : dict of static input buffers — fill before replay.
        - ``"outs"`` : tuple of output tensors — read after replay.
        - ``"B"``, ``"M"``, ``"N"`` : the captured dimensions.
    """
    if device is None:
        device = model.device
    dev = device

    bufs = {
        "kpts0": torch.zeros(B, M, 2, device=dev),
        "desc0": torch.zeros(B, M, 64, device=dev),
        "size0": torch.zeros(B, 2, device=dev, dtype=torch.long),
        "kpts1": torch.zeros(B, N, 2, device=dev),
        "desc1": torch.zeros(B, N, 64, device=dev),
        "size1": torch.zeros(B, 2, device=dev, dtype=torch.long),
        "mask0": torch.zeros(B, M, 1, device=dev, dtype=torch.bool),
        "mask1": torch.zeros(B, N, 1, device=dev, dtype=torch.bool),
    }
    args = (bufs["kpts0"], bufs["desc0"], bufs["size0"],
            bufs["kpts1"], bufs["desc1"], bufs["size1"],
            bufs["mask0"], bufs["mask1"], model.net.conf.filter_threshold)

    s = stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(n_warmup):
                model(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize(dev)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            outs = model(*args)

    return {"model": model, "graph": g, "bufs": bufs, "outs": outs,
            "B": B, "M": M, "N": N}


@torch.inference_mode()
def replay_cg_lightglue(
    cg: dict,
    d0_list: list[dict],
    d1_list: list[dict],
    min_conf: float = 0.1,
):
    """Fill buffers and replay captured LightGlue CUDA Graph.

    Supports B >= 1 (B=1 produces a single pair result).

    Parameters
    ----------
    cg : dict
        Returned by ``capture_cg_lightglue``.
    d0_list, d1_list : list[dict]
        Each element is a XFeat output dict whose **values are torch.Tensor**:
        ``{'keypoints': (K0, 2), 'descriptors': (K0, 64), 'scores': (K0,),
          'image_size': (2,) long}``.
        Every field is coerced onto ``cg``'s device inside this function, so
        CPU tensors (or even numpy arrays) are tolerated, but passing device
        tensors avoids an extra host<->device copy.  Length must equal the
        captured ``B = len(d0_list) = len(d1_list)``.
    min_conf : float
        Match confidence threshold. **Ignored at replay time** — the threshold
        is baked into the captured graph via ``model.net.conf.filter_threshold``,
        so it must be set on the model BEFORE ``capture_cg_lightglue``.  Kept
        only for API compatibility with ``XFeat.match_lighterglue``.

    Returns
    -------
    results : list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
        Same output format as ``XFeat.match_lighterglue_batch``:
        For each pair: ``(mkpts0, mkpts1, idxs, mean_score)`` where:
        - ``mkpts0`` : (N_match, 2) np.ndarray of original keypoints
        - ``mkpts1`` : (N_match, 2) np.ndarray of original keypoints
        - ``idxs`` : (N_match, 2) np.ndarray (idx0, idx1)
        - ``mean_score`` : average matching score
    """
    bufs = cg["bufs"]
    Bc, Mc, Nc = cg["B"], cg["M"], cg["N"]
    B = len(d0_list)
    assert B == Bc, f"Batch size mismatch: got {B}, expected {Bc}"
    assert B == len(d1_list), "d0_list and d1_list must have equal length"

    # Zero output buffers
    bufs["kpts0"].zero_(); bufs["desc0"].zero_(); bufs["mask0"].zero_()
    bufs["kpts1"].zero_(); bufs["desc1"].zero_(); bufs["mask1"].zero_()

    dev = bufs["kpts0"].device

    # Fill each batch element by sorting top-M by score.
    # Each field is coerced to a device tensor FIRST ("先变成 tensor 再调用")
    # before it is written into the static CUDA-Graph buffers.
    for b in range(B):
        # --- Left image 0: take top-M by score ---
        d0 = d0_list[b]
        kpts0 = torch.as_tensor(d0['keypoints'], device=dev)
        desc0 = torch.as_tensor(d0['descriptors'], device=dev)
        scores0 = torch.as_tensor(d0['scores'], device=dev)
        n0 = min(int(scores0.shape[0]), Mc)

        if n0 > 0:
            idxs0 = (-scores0).argsort()[:n0]
            bufs["kpts0"][b, :n0] = kpts0[idxs0]
            bufs["desc0"][b, :n0] = desc0[idxs0]
            bufs["mask0"][b, :n0, 0] = True
        bufs["size0"][b] = torch.as_tensor(d0['image_size'], dtype=torch.long, device=dev)

        # --- Right image 1: take top-N by score ---
        d1 = d1_list[b]
        kpts1 = torch.as_tensor(d1['keypoints'], device=dev)
        desc1 = torch.as_tensor(d1['descriptors'], device=dev)
        scores1 = torch.as_tensor(d1['scores'], device=dev)
        n1 = min(int(scores1.shape[0]), Nc)

        if n1 > 0:
            idxs1 = (-scores1).argsort()[:n1]
            bufs["kpts1"][b, :n1] = kpts1[idxs1]
            bufs["desc1"][b, :n1] = desc1[idxs1]
            bufs["mask1"][b, :n1, 0] = True
        bufs["size1"][b] = torch.as_tensor(d1['image_size'], dtype=torch.long, device=dev)

    # Replay the graph
    cg["graph"].replay()
    m0, m1, ms0, ms1 = cg["outs"]

    # Post-process: collect matches back to original coordinate space
    results = []
    for b in range(B):
        # Get matched buffer positions:
        #   matches_b = side0 buffer positions i that have a match
        #   m0[b][i]  = matched side1 buffer position j
        # (m1 is the inverse mapping and must NOT be used here — doing so
        #  would index the wrong side and scramble the output. That was the
        #  original bug: idxs0 was set to m0[matches_b] but then indexed
        #  orig_idx0, while idxs1 was set to m1[matches_b], which is garbage.)
        matches_b = (m0[b] != -1).nonzero().squeeze(1)
        idxs0_buf = matches_b                       # side0 buffer positions
        idxs1_buf = m0[b][matches_b]                # side1 buffer positions
        mean_score = 0.0
        if len(matches_b) > 0:
            mean_score = float(ms0[b][matches_b].mean().item())

        # Map buffer positions back to ORIGINAL (pre-top-M/N) keypoints.
        # The fill step sorted each side by score and placed original index
        # orig_idx_s[p] at buffer position p, so orig_idx_s[p] recovers it.
        d0_orig = d0_list[b]
        d1_orig = d1_list[b]
        if len(matches_b) > 0:
            kpts0_full = torch.as_tensor(d0_orig['keypoints'], device=dev)
            kpts1_full = torch.as_tensor(d1_orig['keypoints'], device=dev)
            scores0_full = torch.as_tensor(d0_orig['scores'], device=dev)
            scores1_full = torch.as_tensor(d1_orig['scores'], device=dev)

            orig_idx0 = (-scores0_full).argsort()[:Mc]
            orig_idx1 = (-scores1_full).argsort()[:Nc]
            idxs0_np = idxs0_buf.cpu().numpy().astype(np.int64)
            idxs1_np = idxs1_buf.cpu().numpy().astype(np.int64)
            pts0 = kpts0_full[orig_idx0[idxs0_np]].cpu().numpy()
            pts1 = kpts1_full[orig_idx1[idxs1_np]].cpu().numpy()
            idxs_out = np.stack([orig_idx0[idxs0_np].cpu().numpy(),
                                 orig_idx1[idxs1_np].cpu().numpy()], axis=1)
            results.append((pts0, pts1, idxs_out, mean_score))
        else:
            results.append((np.empty((0, 2), dtype=np.float32),
                            np.empty((0, 2), dtype=np.float32),
                            np.empty((0, 2), dtype=np.int64),
                            mean_score))

    return results