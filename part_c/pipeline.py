from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------

@dataclass
class TryOnConfig:
    category: str = "top"
    scale_adjust: float = 1.0
    y_offset: int = 0
    blend_alpha: float = 0.97
    hmr_landmarks: Optional[Dict[str, Any]] = None
    shoulder_padding: float = 1.30
    neck_offset_frac: float = 0.12
    sleeve_extend_px: int = 18
    suppress_torso: bool = True
    suppress_side_pad: float = 0.28
    suppress_top_pad: float = 0.08
    suppress_bot_pad: float = 0.10
    feather_radius: int = 31


@dataclass
class TryOnResult:
    image_bgr: np.ndarray
    confidence: float
    mode: str
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pose estimation  (model_complexity=2, segmentation enabled)
# ---------------------------------------------------------------------------

def _load_pose_model():
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,           # best accuracy
            enable_segmentation=True,     # free person mask
            min_detection_confidence=0.5,
        )
        return pose, mp
    except Exception:
        return None, None


def _extract_pose_points(
    user_bgr: np.ndarray,
) -> Tuple[Optional[Dict[str, Tuple[int, int]]], float, Optional[np.ndarray]]:
    pose_model, _ = _load_pose_model()
    if pose_model is None:
        return None, 0.0, None

    h, w = user_bgr.shape[:2]
    rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)
    if not result.pose_landmarks:
        return None, 0.0, None

    lm = result.pose_landmarks.landmark

    def pt(idx: int) -> Tuple[int, int, float]:
        p = lm[idx]
        return int(p.x * w), int(p.y * h), float(p.visibility)

    ls  = pt(11);  rs  = pt(12)
    lh  = pt(23);  rh  = pt(24)
    le  = pt(13);  re  = pt(14)
    ln  = pt(9);   rn  = pt(10)   # mouth corners as neck proxy

    conf = float(np.mean([ls[2], rs[2], lh[2], rh[2]]))
    if conf < 0.30:
        return None, conf, None

    neck_x = (ln[0] + rn[0]) // 2
    neck_y = (ln[1] + rn[1]) // 2

    anchors = {
        "left_shoulder":  (ls[0], ls[1]),
        "right_shoulder": (rs[0], rs[1]),
        "left_hip":       (lh[0], lh[1]),
        "right_hip":      (rh[0], rh[1]),
        "neck":           (neck_x, neck_y),
        "left_elbow":     (le[0], le[1]),
        "right_elbow":    (re[0], re[1]),
    }
    return anchors, conf, result.segmentation_mask


# ---------------------------------------------------------------------------
# HMR landmark parser
# ---------------------------------------------------------------------------

def _extract_hmr_points(
    hmr_landmarks: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
) -> Tuple[Optional[Dict[str, Tuple[int, int]]], float]:
    if not hmr_landmarks:
        return None, 0.0

    h, w = image_shape[:2]
    keys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

    if all(k in hmr_landmarks for k in keys):
        anchors: Dict[str, Tuple[int, int]] = {}
        for k in keys:
            pt2 = hmr_landmarks[k]
            if not isinstance(pt2, (list, tuple)) or len(pt2) < 2:
                return None, 0.0
            x, y = float(pt2[0]), float(pt2[1])
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x, y = x * w, y * h
            anchors[k] = (int(x), int(y))
        lsx, lsy = anchors["left_shoulder"]
        rsx, rsy = anchors["right_shoulder"]
        smid_y = (lsy + rsy) // 2
        anchors["neck"] = ((lsx + rsx) // 2, smid_y - int(0.12 * h))
        sw = abs(rsx - lsx)
        anchors["left_elbow"]  = (lsx - int(0.55 * sw), lsy + int(0.25 * sw))
        anchors["right_elbow"] = (rsx + int(0.55 * sw), rsy + int(0.25 * sw))
        score = float(hmr_landmarks.get("score", 0.95))
        return anchors, float(np.clip(score, 0.0, 1.0))

    joints_2d = hmr_landmarks.get("joints_2d") if isinstance(hmr_landmarks, dict) else None
    if isinstance(joints_2d, list) and len(joints_2d) >= 13:
        idx_map = {"left_shoulder": 5, "right_shoulder": 6,
                   "left_hip": 11, "right_hip": 12}
        anchors = {}
        for name, idx in idx_map.items():
            pt2 = joints_2d[idx]
            if not isinstance(pt2, (list, tuple)) or len(pt2) < 2:
                return None, 0.0
            x, y = float(pt2[0]), float(pt2[1])
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x, y = x * w, y * h
            anchors[name] = (int(x), int(y))
        lsx, lsy = anchors["left_shoulder"]
        rsx, rsy = anchors["right_shoulder"]
        smid_y = (lsy + rsy) // 2
        sw = abs(rsx - lsx)
        anchors["neck"] = ((lsx + rsx) // 2, smid_y - int(0.12 * h))
        anchors["left_elbow"]  = (lsx - int(0.55 * sw), lsy + int(0.25 * sw))
        anchors["right_elbow"] = (rsx + int(0.55 * sw), rsy + int(0.25 * sw))
        score = float(hmr_landmarks.get("score", 0.9))
        return anchors, float(np.clip(score, 0.0, 1.0))

    return None, 0.0


# ---------------------------------------------------------------------------
# Garment masking — handles DARK and LIGHT studio backgrounds + GrabCut
# ---------------------------------------------------------------------------

def _garment_mask(garment_img: np.ndarray) -> np.ndarray:
    """Return uint8 H x W mask (0 / 255) of the garment foreground."""
    if garment_img.ndim == 3 and garment_img.shape[2] == 4:
        alpha = garment_img[:, :, 3]
        mask = (alpha > 20).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    bgr = garment_img[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    dark_bg  = cv2.inRange(hsv, np.array([0,   0,   0],   dtype=np.uint8),
                                np.array([180, 255, 50],  dtype=np.uint8))
    fg_dark  = cv2.bitwise_not(dark_bg)

    light_bg = cv2.inRange(hsv, np.array([0,   0,  185],  dtype=np.uint8),
                                np.array([180, 40, 255],  dtype=np.uint8))
    fg_light = cv2.bitwise_not(light_bg)

    def _fg_frac(m: np.ndarray) -> float:
        return float(np.sum(m > 0)) / m.size

    fg = fg_dark if abs(_fg_frac(fg_dark) - 0.5) < abs(_fg_frac(fg_light) - 0.5) else fg_light
    fg = _grabcut_refine(bgr, fg)

    kernel = np.ones((7, 7), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
    fg = cv2.medianBlur(fg, 5)
    return fg


def _grabcut_refine(bgr: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
    try:
        gc = np.where(init_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        return init_mask


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Torso suppression
# ---------------------------------------------------------------------------

def _suppress_existing_torso(
    user_bgr: np.ndarray,
    anchors: Dict[str, Tuple[int, int]],
    seg_mask: Optional[np.ndarray],
    config: TryOnConfig,
) -> np.ndarray:
    if not config.suppress_torso:
        return user_bgr

    h, w = user_bgr.shape[:2]
    ls   = np.array(anchors["left_shoulder"],  dtype=np.float32)
    rs   = np.array(anchors["right_shoulder"], dtype=np.float32)
    lhip = np.array(anchors["left_hip"],       dtype=np.float32)
    rhip = np.array(anchors["right_hip"],      dtype=np.float32)

    sw     = max(1.0, float(np.linalg.norm(rs - ls)))
    pad_x  = sw * config.suppress_side_pad
    pad_yt = sw * config.suppress_top_pad
    pad_yb = sw * config.suppress_bot_pad

    le = np.array(anchors.get("left_elbow",  (int(ls[0] - sw * 0.5), int(ls[1] + sw * 0.2))), dtype=np.float32)
    re = np.array(anchors.get("right_elbow", (int(rs[0] + sw * 0.5), int(rs[1] + sw * 0.2))), dtype=np.float32)

    pts = np.array([
        [le[0] - pad_x * 0.3, le[1]],
        [ls[0] - pad_x,       ls[1] - pad_yt],
        [rs[0] + pad_x,       rs[1] - pad_yt],
        [re[0] + pad_x * 0.3, re[1]],
        [rhip[0] + pad_x,     rhip[1] + pad_yb],
        [lhip[0] - pad_x,     lhip[1] + pad_yb],
    ], dtype=np.int32)

    torso = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(torso, pts, 255)

    if seg_mask is not None:
        person = (seg_mask > 0.4).astype(np.uint8) * 255
        torso  = cv2.bitwise_and(torso, person)

    torso_f = cv2.GaussianBlur(torso.astype(np.float32), (71, 71), 0) / 255.0
    suppress = np.array([40, 40, 40], dtype=np.float32)
    out = user_bgr.astype(np.float32)
    for c in range(3):
        out[:, :, c] = suppress[c] * torso_f + out[:, :, c] * (1.0 - torso_f)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Perspective warp  (4-point)
# ---------------------------------------------------------------------------

def _perspective_warp_garment(
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_mask: np.ndarray,
    anchors: Dict[str, Tuple[int, int]],
    config: TryOnConfig,
) -> Tuple[np.ndarray, np.ndarray, str]:
    h_u, w_u = user_bgr.shape[:2]
    bbox = _bbox_from_mask(garment_mask)
    if bbox is None:
        return np.zeros_like(user_bgr), np.zeros((h_u, w_u), dtype=np.uint8), "fallback_preview"

    x1, y1, x2, y2 = bbox
    gw = max(1, x2 - x1); gh = max(1, y2 - y1)

    src = np.float32([
        [x1 + 0.06 * gw, y1 + 0.03 * gh],
        [x1 + 0.94 * gw, y1 + 0.03 * gh],
        [x1 + 0.94 * gw, y1 + 0.98 * gh],
        [x1 + 0.06 * gw, y1 + 0.98 * gh],
    ])

    ls   = np.array(anchors["left_shoulder"],  dtype=np.float32)
    rs   = np.array(anchors["right_shoulder"], dtype=np.float32)
    lhip = np.array(anchors["left_hip"],       dtype=np.float32)
    rhip = np.array(anchors["right_hip"],      dtype=np.float32)
    neck = np.array(anchors["neck"],           dtype=np.float32)

    sv  = rs - ls
    sn  = max(1.0, float(np.linalg.norm(sv)))
    dir = sv / sn
    sw  = sn * config.scale_adjust * config.shoulder_padding
    sm  = (ls + rs) / 2.0
    lift = (neck - sm) * config.neck_offset_frac
    voff = np.array([0.0, float(config.y_offset)], dtype=np.float32)

    ext = config.sleeve_extend_px
    dst_ls   = sm - dir * (sw / 2.0) + lift + voff + np.array([-ext, 0.0])
    dst_rs   = sm + dir * (sw / 2.0) + lift + voff + np.array([ ext, 0.0])
    tl       = max(40.0, float(np.linalg.norm(((lhip + rhip) / 2.0) - sm)))
    dst_lhip = lhip + np.array([-sw * 0.06, tl * 0.05]) + voff
    dst_rhip = rhip + np.array([ sw * 0.06, tl * 0.05]) + voff

    dst = np.float32([dst_ls, dst_rs, dst_rhip, dst_lhip])
    M   = cv2.getPerspectiveTransform(src, dst)

    warped_rgb  = cv2.warpPerspective(garment_bgr[:, :, :3], M, (w_u, h_u),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpPerspective(garment_mask, M, (w_u, h_u),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_rgb, warped_mask, "perspective_overlay"


# ---------------------------------------------------------------------------
# Dense mesh warp (Delaunay, used at high confidence)
# ---------------------------------------------------------------------------

def _dense_mesh_warp(
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_mask: np.ndarray,
    anchors: Dict[str, Tuple[int, int]],
    config: TryOnConfig,
) -> Tuple[np.ndarray, np.ndarray, str]:
    h_u, w_u = user_bgr.shape[:2]
    bbox = _bbox_from_mask(garment_mask)
    if bbox is None:
        return np.zeros_like(user_bgr), np.zeros((h_u, w_u), dtype=np.uint8), "fallback_preview"

    x1, y1, x2, y2 = bbox
    gw = max(1, x2 - x1); gh = max(1, y2 - y1)

    ls   = np.array(anchors["left_shoulder"],  dtype=np.float32)
    rs   = np.array(anchors["right_shoulder"], dtype=np.float32)
    lhip = np.array(anchors["left_hip"],       dtype=np.float32)
    rhip = np.array(anchors["right_hip"],      dtype=np.float32)
    neck = np.array(anchors["neck"],           dtype=np.float32)
    voff = np.array([0.0, float(config.y_offset)], dtype=np.float32)

    sw  = max(1.0, float(np.linalg.norm(rs - ls)))
    pad = sw * config.scale_adjust * config.shoulder_padding
    dir = (rs - ls) / sw
    sm  = (ls + rs) / 2.0
    lift = (neck - sm) * config.neck_offset_frac
    tl   = max(40.0, float(np.linalg.norm(((lhip + rhip) / 2.0) - sm)))

    ext = float(config.sleeve_extend_px)
    dst_ls   = sm - dir * (pad / 2.0) + lift + voff + np.array([-ext, 0.0])
    dst_rs   = sm + dir * (pad / 2.0) + lift + voff + np.array([ ext, 0.0])
    dst_lhip = lhip + np.array([-sw * 0.06, tl * 0.05]) + voff
    dst_rhip = rhip + np.array([ sw * 0.06, tl * 0.05]) + voff

    waist_l  = dst_ls * 0.35 + dst_lhip * 0.65
    waist_r  = dst_rs * 0.35 + dst_rhip * 0.65
    chest_l  = dst_ls * 0.65 + dst_lhip * 0.35
    chest_r  = dst_rs * 0.65 + dst_rhip * 0.35

    dst_pts = np.array([
        dst_ls, (dst_ls + dst_rs) / 2.0, dst_rs,
        chest_l, (chest_l + chest_r) / 2.0, chest_r,
        waist_l, (waist_l + waist_r) / 2.0, waist_r,
        dst_lhip, (dst_lhip + dst_rhip) / 2.0, dst_rhip,
    ], dtype=np.float32)

    bx1 = float(dst_pts[:, 0].min()); bx2 = float(dst_pts[:, 0].max())
    by1 = float(dst_pts[:, 1].min()); by2 = float(dst_pts[:, 1].max())
    bw_ = max(1.0, bx2 - bx1);       bh_ = max(1.0, by2 - by1)

    src_pts = np.zeros_like(dst_pts)
    src_pts[:, 0] = x1 + ((dst_pts[:, 0] - bx1) / bw_) * gw
    src_pts[:, 1] = y1 + ((dst_pts[:, 1] - by1) / bh_) * gh

    m = 8
    corners_dst = np.array([[bx1-m, by1-m],[bx2+m, by1-m],[bx2+m, by2+m],[bx1-m, by2+m]], dtype=np.float32)
    corners_src = np.array([[x1-m,  y1-m ],[x2+m,  y1-m ],[x2+m,  y2+m ],[x1-m,  y2+m ]], dtype=np.float32)

    all_dst = np.vstack([dst_pts, corners_dst])
    all_src = np.vstack([src_pts, corners_src])

    try:
        tri = Delaunay(all_dst)
    except Exception:
        return _perspective_warp_garment(user_bgr, garment_bgr, garment_mask, anchors, config)

    output      = np.zeros_like(user_bgr)
    output_mask = np.zeros((h_u, w_u), dtype=np.uint8)

    for simplex in tri.simplices:
        dst_tri = all_dst[simplex].astype(np.float32)
        src_tri = all_src[simplex].astype(np.float32)

        sr = cv2.boundingRect(src_tri); dr = cv2.boundingRect(dst_tri)
        sx, sy, sw2, sh2 = sr; dx, dy, dw2, dh2 = dr

        sx = max(0, sx); sy = max(0, sy)
        sw2 = min(sw2, garment_bgr.shape[1] - sx)
        sh2 = min(sh2, garment_bgr.shape[0] - sy)
        dx = max(0, dx); dy = max(0, dy)
        dw2 = min(dw2, w_u - dx); dh2 = min(dh2, h_u - dy)

        if sw2 <= 0 or sh2 <= 0 or dw2 <= 0 or dh2 <= 0:
            continue

        src_crop  = garment_bgr[sy:sy+sh2, sx:sx+sw2, :3]
        mask_crop = garment_mask[sy:sy+sh2, sx:sx+sw2]

        s_local = src_tri - np.array([sx, sy], dtype=np.float32)
        d_local = dst_tri - np.array([dx, dy], dtype=np.float32)

        M  = cv2.getAffineTransform(s_local, d_local)
        wp = cv2.warpAffine(src_crop,  M, (dw2, dh2), flags=cv2.INTER_LINEAR)
        wm = cv2.warpAffine(mask_crop, M, (dw2, dh2), flags=cv2.INTER_LINEAR)

        tri_m = np.zeros((dh2, dw2), dtype=np.uint8)
        cv2.fillConvexPoly(tri_m, np.int32(d_local), 255)
        tri_m = cv2.bitwise_and(tri_m, wm)

        roi_img = output[dy:dy+dh2, dx:dx+dw2]
        roi_msk = output_mask[dy:dy+dh2, dx:dx+dw2]
        sel = tri_m > 0
        roi_img[sel] = wp[sel]
        roi_msk[sel] = 255

    return output, output_mask, "dense_mesh_warp"


# ---------------------------------------------------------------------------
# Mask feathering
# ---------------------------------------------------------------------------

def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    r = radius | 1
    blurred  = cv2.GaussianBlur(mask.astype(np.float32), (r, r), 0)
    er_k = max(1, r // 4)
    interior = cv2.erode(mask, np.ones((er_k, er_k), np.uint8))
    return np.clip(np.maximum(blurred, interior.astype(np.float32)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def _blend(
    user_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    overlay_mask: np.ndarray,
    alpha_scale: float,
    feather_radius: int = 31,
) -> np.ndarray:
    smooth = _feather_mask(overlay_mask, feather_radius)
    alpha  = np.clip(smooth.astype(np.float32) / 255.0 * alpha_scale, 0.0, 1.0)

    try:
        bin_mask = (overlay_mask > 127).astype(np.uint8) * 255
        ys, xs = np.where(bin_mask > 0)
        if len(xs) > 100:
            cx = int(xs.mean())
            cy = int(ys.mean())
            h, w = user_bgr.shape[:2]
            if 0 < cx < w and 0 < cy < h:
                cloned = cv2.seamlessClone(overlay_bgr, user_bgr, bin_mask, (cx, cy), cv2.NORMAL_CLONE)
                # Guard: reject Poisson output if it collapses garment brightness.
                sel = bin_mask > 0
                ov_mean = float(np.mean(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2HSV)[:, :, 2][sel]))
                cl_mean = float(np.mean(cv2.cvtColor(cloned, cv2.COLOR_BGR2HSV)[:, :, 2][sel]))
                if cl_mean >= 0.55 * max(1.0, ov_mean):
                    out = user_bgr.astype(np.float32)
                    cl = cloned.astype(np.float32)
                    for c in range(3):
                        out[:, :, c] = cl[:, :, c] * alpha + out[:, :, c] * (1.0 - alpha)
                    return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        pass

    out = user_bgr.copy().astype(np.float32)
    ov = overlay_bgr.astype(np.float32)
    for c in range(3):
        out[:, :, c] = ov[:, :, c] * alpha + out[:, :, c] * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_center_overlay(
    user_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    garment_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h_u, w_u = user_bgr.shape[:2]
    h_g, w_g = garment_mask.shape[:2]
    target_w = int(w_u * 0.50)
    scale    = target_w / max(1, w_g)
    target_h = max(1, int(h_g * scale))

    g_res = cv2.resize(garment_bgr[:, :, :3], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    m_res = cv2.resize(garment_mask,           (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    x  = max(0, (w_u - target_w) // 2)
    y  = max(0, int(h_u * 0.18))
    y2 = min(h_u, y + target_h); x2 = min(w_u, x + target_w)

    overlay = np.zeros_like(user_bgr)
    mask    = np.zeros((h_u, w_u), dtype=np.uint8)
    overlay[y:y2, x:x2] = g_res[: y2 - y, : x2 - x]
    mask[y:y2, x:x2]    = m_res[: y2 - y, : x2 - x]
    return overlay, mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def virtual_try_on(
    user_bgr: np.ndarray,
    garment_img: np.ndarray,
    config: Optional[TryOnConfig] = None,
) -> TryOnResult:
    """
    Pose-guided virtual try-on with significantly improved accuracy.

    Key improvements over original:
    - model_complexity=2 + enable_segmentation (best MediaPipe quality)
    - GrabCut garment masking: handles dark AND light studio backgrounds
    - Torso suppression: darkens existing clothing before overlay
    - conf >= 0.70: dense Delaunay mesh warp (12 body control pts + corners)
    - conf <  0.70: 4-point perspective warp (replaces 3-pt affine)
    - Feathered mask edges (erode + Gaussian) for seamless blending
    - Shoulder padding + neck-offset for natural garment placement
    - HMR landmarks preserved + neck anchor synthesised automatically
    """
    if config is None:
        config = TryOnConfig()

    warnings: List[str] = []

    garment_mask = _garment_mask(garment_img)
    if _bbox_from_mask(garment_mask) is None:
        return TryOnResult(user_bgr.copy(), 0.0, "failed",
                           ["Could not isolate garment foreground"])

    anchors:  Optional[Dict[str, Tuple[int, int]]] = None
    pose_conf = 0.0
    seg_mask: Optional[np.ndarray] = None

    if config.hmr_landmarks is not None:
        anchors, pose_conf = _extract_hmr_points(config.hmr_landmarks, user_bgr.shape)
        if anchors is None:
            warnings.append("HMR landmarks invalid; fell back to MediaPipe")

    if anchors is None:
        anchors, pose_conf, seg_mask = _extract_pose_points(user_bgr)

    if anchors is None:
        overlay, mask = _fallback_center_overlay(user_bgr, garment_img, garment_mask)
        mode = "fallback_preview"
        warnings.append("Pose landmarks not reliable; used centred preview mode")
        base = user_bgr
    else:
        base = _suppress_existing_torso(user_bgr, anchors, seg_mask, config)

        if pose_conf >= 0.70:
            overlay, mask, mode = _dense_mesh_warp(base, garment_img, garment_mask, anchors, config)
            if mode == "fallback_preview":
                overlay, mask, mode = _perspective_warp_garment(base, garment_img, garment_mask, anchors, config)
        else:
            overlay, mask, mode = _perspective_warp_garment(base, garment_img, garment_mask, anchors, config)

    out = _blend(base, overlay, mask, config.blend_alpha, config.feather_radius)
    confidence = (0.40 + 0.60 * pose_conf) if mode in ("dense_mesh_warp", "perspective_overlay") else 0.35

    return TryOnResult(out, float(np.clip(confidence, 0.0, 1.0)), mode, warnings)
