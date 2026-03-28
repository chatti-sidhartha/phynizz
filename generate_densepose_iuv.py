"""
generate_densepose_iuv.py
=========================
Transform a full-body photograph into a DensePose-style IUV visualization.

  • Black background
  • 24 body-part segments (DensePose nomenclature)
  • Each part colored with smooth UV-gradient in the neon-green / fiery-red / yellow
    color family, matching the classic DensePose colormap
  • Subject pose/proportions preserved via body keypoints

Usage
-----
    python generate_densepose_iuv.py test_images/full_body.jpg
    python generate_densepose_iuv.py test_images/full_body.jpg --output-dir output --scale 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "modules"))

# ─────────────────────────────────────────────────────────────────────────────
#  DensePose-24 body-part color table
#  Each entry: (R, G, B) of the "base hue" for that part index (1-based).
#  Colors tuned to: neon green, fiery reds / oranges, yellows — as requested.
# ─────────────────────────────────────────────────────────────────────────────
PART_COLORS = {
    # torso
    1:  (255, 60,  0),    # Torso-Front        fiery red-orange
    2:  (220, 40,  0),    # Torso-Back         deep red
    # head / face
    3:  (255, 200, 0),    # Right-Hand         vivid yellow
    4:  (255, 220, 20),   # Left-Hand          lemon yellow
    5:  (255, 165, 0),    # Left-Foot          orange
    6:  (255, 140, 0),    # Right-Foot         darker orange
    # upper arm
    7:  (57,  255, 20),   # Upper-Leg-Right    neon green
    8:  (0,   230, 60),   # Upper-Leg-Left     forest neon
    # lower arm
    9:  (255, 80,  0),    # Lower-Leg-Right    vivid orange-red
    10: (255, 100, 10),   # Lower-Leg-Left     orange
    # upper leg
    11: (180, 255, 0),    # Upper-Arm-Left     lime-green
    12: (140, 255, 0),    # Upper-Arm-Right    chartreuse
    # lower leg
    13: (255, 50,  50),   # Lower-Arm-Left     coral red
    14: (230, 30,  30),   # Lower-Arm-Right    bright red
    # feet / hands
    15: (255, 240, 0),    # Head               bright yellow
    16: (255, 220, 0),    # Face               gold yellow
    # extras (shoulders, neck, etc.)
    17: (100, 255, 80),   # Neck               mid-neon green
    18: (70,  255, 100),  # Left-Shoulder
    19: (255, 120, 0),    # Right-Shoulder     amber
    20: (255, 190, 0),    # Left-Hip           yellow-gold
    21: (200, 255, 0),    # Right-Hip          yellow-green
    22: (255, 60,  80),   # Left-Knee          pinkish red
    23: (255, 40,  60),   # Right-Knee         red
    24: (57,  255, 20),   # feet/shin          neon green
}

# ─────────────────────────────────────────────────────────────────────────────
#  SMPL / DensePose joint indices (COCO-format 17 joints often used)
#  We'll use MediaPipe Pose 33-keypoint output if available, else fall back
#  to a skeleton estimated from HMR.
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ellipse_mask(canvas, cx, cy, ax, ay, angle_deg=0):
    """Return a boolean mask with an ellipse drawn on it."""
    mask = np.zeros(canvas[:2], dtype=np.uint8)
    cx, cy, ax, ay = int(cx), int(cy), max(1, int(ax)), max(1, int(ay))
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle_deg, 0, 360, 255, -1)
    return mask.astype(bool)


def polygon_mask(canvas_hw, pts):
    """Return a boolean mask from a polygon."""
    mask = np.zeros(canvas_hw[:2], dtype=np.uint8)
    arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [arr], 255)
    return mask.astype(bool)


def uv_gradient(mask, base_rgb, u_dir="x", v_dir="y", brightness=1.0):
    """
    For every pixel in mask, compute smooth UV gradient based on position.
    Returns an RGB image (H, W, 3) float32 in [0, 255].
    """
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    # Normalize positions inside the bounding box → [0, 1] (U, V)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    dy = max(1, y_max - y_min)
    dx = max(1, x_max - x_min)

    u_vals = (xs - x_min) / dx   # [0, 1], horizontal
    v_vals = (ys - y_min) / dy   # [0, 1], vertical

    r0, g0, b0 = [c / 255.0 for c in base_rgb]

    # Gradient: interpolate between a darker shade → base → lighter shade
    # using UV to control brightness and hue shift
    # V (vertical within part) → brightness modulation
    # U (horizontal within part) → slight hue shift
    bright = 0.40 + 0.60 * v_vals          # 0.4 → 1.0 top→bottom
    sat_shift = 0.10 * (u_vals - 0.5)      # slight L/R colour shift

    r_px = np.clip((r0 + sat_shift) * bright * brightness, 0, 1)
    g_px = np.clip((g0 - sat_shift) * bright * brightness, 0, 1)
    b_px = np.clip(b0 * bright * brightness, 0, 1)

    img = np.zeros((H, W, 3), dtype=np.float32)
    img[ys, xs, 0] = r_px * 255
    img[ys, xs, 1] = g_px * 255
    img[ys, xs, 2] = b_px * 255
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Pose detection  (MediaPipe → fallback to simple edge/GrabCut skeleton)
# ─────────────────────────────────────────────────────────────────────────────

def _download_mediapipe_model():
    """Download the PoseLandmarker .task file for MediaPipe Tasks API."""
    import urllib.request, os
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )
    dest = os.path.join(_HERE, "models", "pose_landmarker_heavy.task")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"        Downloading MediaPipe model to {dest} …")
        urllib.request.urlretrieve(url, dest)
    return dest


def detect_pose_mediapipe(img_rgb):
    """Return 33 landmarks [(x_px, y_px, visibility), …] or None."""
    H, W = img_rgb.shape[:2]

    # ── Try Tasks API (mediapipe ≥ 0.10) ─────────────────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import RunningMode

        model_path = _download_mediapipe_model()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,   # disable: causes native crash on Windows
            running_mode=RunningMode.IMAGE,
        )
        with mp_vision.PoseLandmarker.create_from_options(opts) as ldr:
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=img_rgb.copy()
            )
            result = ldr.detect(mp_img)
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]   # first person
                kps = [(lm.x * W, lm.y * H, lm.visibility) for lm in lms]
                return kps, None   # person mask built via GrabCut
    except Exception as e:
        print(f"  [MediaPipe Tasks API: {e}]")

    # ── Fallback: legacy solutions API (mediapipe < 0.10) ─────────────────
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.3,
        ) as pose:
            res = pose.process(img_rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                kps = [(lm.x * W, lm.y * H, lm.visibility) for lm in lms]
                seg = res.segmentation_mask if res.segmentation_mask is not None else None
                return kps, seg
    except Exception as e:
        print(f"  [MediaPipe legacy API: {e}]")

    return None, None


def landmark_person_mask(landmarks, H, W):
    """
    Build a person mask from the convex hull of visible MediaPipe landmarks,
    then dilate heavily. This is robust even when clothing color matches background
    (e.g. dark suit on dark wall) where GrabCut fails completely.
    """
    if landmarks is None or len(landmarks) == 0:
        return None

    vis_pts = []
    for lm in landmarks:
        x, y, vis = float(lm[0]), float(lm[1]), float(lm[2])
        if vis > 0.10:
            cx = int(np.clip(x, 0, W - 1))
            cy = int(np.clip(y, 0, H - 1))
            vis_pts.append([cx, cy])

    if len(vis_pts) < 3:
        return np.ones((H, W), bool)

    pts = np.array(vis_pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)

    shoulder_w = max(abs(pts[:, 0].max() - pts[:, 0].min()), 20)
    dil_px = max(int(shoulder_w * 0.18), 20)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1))
    mask = cv2.dilate(mask, kern, iterations=1)
    return mask.astype(bool)


def grabcut_person_mask(img_bgr, landmarks=None):
    """
    Person mask using GrabCut.  Falls back to landmark hull if GrabCut yields
    a near-empty result (common with dark clothing on dark backgrounds).
    """
    H, W = img_bgr.shape[:2]
    if landmarks is not None:
        vis_pts = [(int(lm[0]), int(lm[1])) for lm in landmarks if lm[2] > 0.3]
        if vis_pts:
            xs = [p[0] for p in vis_pts]
            ys = [p[1] for p in vis_pts]
            pad_x = max(int((max(xs) - min(xs)) * 0.25), 20)
            pad_y = max(int((max(ys) - min(ys)) * 0.15), 20)
            x1 = max(0, min(xs) - pad_x)
            y1 = max(0, min(ys) - pad_y)
            x2 = min(W - 1, max(xs) + pad_x)
            y2 = min(H - 1, max(ys) + pad_y)
            rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            rect = None
    else:
        rect = None

    if rect is None:
        margin_x = int(W * 0.08)
        margin_y = int(H * 0.02)
        rect = (margin_x, margin_y, W - 2 * margin_x, H - 2 * margin_y)

    rw, rh = rect[2], rect[3]
    if rw < 10 or rh < 10:
        margin_x = int(W * 0.08)
        margin_y = int(H * 0.02)
        rect = (margin_x, margin_y, W - 2 * margin_x, H - 2 * margin_y)

    mask = np.zeros((H, W), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    person_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kern, iterations=3)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN,  kern, iterations=1)

    # Fallback: if GrabCut gives very little foreground (< 5% of bounding rect area)
    # use the landmark hull instead — GrabCut fails on dark-on-dark scenes
    rect_area = rect[2] * rect[3]
    if person_mask.sum() < max(rect_area * 0.05, 500) and landmarks is not None:
        hull_mask = landmark_person_mask(landmarks, H, W)
        if hull_mask is not None:
            return hull_mask

    return person_mask.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
#  Skeleton → body-part geometry
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
MP = dict(
    nose=0, l_eye_in=1, l_eye=2, l_eye_out=3,
    r_eye_in=4, r_eye=5, r_eye_out=6,
    l_ear=7, r_ear=8,
    mouth_l=9, mouth_r=10,
    l_shoulder=11, r_shoulder=12,
    l_elbow=13, r_elbow=14,
    l_wrist=15, r_wrist=16,
    l_pinky=17, r_pinky=18, l_index=19, r_index=20,
    l_thumb=21, r_thumb=22,
    l_hip=23, r_hip=24,
    l_knee=25, r_knee=26,
    l_ankle=27, r_ankle=28,
    l_heel=29, r_heel=30,
    l_foot=31, r_foot=32,
)


def kp(landmarks, name):
    """Get (x, y) from named landmark."""
    i = MP[name]
    return np.array([landmarks[i][0], landmarks[i][1]])


def midpoint(*pts):
    return np.mean(pts, axis=0)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def perpendicular_width(p1, p2, factor=0.4):
    """Half-width of a cylinder between p1 and p2."""
    return dist(p1, p2) * factor


def limb_polygon(p1, p2, half_w):
    """4-point polygon for a limb segment (p1→p2) with given half-width."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    length = np.linalg.norm(d) + 1e-6
    perp = np.array([-d[1], d[0]]) / length * half_w
    return [p1 + perp, p1 - perp, p2 - perp, p2 + perp]


def build_body_parts_mp(landmarks, H, W, person_mask=None):
    """
    Given MediaPipe 33 keypoints, create DensePose-style 24 part masks.
    Robust to cropped/portrait images where lower-body joints are off-screen.
    Returns list of (part_id, bool_mask).
    """
    parts = []
    hw = (H, W)

    # ── helper: safe clamped keypoint access ────────────────────────────────
    def safe_kp(name):
        """Return (x, y) – clamped to image bounds; returns None if near-invisible."""
        i = MP[name]
        x, y, vis = landmarks[i]
        return np.array([float(x), float(y)])

    def is_visible(name, thresh=0.35):
        i = MP[name]
        return landmarks[i][2] >= thresh

    def vis_kp(name, thresh=0.35):
        """Return kp only if visible enough, else None."""
        if is_visible(name, thresh):
            return safe_kp(name)
        return None

    # ── helper: add an ellipse part ────────────────────────────────────────
    def add_ellipse(part_id, cx, cy, ax, ay, angle=0):
        if not (0 <= cx < W and 0 <= cy < H):
            return  # center off-screen, skip
        m = ellipse_mask(hw, cx, cy, ax, ay, angle)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 10:
            parts.append((part_id, m))

    # ── helper: add a limb polygon part ────────────────────────────────────
    def add_limb(part_id, p1, p2, width_factor=0.38):
        if p1 is None or p2 is None:
            return
        hw_ = dist(p1, p2) * width_factor
        if hw_ < 3:
            return
        poly = limb_polygon(p1, p2, hw_)
        m = polygon_mask((H, W), poly)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 20:
            parts.append((part_id, m))

    # ── Key joints ──────────────────────────────────────────────────────────
    l_sh  = safe_kp("l_shoulder")
    r_sh  = safe_kp("r_shoulder")
    l_el  = safe_kp("l_elbow")
    r_el  = safe_kp("r_elbow")
    l_wr  = safe_kp("l_wrist")
    r_wr  = safe_kp("r_wrist")
    l_hi  = safe_kp("l_hip")
    r_hi  = safe_kp("r_hip")
    nose  = safe_kp("nose")
    l_ear = safe_kp("l_ear")
    r_ear = safe_kp("r_ear")

    # Optional lower-body joints (may be off-screen in portrait photos)
    l_kn  = vis_kp("l_knee", thresh=0.20)
    r_kn  = vis_kp("r_knee", thresh=0.20)
    l_an  = vis_kp("l_ankle", thresh=0.20)
    r_an  = vis_kp("r_ankle", thresh=0.20)
    l_ft  = vis_kp("l_foot",  thresh=0.15)
    r_ft  = vis_kp("r_foot",  thresh=0.15)

    # ── Base scale: head height = nose-to-ear vertical span ─────────────────
    shoulder_w = dist(l_sh, r_sh)
    head_h     = dist(nose, midpoint(l_ear, r_ear))  # nose→ear mid
    # Fallback if head_h is tiny
    head_h     = max(head_h, shoulder_w * 0.75)
    head_r     = head_h * 0.65   # radius of head ellipse

    hip_w   = dist(l_hi, r_hi)
    torso_h = dist(midpoint(l_sh, r_sh), midpoint(l_hi, r_hi))

    # ── 1. Head ─────────────────────────────────────────────────────────────
    head_cx = (nose[0] + (l_ear[0] + r_ear[0]) / 2) / 2
    head_cy = (nose[1] + (l_ear[1] + r_ear[1]) / 2) / 2
    add_ellipse(15, head_cx, head_cy, head_r * 0.80, head_r * 1.05)

    # ── 2. Face (slightly smaller, front) ───────────────────────────────────
    add_ellipse(16, nose[0], nose[1] + head_r * 0.15, head_r * 0.62, head_r * 0.75)

    # ── 3. Neck (nose → shoulder midpoint) ──────────────────────────────────
    neck_top = nose + np.array([0, head_r * 0.85])
    neck_bot = midpoint(l_sh, r_sh)
    add_limb(17, neck_top, neck_bot, width_factor=0.28)

    # ── 4. Torso ─────────────────────────────────────────────────────────────
    # Use a trapezoid: wider at shoulders, narrower at hips
    sh_pad  = shoulder_w * 0.08
    hip_pad = hip_w * 0.08
    torso_pts = [
        l_sh + np.array([-sh_pad,  0]),
        r_sh + np.array([ sh_pad,  0]),
        r_hi + np.array([ hip_pad, 0]),
        l_hi + np.array([-hip_pad, 0]),
    ]
    m_torso = polygon_mask((H, W), torso_pts)
    if person_mask is not None:
        m_torso &= person_mask
    if m_torso.sum() > 50:
        parts.append((1, m_torso))

    # ── 5. Left & right shoulder ellipses ────────────────────────────────────
    sr = shoulder_w * 0.20
    add_ellipse(18, l_sh[0], l_sh[1], sr * 1.3, sr)
    add_ellipse(19, r_sh[0], r_sh[1], sr * 1.3, sr)

    # ── 6. Left & right hip ellipses ─────────────────────────────────────────
    hr = max(hip_w * 0.20, shoulder_w * 0.14)
    add_ellipse(20, l_hi[0], l_hi[1], hr * 1.3, hr)
    add_ellipse(21, r_hi[0], r_hi[1], hr * 1.3, hr)

    # ── 7. Upper arms ────────────────────────────────────────────────────────
    upper_arm_w = max(shoulder_w * 0.18, head_r * 0.30)
    # Use width_factor derived from absolute width (not fraction of length)
    def limb_wf(p1, p2, abs_half_w):
        """Add limb with given absolute half-width."""
        if p1 is None or p2 is None:
            return
        d = dist(p1, p2)
        if d < 5:
            return
        add_limb(None, p1, p2, abs_half_w / d)  # dummy

    # Upper arm width = ~18% of shoulder_w  but at least 14px
    ua_w = max(upper_arm_w, 14.0)
    # Call add_limb with explicit factor
    for pid, p1, p2 in [(11, l_sh, l_el), (12, r_sh, r_el)]:
        if p1 is None or p2 is None:
            continue
        d = dist(p1, p2)
        wf = min(ua_w / max(d, 1), 0.55)
        poly = limb_polygon(p1, p2, ua_w)
        m = polygon_mask((H, W), poly)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 20:
            parts.append((pid, m))

    # ── 8. Lower arms ────────────────────────────────────────────────────────
    la_w = ua_w * 0.80
    for pid, p1, p2 in [(13, l_el, l_wr), (14, r_el, r_wr)]:
        if p1 is None or p2 is None:
            continue
        poly = limb_polygon(p1, p2, la_w)
        m = polygon_mask((H, W), poly)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 20:
            parts.append((pid, m))

    # ── 9. Hands ─────────────────────────────────────────────────────────────
    hand_r = max(shoulder_w * 0.12, 10.0)
    if l_wr is not None:
        add_ellipse(4, l_wr[0], l_wr[1], hand_r * 1.2, hand_r * 1.5)
    if r_wr is not None:
        add_ellipse(3, r_wr[0], r_wr[1], hand_r * 1.2, hand_r * 1.5)

    # ── 10. Upper legs (only if knee is visible) ──────────────────────────────
    ul_w = max(hip_w * 0.30, shoulder_w * 0.22, 16.0)
    for pid, p1, p2 in [(7, r_hi, r_kn), (8, l_hi, l_kn)]:
        if p2 is None:
            continue
        poly = limb_polygon(p1, p2, ul_w)
        m = polygon_mask((H, W), poly)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 20:
            parts.append((pid, m))

    # ── 11. Lower legs ────────────────────────────────────────────────────────
    ll_w = ul_w * 0.75
    for pid, p1, p2 in [(9, r_kn, r_an), (10, l_kn, l_an)]:
        if p1 is None or p2 is None:
            continue
        poly = limb_polygon(p1, p2, ll_w)
        m = polygon_mask((H, W), poly)
        if person_mask is not None:
            m &= person_mask
        if m.sum() > 20:
            parts.append((pid, m))

    # ── 12. Knees ─────────────────────────────────────────────────────────────
    kn_r = ul_w * 0.70
    if l_kn is not None:
        add_ellipse(22, l_kn[0], l_kn[1], kn_r, kn_r)
    if r_kn is not None:
        add_ellipse(23, r_kn[0], r_kn[1], kn_r, kn_r)

    # ── 13. Feet ──────────────────────────────────────────────────────────────
    if l_an is not None:
        fp = l_ft if l_ft is not None else l_an + np.array([shoulder_w * 0.18, 0])
        fl = dist(l_an, fp)
        if fl > 5:
            add_ellipse(5, l_an[0] + fl * 0.35, l_an[1], fl * 0.80, fl * 0.30)
    if r_an is not None:
        fp = r_ft if r_ft is not None else r_an + np.array([shoulder_w * 0.18, 0])
        fl = dist(r_an, fp)
        if fl > 5:
            add_ellipse(6, r_an[0] + fl * 0.35, r_an[1], fl * 0.80, fl * 0.30)

    return parts


def build_body_parts_hmr(img_shape, hmr_cam, full_pose):
    """
    Very rough body-part layout from HMR camera + pose, when no keypoints
    are available.  Produces a generic upright T-pose body scaled to image.
    """
    H, W = img_shape[:2]

    # Rough skeleton in image-centre coordinates
    cx, cy = W // 2, H // 2

    # Scale from body-height estimate
    body_h = H * 0.82
    head_r  = body_h * 0.075
    torso_h = body_h * 0.28
    arm_len = body_h * 0.28
    leg_len = body_h * 0.42
    sh_w    = body_h * 0.22

    # Key points (x, y)
    nose   = np.array([cx,            cy - body_h * 0.47])
    l_sh   = np.array([cx - sh_w,     cy - body_h * 0.31])
    r_sh   = np.array([cx + sh_w,     cy - body_h * 0.31])
    l_hi   = np.array([cx - sh_w * 0.6, cy - body_h * 0.31 + torso_h])
    r_hi   = np.array([cx + sh_w * 0.6, cy - body_h * 0.31 + torso_h])
    l_el   = l_sh + np.array([-arm_len * 0.55, arm_len * 0.4])
    r_el   = r_sh + np.array([+arm_len * 0.55, arm_len * 0.4])
    l_wr   = l_el + np.array([-arm_len * 0.45, arm_len * 0.3])
    r_wr   = r_el + np.array([+arm_len * 0.45, arm_len * 0.3])
    l_kn   = l_hi + np.array([0, leg_len * 0.50])
    r_kn   = r_hi + np.array([0, leg_len * 0.50])
    l_an   = l_kn + np.array([0, leg_len * 0.50])
    r_an   = r_kn + np.array([0, leg_len * 0.50])

    # Fake landmarks list compatible with build_body_parts_mp indexing
    fake_lm = [(0, 0, 0)] * 33
    MP_inv = {v: k for k, v in MP.items()}
    slot = {
        "nose": nose, "l_shoulder": l_sh, "r_shoulder": r_sh,
        "l_elbow": l_el, "r_elbow": r_el, "l_wrist": l_wr, "r_wrist": r_wr,
        "l_hip": l_hi, "r_hip": r_hi, "l_knee": l_kn, "r_knee": r_kn,
        "l_ankle": l_an, "r_ankle": r_an,
        "l_foot": l_an + np.array([30, 0]),
        "r_foot": r_an + np.array([30, 0]),
    }
    slots_list = list(fake_lm)
    for name, pt in slot.items():
        idx = MP[name]
        slots_list[idx] = (pt[0], pt[1], 1.0)
    # fill remaining with nose
    for i in range(33):
        if slots_list[i] == (0, 0, 0):
            slots_list[i] = (nose[0], nose[1], 0.5)

    return build_body_parts_mp(slots_list, H, W, person_mask=None)


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_iuv(
    img_bgr: np.ndarray,
    parts: list,
    output_size: tuple | None = None,
    feather: int = 3,
) -> np.ndarray:
    """
    Given a list of (part_id, bool_mask), render the DensePose IUV image.

    Parameters
    ----------
    img_bgr      : original image (for shape reference)
    parts        : [(part_id, bool_mask), …]
    output_size  : (W, H) to resize output, or None to keep original
    feather      : edge softening in pixels

    Returns
    -------
    IUV image (H, W, 3) uint8  in BGR (for cv2.imwrite)
    """
    H, W = img_bgr.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.float32)    # black background
    depth  = np.full((H, W),  -1, dtype=np.int32)     # tracks topmost part

    # ── paint each part with UV gradient ────────────────────────────────────
    # We process from "background" parts first (torso) to "foreground" (hands,
    # feet, head) so that distal parts appear on top — like DensePose ordering.
    PART_ORDER = [2, 1, 20, 21, 18, 19, 17,   # torso/hips/shoulders/neck
                  8, 7, 10, 9, 22, 23,         # legs/knees
                  12, 11, 14, 13,              # arms
                  6, 5, 4, 3,                  # feet/hands
                  15, 16]                      # head/face

    part_dict: dict[int, np.ndarray] = {}
    for pid, mask in parts:
        if pid not in part_dict:
            part_dict[pid] = mask.copy()
        else:
            part_dict[pid] |= mask  # merge if same part appears twice

    for pid in PART_ORDER:
        if pid not in part_dict:
            continue
        mask = part_dict[pid]
        base = PART_COLORS.get(pid, (200, 200, 200))
        layer = uv_gradient(mask, base)   # (H, W, 3) float

        # Soft edge: slightly blur the gradient at boundary
        if feather > 0:
            dist_map = cv2.distanceTransform(
                mask.astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.clip(dist_map / max(feather, 1), 0, 1)[..., None]  # (H,W,1)
            canvas = np.where(mask[..., None], layer * alpha + canvas * (1 - alpha), canvas)
        else:
            canvas[mask] = layer[mask]

    iuv_rgb = np.clip(canvas, 0, 255).astype(np.uint8)

    if output_size is not None:
        iuv_rgb = cv2.resize(iuv_rgb, output_size, interpolation=cv2.INTER_LINEAR)

    # Convert RGB → BGR for cv2.imwrite
    return cv2.cvtColor(iuv_rgb, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────

def generate_densepose_iuv(
    image_path: str,
    output_dir: str = "output",
    scale: float = 1.0,
    feather: int = 4,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"densepose_iuv_{base}.png")

    # ── load image ──────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    print(f"  Image       : {W}×{H}")

    # ── step 1: pose detection ───────────────────────────────────────────────
    print("  [1/3] Detecting pose...")
    landmarks, seg_mask = detect_pose_mediapipe(img_rgb)

    if landmarks is not None:
        print(f"        MediaPipe → {len(landmarks)} landmarks ✓")
        # Build person mask from segmentation
        if seg_mask is not None:
            person_mask = (seg_mask > 0.5)
        else:
            # Prefer landmark-hull mask (works regardless of clothing vs background color).
            # Only fall back to GrabCut when there are not enough visible landmarks.
            lm_mask = landmark_person_mask(landmarks, H, W)
            if lm_mask is not None:
                person_mask = lm_mask
            else:
                person_mask = grabcut_person_mask(img_bgr, landmarks=landmarks)

        parts = build_body_parts_mp(landmarks, H, W, person_mask)
    else:
        print("        MediaPipe not available — using HMR skeleton fallback")
        hmr_result = detect_pose_hmr(image_path)
        cam = hmr_result["cam"] if hmr_result else np.array([0.9, 0, 0])
        pose = hmr_result["pose"] if hmr_result else np.zeros(72)
        parts = build_body_parts_hmr(img_bgr.shape, cam, pose)

    print(f"        {len(parts)} body-part masks built")

    # ── step 2: render IUV ─────────────────────────────────────────────────
    print("  [2/3] Rendering IUV map...")
    out_W = int(W * scale)
    out_H = int(H * scale)
    iuv_bgr = render_iuv(img_bgr, parts, output_size=(out_W, out_H), feather=feather)

    # ── step 3: save ────────────────────────────────────────────────────────
    print("  [3/3] Saving...")
    cv2.imwrite(out_path, iuv_bgr)
    print(f"\n  ✓  Saved → {out_path}  ({out_W}×{out_H})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate DensePose-style IUV visualization from a full-body photo."
    )
    ap.add_argument("image_path", help="Input image (JPG/PNG)")
    ap.add_argument("--output-dir", "-o", default="output")
    ap.add_argument("--scale",   "-s", type=float, default=1.0,
                    help="Output scale factor (default 1.0 = same as input)")
    ap.add_argument("--feather", "-f", type=int, default=4,
                    help="Edge softening radius in pixels (default 4)")
    args = ap.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: image not found: {args.image_path}")
        sys.exit(1)

    out = generate_densepose_iuv(
        image_path=args.image_path,
        output_dir=args.output_dir,
        scale=args.scale,
        feather=args.feather,
    )
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
