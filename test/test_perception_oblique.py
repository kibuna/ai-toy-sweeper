"""斜め上のカメラの、背景との差による物体の検出（toy_sweeper.perception_oblique）の試験。合成画像（平らな物体）。"""
import cv2
import numpy as np
import pytest

from toy_sweeper.calibration import CameraModel, intrinsics_from_fov
from toy_sweeper.perception_oblique import BackgroundDetectCfg, BackgroundDetector, robot_mask

PPM = 1500
HALF = 0.5
CFG = BackgroundDetectCfg(luma_weight=0.3, threshold=25.0, min_area_m2=1e-4, max_area_m2=0.02, object_mid_height_m=0.0,
                          robot_height_m=0.04, robot_margin_m=0.005, jaw_margin_m=0.003,
                          held_radius_m=0.025, reassociate_max_m=0.2, reassociate_size_ratio=1.4, merged_max_length_m=0.0,
                          track_chroma_gate=0.0,
                          recover_back_steps=4, recover_search_m=0.4, recover_max=3,
                          recover_unseen_steps=8, open_px=3, close_px=0, color_split=False, split_chroma_min=15.0,
                          split_min_ab=30.0, split_min_fraction=0.3, split_min_separation=0.4)


def _cam():
    intr = intrinsics_from_fov((1280, 720), 1.22)
    pos = np.array([0.0, -0.85, 0.8])
    z = np.array([0.0, -0.08, 0.0]) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return CameraModel(intr, cv2.Rodrigues(R)[0], -R @ pos, 0.0, (0.8, 0.8), 0.0)


def _floor(objects):
    n = int(2 * HALF * PPM)
    tex = np.full((n, n, 3), (200, 190, 170), np.uint8)
    px = lambda x, y: (int((x + HALF) * PPM), int((HALF - y) * PPM))
    cv2.rectangle(tex, px(-0.4, 0.4), px(-0.2, 0.2), (180, 215, 180), -1)     # ゴールの緑の面（背景にもある）
    for (x, y, yaw, sx, sy, rgb) in objects:
        c, s = np.cos(yaw), np.sin(yaw)
        pts = [px(x + c * a - s * b, y + s * a + c * b) for a, b in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]
        cv2.fillConvexPoly(tex, np.int32(pts), rgb)
    return tex


def _render(tex, cam):
    R = cam.R
    H = cam.intr.K @ np.column_stack([R[:, 0], R[:, 1], cam.tvec.ravel()])
    T = np.array([[1 / PPM, 0, -HALF], [0, -1 / PPM, HALF], [0, 0, 1]])
    return cv2.warpPerspective(tex, H @ T, cam.intr.size, flags=cv2.INTER_AREA, borderValue=(120, 120, 120))


def test_finds_objects_not_goal_patch():
    cam = _cam()
    objs = [(-0.3, 0.3, 0.3, 0.032, 0.016, (220, 50, 50)),      # ゴールの面の上の赤い LEGO
            (0.1, -0.1, 1.2, 0.065, 0.03, (40, 90, 220)),       # 青いミニカー
            (0.25, 0.3, 0.0, 0.032, 0.016, (150, 205, 40))]     # 奥の黄緑の LEGO
    det = BackgroundDetector(cam, _render(_floor([]), cam), CFG, robot_polygons=lambda r, j: [], jaw_top_m=0.01)
    found = det.detect(_render(_floor(objs), cam), None)
    assert len(found) == 3
    for x, y, yaw, sx, sy, rgb in objs:
        d = min(found, key=lambda f: np.hypot(f.pose.x - x, f.pose.y - y))
        assert np.hypot(d.pose.x - x, d.pose.y - y) < 0.004
        assert d.size_m[0] == pytest.approx(sx, abs=0.006) and d.size_m[1] == pytest.approx(sy, abs=0.006)
        assert np.linalg.norm(np.array(d.mean_rgb) - rgb) < 40


def test_robot_region_is_excluded():
    cam = _cam()
    objs = [(0.0, 0.0, 0.0, 0.08, 0.08, (60, 90, 200))]          # ロボットの車体と同じ所の箱
    det = BackgroundDetector(cam, _render(_floor([]), cam), CFG, robot_polygons=lambda r, j: [], jaw_top_m=0.01)
    img = _render(_floor(objs), cam)
    poly = [[(-0.045, -0.045), (0.045, -0.045), (0.045, 0.045), (-0.045, 0.045)]]
    m = robot_mask(img.shape[:2], cam, poly, [CFG.robot_height_m], [CFG.robot_margin_m])
    assert det.detect(img, None) and not det.detect(img, m)


def test_huge_change_raises():
    cam = _cam()
    det = BackgroundDetector(cam, _render(_floor([]), cam), CFG, robot_polygons=lambda r, j: [], jaw_top_m=0.01)
    dark = (_render(_floor([]), cam) * 0.4).astype(np.uint8)      # 照明が大きく変わった
    with pytest.raises(RuntimeError, match="校正し直す"):
        det.detect(dark, None)


def test_vlm_identifier_boxes_and_names():
    """指示を受けたときの VLM の同定: 追跡の床の外形を画像の枠にして聞き、"other" は名前を付けない。"""
    from toy_sweeper.field.interfaces import FieldStop, Frame, TrackView
    from toy_sweeper.field.vlm_identify import VlmObjectIdentifier
    from toy_sweeper.geometry import Pose2D
    from toy_sweeper.spaces import ObjectType
    cam = _cam()
    tracks = {3: TrackView(3, Pose2D(0.1, -0.1, 0.3), (0.032, 0.016), (200, 50, 50), 0),
              7: TrackView(7, Pose2D(-0.2, 0.25, 0.0), (0.065, 0.03), (40, 90, 220), 0),
              9: TrackView(9, Pose2D(0.3, 0.3, 0.0), (0.03, 0.03), (90, 90, 90), 0)}

    class Server:
        def __init__(self, answers):
            self.answers, self.boxes = answers, None

        def identify(self, png, boxes):
            self.boxes = boxes
            return {"objects": self.answers, "seconds": 0.1}
    s = Server([{"class": "lego", "color": "red"}, {"class": "toy_car", "color": "blue"}, {"class": "other", "color": "black"}])
    img = np.zeros((720, 1280, 3), np.uint8)
    out = VlmObjectIdentifier(s, cam, 0.025).identify(Frame(img, 0.0, 1), tracks)
    assert {i: (o.type, o.color) for i, o in out.items()} == {3: (ObjectType.LEGO, "red"), 7: (ObjectType.TOY_CAR, "blue")}
    u, v = cam.to_px(np.array([[0.1, -0.1]]), 0.0)[0]
    x0, y0, x1, y1 = s.boxes[0]
    assert x0 < u < x1 and y0 < v < y1
    with pytest.raises(FieldStop):
        VlmObjectIdentifier(Server([{"error": "x"}, {"class": "lego", "color": "red"}, {"class": "lego", "color": "red"}]),
                            cam, 0.025).identify(Frame(img, 0.0, 1), tracks)


def _split_detector():
    import dataclasses
    cfg = dataclasses.replace(CFG, color_split=True)
    bg = np.full((200, 300, 3), (190, 160, 110), np.uint8)    # 木の床の色
    return BackgroundDetector(_cam_small(), bg, cfg, lambda r, j: [], 0.01)


def _cam_small():
    intr = intrinsics_from_fov((300, 200), 1.22)
    return CameraModel(intr, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0, (0.8, 0.8), 0.0)


def _blob_parts(rgb):
    det = _split_detector()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    blob = (np.abs(rgb.astype(int) - (190, 160, 110)).sum(axis=2) > 0).astype(np.uint8)
    return det.split_by_color(lab, blob)


def test_split_touching_objects_of_different_colors():
    """くっついた違う色の物体（黄緑とオレンジの LEGO、2026-10-22 の実機）は 2 つに分ける。"""
    rgb = np.full((200, 300, 3), (190, 160, 110), np.uint8)
    rgb[80:120, 100:150] = (150, 200, 60)    # 黄緑
    rgb[80:120, 150:200] = (240, 140, 30)    # オレンジ（くっついている）
    parts = _blob_parts(rgb)
    assert len(parts) == 2
    cols = sorted(int(np.nonzero(p)[1].mean()) for p in parts)
    assert abs(cols[0] - 125) < 3 and abs(cols[1] - 175) < 3


def test_no_split_for_car_with_dark_windows_or_color_fringe():
    """ミニカー（黄の車体の中に暗い窓）と、縁だけ色のずれた 1 個の物体は分けない。"""
    car = np.full((200, 300, 3), (190, 160, 110), np.uint8)
    car[80:120, 100:200] = (230, 200, 20)
    car[90:110, 130:170] = (40, 40, 45)      # 窓（色を持たない）
    assert len(_blob_parts(car)) == 1
    ring = np.full((200, 300, 3), (190, 160, 110), np.uint8)
    ring[78:122, 98:152] = (200, 120, 60)    # 縁（床と混ざった色）
    ring[80:120, 100:150] = (60, 150, 60)    # 緑の LEGO
    assert len(_blob_parts(ring)) == 1


def test_mask_uses_height_per_body_part():
    """車体を高さの違う部品に分けて除ける（body_heights_m）。実機の前のグリッパーの取り付け部は床から 約 4 cm で、後ろ
    （micro:bit・マーカー、約 7 cm）と同じ高さで除くと、爪の前の物まで隠れた（2026-10-30。実測でグリッパーの
    支点あたりは 4 cm、手の部分は 1 cm）。指定しなければ今までどおり
    先頭の 1 つが車体（cfg.robot_height_m）。"""
    cam = _cam()
    rear = [(-0.045, -0.05), (0.06, -0.05), (0.06, 0.05), (-0.045, 0.05)]
    mount = [(0.06, -0.03), (0.08, -0.03), (0.08, 0.03), (0.06, 0.03)]
    jaw = [(0.08, -0.01), (0.1, -0.01), (0.1, 0.01), (0.08, 0.01)]
    polys = lambda r, j: [rear, mount, jaw]
    bg = _render(_floor([]), cam)
    two = BackgroundDetector(cam, bg, CFG, robot_polygons=polys, jaw_top_m=0.01, body_heights_m=(0.07, 0.04))
    want = robot_mask(bg.shape[:2], cam, [rear, mount, jaw], [0.07, 0.04, 0.01],
                      [CFG.robot_margin_m, CFG.robot_margin_m, CFG.jaw_margin_m])
    assert np.array_equal(two.mask(bg.shape[:2], None, 0.0), want)
    one = BackgroundDetector(cam, bg, CFG, robot_polygons=polys, jaw_top_m=0.01)
    want1 = robot_mask(bg.shape[:2], cam, [rear, mount, jaw], [CFG.robot_height_m, 0.01, 0.01],
                       [CFG.robot_margin_m, CFG.jaw_margin_m, CFG.jaw_margin_m])
    assert np.array_equal(one.mask(bg.shape[:2], None, 0.0), want1)
    with pytest.raises(ValueError):
        BackgroundDetector(cam, bg, CFG, robot_polygons=polys, jaw_top_m=0.01, body_heights_m=())
