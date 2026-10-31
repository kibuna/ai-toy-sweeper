"""実機の検出（背景との差、field.real_rig）を映像に重ねる（検出の確かめの画面 scripts/check_detection.py と、管理の画面 /admin）。

重ねるもの: 物と見た画素（赤）、ロボットを除く範囲（青）、見つけた物（緑の丸と、床の長さ × 幅 mm）、ロボットの車軸の中点（黄）、
作業領域（緑の線）、物を探す範囲（水色の線、real_rig の roi）。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..perception_oblique import BackgroundDetector, BlobTooLarge
from .hardware import ArucoRobotPose, RobotMarkerError
from .interfaces import Frame


def _draw_area(out: np.ndarray, rig) -> None:
    """作業領域（緑）と物を探す範囲（水色）の線。辺は細かく分けて投影する（レンズの歪みで曲がる）。"""
    hx, hy = rig.area_size[0] / 2, rig.area_size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    pts = np.vstack([np.linspace(a, b, 30) for a, b in zip(corners, corners[1:] + corners[:1])])
    cv2.polylines(out, [np.round(rig.hom.to_px(pts)).astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
    cnts, _ = cv2.findContours(rig.detector.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2, cv2.LINE_AA)


def detection_overlay(rig, rgb: np.ndarray, jaw_angle: float) -> tuple[np.ndarray, dict]:
    """rig: field.real_rig.RealRig。戻り値は重ねた画像（BGR）と、ロボットの姿勢・見つけた物などの情報。
    塊が大きすぎるときは、止めずに理由を情報に入れる（見るための画面。RecordingDetector の画像の保存もしない）。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    info = {"robot": None, "detections": []}
    frame = Frame(rgb, time.monotonic(), 0)
    try:
        robot = ArucoRobotPose.estimate(rig.pose, frame)   # 見るための画面なので、見えないときの画像は残さない
    except RobotMarkerError as e:
        info["robot_error"] = str(e)
        robot = None
    det = rig.detector
    mask = None if robot is None else det.mask(rgb.shape[:2], robot, jaw_angle)
    fg = det.foreground(rgb, mask)
    out = bgr.copy()
    out[fg > 0] = (0, 0, 255)
    _draw_area(out, rig)
    if mask is not None:
        out[mask > 0] = (out[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        u, v = rig.hom.to_px(np.array([robot.xy]), 0.0)[0].astype(int)
        cv2.circle(out, (u, v), 10, (0, 255, 255), -1)
        info["robot"] = [round(robot.x, 3), round(robot.y, 3), round(float(np.degrees(robot.yaw)), 1)]
    if robot is not None:
        try:
            dets = BackgroundDetector.__call__(det, frame, robot, jaw_angle)
        except BlobTooLarge as e:
            info["detect_error"] = str(e)
            x0, y0, x1, y1 = e.bbox_px
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 3)
            dets = []
        for area, (x0, y0, x1, y1) in det.last_large:
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 3)
            info["detect_error"] = f"大きすぎて物と見なかった塊（{area:.3f} m²）。続くならカメラが動いた: 校正し直す"
        for d in dets:
            u, v = rig.hom.to_px(np.array([d.pose.xy]), 0.008)[0].astype(int)
            cv2.circle(out, (u, v), 18, (0, 255, 0), 3)
            cv2.putText(out, f"{d.size_m[0] * 1000:.0f}x{d.size_m[1] * 1000:.0f}", (u + 20, v), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            info["detections"].append({"xy_m": [round(d.pose.x, 3), round(d.pose.y, 3)],
                                       "size_mm": [round(1000 * s) for s in d.size_m],
                                       "rgb": [round(c) for c in d.mean_rgb]})
    info["foreground_px"] = int(fg.sum())
    return out, info
