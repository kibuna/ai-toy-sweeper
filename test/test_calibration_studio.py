"""本番の画面の校正（toy_sweeper.field.calibration_web）の試験。カメラの姿勢が分かっている合成画像をフレームとして流し、
床の写真で外部パラメータを求めて保存し、前の校正が履歴に残ることを確かめる。"""
import asyncio
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_calibration import WALL, _floor_texture, _look_at, _render
from toy_sweeper.calibration import CalibrationError, CameraModel, intrinsics_from_fov
from toy_sweeper.config import load_all
from toy_sweeper.field.calibration_web import (BackgroundError, CalibrationStudio, calibration_router,
                                              load_background)
from toy_sweeper.field.interfaces import Frame
from toy_sweeper.field.latest import Latest
from toy_sweeper.markers import load_markers_cfg

INTR = intrinsics_from_fov((1280, 720), 1.22)
CAM = np.array([0.05, -0.85, 0.8])


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture(scope="module")
def m():
    return load_markers_cfg()


def _floor_frame(cfg, m) -> Frame:
    rvec, tvec = _look_at(CAM, (0.0, -0.08, 0.0))
    bgr = _render(_floor_texture(m, cfg, rotations=(0, 1, 2, 3)), INTR, rvec, tvec)
    return Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.0, 1)


def _studio(cfg, m, tmp_path, frames):
    return CalibrationStudio(cfg, m, frames, tmp_path / "calibration" / "camera.yaml", tmp_path / "photos", WALL,
                             tmp_path / "camera" / "background.png")


def test_floor_needs_intrinsics_then_saves_with_history(cfg, m, tmp_path):
    async def body():
        frames = Latest("camera")
        frames.set(_floor_frame(cfg, m), 0.0)
        s = _studio(cfg, m, tmp_path, frames)
        with pytest.raises(CalibrationError, match="内部パラメータ"):
            s.solve_floor(s.snapshot(max_age_s=None))
        with pytest.raises(CalibrationError, match="まだ"):
            s.save()
        # 前の校正（内部パラメータは同じ、姿勢は違う）を置いておく
        old = CameraModel(INTR, np.zeros(3), np.array([0.0, 0.0, 1.0]), 9.9, (0.8, 0.8), 0.0)
        old.save(s.calib_path)
        r = s.solve_floor(s.snapshot(max_age_s=None))
        assert r["floor_reproj_px"] < 1.0
        assert np.linalg.norm(np.array(r["camera_position_m"]) - CAM) < 0.005
        assert np.allclose(r["work_area_size_m"], cfg.sim.work_area.size_m, atol=0.003) and r["same_as_sim"]
        assert s.check_jpeg()[:2] == b"\xff\xd8"
        saved = s.save()
        assert np.linalg.norm(CameraModel.load(s.calib_path).position() - CAM) < 0.005
        # 校正の床の写真（物とロボットをどけて撮る）を、そのまま背景にする（2026-10-22、床の写真を別に撮る手間を省くため）
        assert saved["background"] == str(s.background_path)
        assert load_background(s.calib_path, s.background_path).shape == (720, 1280, 3)
        hist = list((s.calib_path.parent / "history").glob("camera_*.yaml"))
        assert len(hist) == 1 and CameraModel.load(hist[0]).reproj_px == 9.9
        assert saved["previous"] == str(hist[0])
        # 失敗した写真も残す（後で原因を見る）
        assert sorted(q.name for q in s.photo_dir.glob("floor_*.png")) == ["floor_001.png", "floor_002.png",
                                                                         "floor_002_check.png"]
    asyncio.run(body())


def test_snapshot_rejects_stale_or_missing_frame(cfg, m, tmp_path):
    async def body():
        frames = Latest("camera")
        s = _studio(cfg, m, tmp_path, frames)
        with pytest.raises(CalibrationError, match="フレームがない"):
            s.snapshot(max_age_s=1.0)
        frames.set(_floor_frame(cfg, m), 0.0)   # 時刻 0（とても古い）
        with pytest.raises(CalibrationError, match="古い"):
            s.snapshot(max_age_s=1.0)
    asyncio.run(body())


def test_router_reports_calibration_errors(cfg, m, tmp_path):
    """校正できないとき（内部パラメータがない）は 400 で理由を返す（黙って何もしないにしない）。"""
    frames = Latest("camera")
    frames.set(_floor_frame(cfg, m), time.monotonic())
    s = _studio(cfg, m, tmp_path, frames)
    app = FastAPI()
    app.include_router(calibration_router(s))
    c = TestClient(app)
    assert "カメラの校正" in c.get("/calibrate").text
    r = c.post("/calibrate/floor")
    assert r.status_code == 400 and "内部パラメータ" in r.json()["detail"]
    st = c.get("/calibrate/status").json()
    assert st["saved"] is None and st["board_photos"] == 0


def test_background_is_tied_to_calibration(cfg, m, tmp_path):
    """背景（空の床の写真）は、撮ったときの校正と結び付ける。校正をし直したら古い背景は使えない（例外）。"""
    frames = Latest("camera")
    frames.set(_floor_frame(cfg, m), time.monotonic())
    s = _studio(cfg, m, tmp_path, frames)
    with pytest.raises(CalibrationError, match="校正"):
        s.take_background(s.snapshot())          # 校正の前は撮らない
    CameraModel(INTR, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.5, (0.8, 0.8), 0.0).save(s.calib_path)
    r = s.take_background(s.snapshot())
    assert r["size"] == [1280, 720]
    img = load_background(s.calib_path, s.background_path)
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8          # RGB（Frame.image と同じ）
    CameraModel(INTR, np.zeros(3), np.array([0.0, 0.0, 1.2]), 0.5, (0.8, 0.8), 0.0).save(s.calib_path)   # 校正し直し
    with pytest.raises(BackgroundError, match="校正"):
        load_background(s.calib_path, s.background_path)


def test_area_problems_block_saving(cfg, m, tmp_path):
    """作業領域が学習の大きさの範囲の外の校正は保存しない（2026-10-26、奥行き 0.469 m（学習は 0.6〜1.0 m）の校正をそのまま
    使った）。ロボットが作業領域 + 外の余白まで出たときにマーカーが画像から出る所があるのは、知らせるだけで保存はする
    （2026-10-30、隅のわずかなはみ出しで保存を断っていたため、警告だけにした）。"""
    import dataclasses
    from toy_sweeper.field.calibration_web import area_problems, area_warnings
    cam = CameraModel(INTR, *_look_at(CAM, (0.0, -0.08, 0.0)), 0.5, (0.8, 0.8), 0.0)
    free = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=dataclasses.replace(
        cfg.task.sampling, work_area_size_range_m=((0.6, 1.0), (0.6, 1.0)))))
    assert area_problems(free, m, cam) == [] and area_warnings(free, m, cam) == []
    small = CameraModel(INTR, *_look_at(CAM, (0.0, -0.08, 0.0)), 0.5, (0.69, 0.47), 0.0)
    assert any("学習" in p for p in area_problems(free, m, small))
    big_margin = dataclasses.replace(free, sim=dataclasses.replace(free.sim, walls=dataclasses.replace(
        free.sim.walls, out_of_area_margin_m=1.0)))
    assert area_problems(big_margin, m, cam) == []
    assert any("画像" in w for w in area_warnings(big_margin, m, cam))
