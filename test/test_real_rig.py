"""実機の機器一式（toy_sweeper.field.real_rig.RealRig）の、機器なしで確かめられる部分。"""
import time

import cv2

import numpy as np
import pytest

from test_calibration import _look_at
from toy_sweeper.calibration import CameraModel, intrinsics_from_fov
from toy_sweeper.config import load_all
from toy_sweeper.field.calibration_web import BackgroundError, CalibrationStudio
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import OpenCVCamera, RobotMarkerError, save_robot_marker_yaw
from toy_sweeper.field.interfaces import Frame
from toy_sweeper.field.latest import Latest
from toy_sweeper.field.real_rig import RealRig, real_robot_polygons
from toy_sweeper.geometry import Pose2D
from toy_sweeper.markers import load_markers_cfg
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.task import goal_rects_for_area

SIZE = (0.75, 0.83)


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture
def files(tmp_path, cfg):
    """校正・背景・マーカーの向きのファイル（実機で作るもの）を合成して置く。"""
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.0, -0.87, 0.66]), (0.0, -0.05, 0.0))
    calib = tmp_path / "calibration" / "camera.yaml"
    CameraModel(intr, rvec, tvec, 0.6, SIZE, 0.012).save(calib)
    frames = Latest("camera")
    frames.set(Frame(np.full((720, 1280, 3), 120, np.uint8), 0.0, 1), time.monotonic())
    bg = tmp_path / "camera" / "background.png"
    studio = CalibrationStudio(cfg, load_markers_cfg(), frames, calib, tmp_path / "photos", 0.0, bg)
    studio.take_background(studio.snapshot())
    marker = tmp_path / "calibration" / "robot_marker.yaml"
    save_robot_marker_yaw(marker, -1.55, {"runs": 6})
    return calib, bg, marker


def _real_worlds(session, rig):
    from toy_sweeper.field.web import World, Worlds
    return Worlds({"real": World("実機", session, rig.camera.frames, True)}, "real")


def _rig(cfg, files):
    calib, bg, marker = files
    f = load_field_cfg(cfg)
    return RealRig(cfg, f, load_perception_cfg(), calib, bg, marker, identify_server=None, camera=OpenCVCamera(f.camera),
                   error_dir=calib.parent.parent / "errors")


def test_goals_follow_calibrated_area(cfg, files):
    rig = _rig(cfg, files)
    assert rig.area_size == SIZE
    assert rig.goal_rects() == goal_rects_for_area(cfg.sim, SIZE)


def test_outcome_is_tracked_completion(cfg, files):
    """実機に真値は無い。ランタイムが完了とした（全対象が追跡でゴールの中）ことを成功とし、そう書く。"""
    ok, how = _rig(cfg, files).outcome({"steps": 10})
    assert ok and "追跡" in how


def test_requires_background_and_marker_yaw(cfg, files):
    calib, bg, marker = files
    marker.unlink()
    with pytest.raises(RobotMarkerError):
        _rig(cfg, files)
    save_robot_marker_yaw(marker, -1.55, {"runs": 6})
    bg.unlink()
    with pytest.raises(BackgroundError):
        _rig(cfg, files)


def test_detection_is_limited_to_work_area(cfg, files):
    """作業領域と外の余白の外（柵・ソファ・部屋の奥）は物を探さない。床より上の画素を床に戻すと計算があふれ、遠くに偽の物が出た
    （2026-10-22、実機の画像）。"""
    rig = _rig(cfg, files)
    roi = rig.detector.roi
    cam = rig.hom
    half = np.array(SIZE) / 2
    inside = cam.to_px(np.array([[0.0, 0.0], [half[0] - 0.01, half[1] - 0.01]]))
    outside = cam.to_px(np.array([[0.0, half[1] + 0.3], [half[0] + 0.3, 0.0]]))
    for u, v in inside.astype(int):
        assert roi[v, u] == 1
    for u, v in outside.astype(int):
        assert not (0 <= u < roi.shape[1] and 0 <= v < roi.shape[0]) or roi[v, u] == 0
    rgb = np.full((720, 1280, 3), 120, np.uint8)
    rgb[:40, :] = 255                                    # 画像の上の端（部屋の奥）だけ違う
    assert rig.detector.foreground(rgb, None)[:40].sum() == 0


def test_real_robot_mask_covers_measured_body(cfg):
    """ロボットを除く範囲の車体は実測（基板: 車軸から後ろ 25 mm・前 80 mm・幅 85 mm）に、録画で見えた後輪・車輪の出っ張りを
    足したもの（2026-10-28: 後ろ 45 mm。幅は 2026-10-30 に実測の ±42.5 mm に戻した）。前の 63〜80 mm はグリッパーの取り付け部で
    低い（高さ 4 cm）ので、
    後ろ（7 cm）と別の部品にする（2026-10-30、実測でグリッパーの支点あたりが 4 cm）。並びは 後ろ、取り付け部、爪…。"""
    f = load_field_cfg(cfg)
    polys = real_robot_polygons(cfg, f.real, Pose2D(0.0, 0.0, 0.0), cfg.sim.gripper.jaw_open_angle_rad)
    rear, mount = np.array(polys[0]), np.array(polys[1])
    assert np.isclose(rear[:, 0].min(), -0.045) and np.isclose(rear[:, 0].max(), f.real.mount_x_m)
    assert np.isclose(mount[:, 0].min(), f.real.mount_x_m) and np.isclose(mount[:, 0].max(), 0.08)
    assert np.isclose(rear[:, 1].max(), 0.0425) and np.isclose(mount[:, 1].max(), 0.0425)
    assert f.real.mount_height_m == 0.04 and f.real.robot_mask_height_m == 0.07
    assert len(polys) > 2   # 爪
    rot = np.array(real_robot_polygons(cfg, f.real, Pose2D(0.1, 0.2, np.pi / 2), 0.0)[1])
    assert np.isclose(rot[:, 1].max(), 0.2 + 0.08)   # 向きと位置に合わせて動く


def test_real_rig_masks_mount_lower_than_rear(cfg, files):
    """実機の検出器は、後ろを robot_mask_height_m、取り付け部を mount_height_m、爪を爪の上端の高さで除く。"""
    f = load_field_cfg(cfg)
    rig = _rig(cfg, files)
    assert rig.detector.body_heights_m == (f.real.robot_mask_height_m, f.real.mount_height_m)


def test_real_rig_uses_real_robot_margin(cfg, files):
    """実機の車体を除く余白は real.robot_margin_m（Gazebo と共通の perception_oblique.yaml の値を実機だけ置き換える）。
    2026-10-30、Fable のレビュー: 車体の幅 ±55 mm + 余白 10 mm が実物より広く、下がった後の赤いブロックが左後輪の脇
    (−20, +40) mm でマスクに入って見えなかった。"""
    f = load_field_cfg(cfg)
    rig = _rig(cfg, files)
    assert rig.detector.cfg.robot_margin_m == f.real.robot_margin_m == 0.005


def test_app_screens(cfg, files, tmp_path):
    """管理のアプリの画面（/admin・/calibrate・/）を組み立てられ、管理の状態とログが返る（カメラは開かない）。"""
    from fastapi.testclient import TestClient
    from toy_sweeper.field.admin_web import admin_router
    from toy_sweeper.field.calibration_web import calibration_router
    from toy_sweeper.field.session import TalkSession
    from toy_sweeper.field.web import create_app
    calib, bg, marker = files
    rig = _rig(cfg, files)

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    session = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    app = create_app(_real_worlds(session, rig), 16000, 10.0)
    studio = CalibrationStudio(cfg, load_markers_cfg(), rig.camera.frames, calib, tmp_path / "photos", 0.0, bg)
    app.include_router(calibration_router(studio))
    log_path = tmp_path / "run.log"
    log_path.write_text("".join(f"行 {i}\n" for i in range(300)))
    app.include_router(admin_router(_real_worlds(session, rig), log_path))
    c = TestClient(app)
    assert c.get("/admin").status_code == 200 and c.get("/calibrate").status_code == 200 and c.get("/").status_code == 200
    st = c.get("/admin/status").json()
    assert st["link"]["connected"] is False and st["area_size_m"] == list(SIZE) and st["vlm_last"] is None
    assert c.get("/admin/log?n=3").text == "行 297\n行 298\n行 299\n"
    assert c.get("/admin/log?n=0").status_code == 400
    assert c.get("/calibrate/status").json()["background"]["ok"] is True


def test_reloads_after_recalibration(cfg, files, tmp_path):
    """管理のアプリの中で校正し直したら、次の指示の前に読み込み直す（2026-10-22、起動のときの校正と背景のまま動き、古い背景との差が
    床じゅうに出て「塊が大きすぎる」で止まった）。"""
    calib, bg, marker = files
    rig = _rig(cfg, files)
    assert rig.reload_if_changed() is False
    old_detector = rig.detector
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.02, -0.86, 0.65]), (0.0, -0.05, 0.0))
    CameraModel(intr, rvec, tvec, 0.5, (0.70, 0.90), 0.01).save(calib)
    frames = Latest("camera")
    frames.set(Frame(np.full((720, 1280, 3), 130, np.uint8), 0.0, 1), time.monotonic())
    studio = CalibrationStudio(cfg, load_markers_cfg(), frames, calib, tmp_path / "photos2", 0.0, bg)
    studio.take_background(studio.snapshot())
    assert rig.reload_if_changed() is True
    assert rig.area_size == (0.70, 0.90) and rig.detector is not old_detector
    assert rig.goal_rects() == goal_rects_for_area(cfg.sim, (0.70, 0.90))
    assert rig.reload_if_changed() is False


def test_blob_too_large_saves_images(cfg, files):
    """大きすぎる塊は物でないとして捨て（ほかの物は見続ける）、画像を残す（何が写っていたかを後で見る）。"""
    from toy_sweeper.field.interfaces import FrameUnusable
    rig = _rig(cfg, files)
    rgb = np.full((720, 1280, 3), 120, np.uint8)
    rgb[300:700, 300:1000] = 250                       # 大きな明るい塊（人の手・影の代わり）
    assert rig.detector(Frame(rgb, 0.0, 1), Pose2D(0.0, 0.3, 0.0), cfg.sim.gripper.jaw_open_angle_rad) == []   # 捨てて続ける
    assert len(rig.detector.last_large) == 1
    saved = sorted(p.name for p in rig.error_dir.iterdir())
    assert len(saved) == 2 and saved[0].endswith("_frame.png") and saved[1].endswith("_overlay.jpg")


def test_admin_detection_view(cfg, files, tmp_path):
    """管理の画面の検出の重ね描き（/admin/detect.jpg）は、塊が大きすぎても止めずに理由を出し、画像を保存しない。"""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from toy_sweeper.field.admin_web import admin_router
    from toy_sweeper.field.session import TalkSession
    rig = _rig(cfg, files)
    rgb = np.full((720, 1280, 3), 120, np.uint8)
    rgb[300:700, 300:1000] = 250
    rig.camera.frames.set(Frame(rgb, 0.0, 1), time.monotonic())

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    session = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    app = FastAPI()
    log_path = tmp_path / "run.log"
    log_path.write_text("")
    app.include_router(admin_router(_real_worlds(session, rig), log_path))
    c = TestClient(app)
    r = c.get("/admin/detect.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    info = c.get("/admin/detect_info").json()
    assert "robot_error" in info          # 合成の画像にロボットのマーカーは無い
    assert not rig.error_dir.exists()


def test_ignores_blobs_beyond_out_of_area_margin(cfg, files):
    """作業領域の外へ out_of_area_margin_m（学習で失敗になる距離）より遠くに中心がある塊は、大きくても物と見ない（止めない）。
    2026-10-22、ソファの脚・影が作業領域の外の帯にかかり、大きな塊で止まった。作業領域の中の大きな塊は今までどおり止める。"""
    import dataclasses
    from toy_sweeper.field.interfaces import FrameUnusable
    keep = 0.10   # v31 の walls.out_of_area_margin_m（config/ の既定は 0）
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls,
                                                                                         out_of_area_margin_m=keep)))
    rig = _rig(cfg, files)
    cam = rig.hom
    half = np.array(SIZE) / 2

    def frame_with_blob(cx, cy, size_m, long_m=None):
        rgb = np.full((720, 1280, 3), 120, np.uint8)
        sy = size_m if long_m is None else long_m
        poly = cam.to_px(np.array([[cx - size_m, cy - sy], [cx + size_m, cy - sy], [cx + size_m, cy + sy],
                                   [cx - size_m, cy + sy]]))
        cv2.fillPoly(rgb, [np.round(poly).astype(np.int32)], (250, 250, 250))
        return Frame(rgb, 0.0, 1)

    far = Pose2D(0.0, 0.3, 0.0)   # ロボットは塊から離れた所
    jaw = cfg.sim.gripper.jaw_open_angle_rad
    outside = frame_with_blob(-(half[0] + keep + 0.025), 0.0, 0.02, 0.3)  # 外の帯（緑と水色の線の間）、縁に沿って長く大きい
    assert rig.detector(outside, far, jaw) == []
    assert rig.detector(frame_with_blob(0.0, -0.15, 0.08), far, jaw) == []   # 作業領域の中、大きい（物でないとして捨てる）
    assert len(rig.detector.last_large) == 1
    small_edge = frame_with_blob(-(half[0] + keep - 0.03), 0.0, 0.012)    # 外 7 cm、小さい（物として残す）
    assert len(rig.detector(small_edge, far, jaw)) == 1


def test_only_latest_page_speaks(cfg, files):
    """読み上げるのは、最後に開いた話しかける画面だけ（2026-10-22、前に開いたタブが起動し直したアプリにつながり、二回ずつ読んだ）。"""
    from fastapi.testclient import TestClient
    from toy_sweeper.field.session import TalkSession
    from toy_sweeper.field.web import create_app
    rig = _rig(cfg, files)

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    session = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    c = TestClient(create_app(_real_worlds(session, rig), 16000, 10.0))
    assert c.post("/speaker", json={"page": "a"}).status_code == 200
    assert c.get("/status?page=a").json()["speaker"] is True
    c.post("/speaker", json={"page": "b"})
    assert c.get("/status?page=a").json()["speaker"] is False
    assert c.get("/status?page=b").json()["speaker"] is True


def test_marker_loss_is_recorded(cfg, files):
    """ロボットのマーカーが見えないフレームを、最後に見えた姿勢とともに画像に残してから、そのフレームを飛ばす（MarkerNotVisible）。"""
    from toy_sweeper.field.hardware import MarkerNotVisible
    rig = _rig(cfg, files)
    rig.pose.last_pose = Pose2D(0.1, 0.1, 0.5)
    with pytest.raises(MarkerNotVisible):
        rig.pose.estimate(Frame(np.full((720, 1280, 3), 120, np.uint8), 0.0, 1))
    assert [p.name.endswith("_marker_lost.jpg") for p in rig.error_dir.iterdir()] == [True]


def test_real_identity_tolerance(cfg, files):
    """色の確かめの上限は、実機だけ field.yaml の real の値（Gazebo の 8 では、影で色が変わっただけで止まった）。"""
    f = load_field_cfg(cfg)
    rig = _rig(cfg, files)
    assert rig.f.perception.identity_chroma_tolerance == f.real.identity_chroma_tolerance != f.perception.identity_chroma_tolerance


def test_old_pages_get_no_speech(cfg, files):
    """前の版の画面（page を送らない）には、ロボットの返事を読み上げない種類（said）で渡す（古いタブが二重に読まない）。"""
    from fastapi.testclient import TestClient
    from toy_sweeper.field.session import TalkSession
    from toy_sweeper.field.web import create_app
    rig = _rig(cfg, files)

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    session = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    session._event("robot", "とまるね！")
    c = TestClient(create_app(_real_worlds(session, rig), 16000, 10.0))
    assert [e["kind"] for e in c.get("/status").json()["events"]] == ["said"]
    assert [e["kind"] for e in c.get("/status?page=a").json()["events"]] == ["robot"]


def test_green_and_lime_are_decided_by_hue():
    """VLM が緑・黄緑と答えた物は、カメラで見た色相で決め直す（2026-10-22、実機で黄緑の LEGO を green と答え、「みどり」の指示で
    黄緑を運んだ。2026-10-26、明るさで分けたら奥の暗い黄緑（L* 41）が緑にされた）。"""
    from toy_sweeper.field.vlm_identify import VlmObjectIdentifier
    from toy_sweeper.field.interfaces import TrackView

    class Server:
        def identify(self, png, boxes):
            return {"objects": [{"class": "lego", "color": "green"}, {"class": "lego", "color": "lime"},
                                {"class": "lego", "color": "red"}, {"class": "lego", "color": "green"}], "seconds": 0.1}
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.0, -0.87, 0.66]), (0.0, -0.05, 0.0))
    cam = CameraModel(intr, rvec, tvec, 0.5, SIZE, 0.0)
    ident = VlmObjectIdentifier(Server(), cam, 0.027, green_lime_split_hue_deg=118.0)
    tracks = {1: TrackView(1, Pose2D(0.0, 0.0, 0.0), (0.032, 0.016), (137.0, 143.0, 14.0), 0),   # 明るい（黄緑）
              2: TrackView(2, Pose2D(0.1, 0.0, 0.0), (0.032, 0.016), (41.0, 71.0, 20.0), 0),     # 暗い（緑）
              3: TrackView(3, Pose2D(-0.1, 0.0, 0.0), (0.032, 0.016), (200.0, 30.0, 30.0), 0),
              4: TrackView(4, Pose2D(0.0, 0.1, 0.0), (0.032, 0.016), (103.0, 100.0, 11.0), 0)}   # 奥の暗い黄緑（L* 41）
    out = ident.identify(Frame(np.full((720, 1280, 3), 120, np.uint8), 0.0, 1), tracks)
    assert {i: o.color for i, o in out.items()} == {1: "lime", 2: "green", 3: "red", 4: "lime"}


def test_pick_camera_by_background():
    """カメラの番号は iPhone の連係カメラの有無で入れ替わる（2026-10-22、1 番が C920 から内蔵カメラに変わった）。背景（校正の
    空の床）にいちばん似た画像のカメラを選ぶ。どれも似ていなければ例外。"""
    from toy_sweeper.field.hardware import pick_camera_by_background
    bg = np.zeros((108, 192, 3), np.uint8)
    bg[:, :96] = 200
    frames = {0: np.full((108, 192, 3), 90, np.uint8), 1: bg.copy(), 2: None}   # 2 は開けない
    frames[1][50:60, 50:60] = 0   # 物が少し置いてある
    assert pick_camera_by_background(frames, bg, max_mean_diff=20.0) == 1
    with pytest.raises(RuntimeError, match="背景"):
        pick_camera_by_background({0: frames[0]}, bg, max_mean_diff=20.0)


def test_switch_worlds(cfg, files):
    """話しかける画面で、実機とシミュレーションを切り替える（1 つのアプリ、1 つの画面。2026-10-22）。動いている間は切り替えない。"""
    from fastapi.testclient import TestClient
    from toy_sweeper.field.session import TalkSession
    from toy_sweeper.field.web import World, Worlds, create_app
    rig = _rig(cfg, files)

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    real = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    sim = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    ws = Worlds({"real": World("実機", real, rig.camera.frames, True), "sim": World("シミュ", sim, rig.camera.frames, False)},
                "real")
    c = TestClient(create_app(ws, 16000, 10.0))
    st = c.get("/status?page=a").json()
    assert st["world"] == "real" and st["detection"] is True and [w["label"] for w in st["worlds"]] == ["実機", "シミュ"]
    assert c.post("/world", json={"name": "sim"}).json()["ok"] is True
    assert c.get("/status?page=a").json()["detection"] is False
    sim.busy = True
    assert c.post("/world", json={"name": "real"}).json()["ok"] is False      # 動いている間は切り替えない
    assert c.post("/world", json={"name": "moon"}).status_code == 400


def test_learned_detector_mode(cfg, files, monkeypatch):
    """real.detector が learned なら、塊を学習した検出器（seg_detector、config/seg_detector.yaml のモデル）で作る（2026-10-25、
    学習した検出器を実機でも使えるように）。範囲・ロボットの範囲・大きすぎる塊の記録は背景との差のときと同じ。"""
    import dataclasses
    import toy_sweeper.seg_detector as sd
    from toy_sweeper.field.real_rig import RecordingDetector
    calib, bg, marker = files
    rig = _rig(cfg, files)
    assert load_field_cfg(cfg).real.detector == "background" and type(rig.detector) is RecordingDetector
    cam = rig.hom
    u, v = np.round(cam.to_px(np.array([[0.1, 0.05]]), 0.008)[0]).astype(int)
    loaded = []

    class FakeModel:
        def __init__(self, path, scfg):
            loaded.append(path)

        def __call__(self, rgb):
            return [(0, 0.9, np.ones((14, 24), np.uint8), u - 12, v - 7)]
    monkeypatch.setattr(sd, "SegModel", FakeModel)
    f = load_field_cfg(cfg)
    f = dataclasses.replace(f, real=dataclasses.replace(f.real, detector="learned"))
    rig2 = RealRig(cfg, f, load_perception_cfg(), calib, bg, marker, identify_server=None, camera=OpenCVCamera(f.camera),
                   error_dir=calib.parent.parent / "errors")
    assert isinstance(rig2.detector, RecordingDetector) and loaded and loaded[0].name == "best.onnx"
    frame = Frame(np.full((720, 1280, 3), 120, np.uint8), 0.0, 1)
    dets = rig2.detector(frame, Pose2D(-0.3, -0.3, 0.0), 0.7)
    assert len(dets) == 1 and abs(dets[0].pose.x - 0.1) < 0.01 and abs(dets[0].pose.y - 0.05) < 0.01
    with pytest.raises(ValueError):
        RealRig(cfg, dataclasses.replace(f, real=dataclasses.replace(f.real, detector="yolo")), load_perception_cfg(), calib,
                bg, marker, identify_server=None, camera=OpenCVCamera(f.camera), error_dir=calib.parent.parent / "errors")


def test_pages_link_each_other(cfg, files):
    """話しかける画面・管理・校正の画面は互いにリンクする（2026-10-25）。"""
    from toy_sweeper.field import admin_web, calibration_web, web
    import inspect
    for mod, links in ((web, ["/admin", "/calibrate"]), (admin_web, ['href="/"', "/calibrate"]),
                       (calibration_web, ['href="/"', "/admin"])):
        src = inspect.getsource(mod)
        for link in links:
            assert link in src, (mod.__name__, link)


def test_camera_must_match_background_at_start(cfg, files):
    """カメラが校正の背景と合うかは、指示を受けて動き始めるときに確かめる（起動したまま校正し直して、そのまま実機で動かすため。
    2026-10-25）。"""
    from toy_sweeper.field.interfaces import FieldStop
    rig = _rig(cfg, files)
    with pytest.raises(FieldStop) as e:              # 映像がまだ来ていない（起動の直後）
        rig.check_camera()
    assert e.value.reason == "no_camera"
    rig.camera.frames.set(Frame(np.full((720, 1280, 3), 120, np.uint8), 0.0, 1), time.monotonic())   # 背景と同じ
    rig.check_camera()
    rig.camera.frames.set(Frame(np.full((720, 1280, 3), 250, np.uint8), 0.0, 2), time.monotonic())   # カメラが動いた
    with pytest.raises(FieldStop) as e:
        rig.check_camera()
    assert e.value.reason == "camera_moved"


def test_session_refreshes_calibration_when_idle(cfg, files, tmp_path):
    """画面の重ね描きの前に、動いていなければ校正を読み込み直す（2026-10-25。校正し直した後、メイン画面と管理画面の
    描画が直らなかった: 読み込み直しが次の指示の前だけだった）。動いている間は読み込み直さない（ランタイムが古い校正を持っている）。"""
    from toy_sweeper.field.session import TalkSession
    calib, bg, marker = files
    rig = _rig(cfg, files)

    async def nothing(_):
        return {"ok": False, "reason": "試験"}
    s = TalkSession(cfg, rig, object(), nothing, nothing, ("とまって",), None)
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.02, -0.86, 0.65]), (0.0, -0.05, 0.0))
    CameraModel(intr, rvec, tvec, 0.5, (0.70, 0.90), 0.01).save(calib)
    frames = Latest("camera")
    frames.set(Frame(np.full((720, 1280, 3), 130, np.uint8), 0.0, 1), time.monotonic())
    studio = CalibrationStudio(cfg, load_markers_cfg(), frames, calib, tmp_path / "p", 0.0, bg)
    studio.take_background(studio.snapshot())
    s.busy = True
    s.refresh_calibration()
    assert rig.area_size == SIZE                     # 動いている間は古いまま
    s.busy = False
    s.refresh_calibration()
    assert rig.area_size == (0.70, 0.90)


def test_real_runtime_uses_motion_model(cfg, files):
    """実機のランタイムは、姿勢の先読みに config/field.yaml の real.motion_*（走りの記録から割り出した動き方）を使う。"""
    from toy_sweeper.field.runtime import MotionModel
    f = load_field_cfg(cfg)
    rig = _rig(cfg, files)
    rig.identifier = object()
    rig.check_camera = lambda: None
    rt = rig.make_runtime(policy=None)
    assert rt.motion_model == MotionModel(f.real.motion_delay_s, f.real.motion_lag_s, f.real.motion_gain_linear,
                                          f.real.motion_gain_angular) and f.real.motion_delay_s > 0


def test_real_runtime_shares_online_motion_gain(cfg, files):
    """走りながら割り出す速さの倍率（field.motion_gain）は、指示をまたいで 1 つ（電池の減りは指示をまたぐ）。最初は
    config/field.yaml の real.motion_* の動き方。"""
    from toy_sweeper.field.runtime import MotionModel
    f = load_field_cfg(cfg)
    rig = _rig(cfg, files)
    rig.identifier = object()
    rig.check_camera = lambda: None
    a, b = rig.make_runtime(policy=None), rig.make_runtime(policy=None)
    assert a.motion_gain is b.motion_gain is rig.motion_gain
    assert rig.motion_gain.model() == MotionModel(f.real.motion_delay_s, f.real.motion_lag_s, f.real.motion_gain_linear,
                                                  f.real.motion_gain_angular)
