"""field.record.draw_overlay の試験（2026-10-25）。"""
import cv2
import numpy as np

from toy_sweeper.calibration import CameraModel, intrinsics_from_fov
from toy_sweeper.field.record import draw_overlay
from toy_sweeper.geometry import Rect


def test_goal_frame_follows_oblique_camera():
    """斜めのカメラでは床の長方形のゴールは台形に写る。枠は 4 隅を移した多角形で描く（2 隅の画素の長方形ではずれた）。"""
    intr = intrinsics_from_fov((640, 360), 1.22)
    pos = np.array([0.0, -0.85, 0.8])
    z = np.array([0.0, 0.0, 0.0]) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    cam = CameraModel(intr, cv2.Rodrigues(R)[0], -R @ pos, 0.0, (0.8, 0.8), 0.0)
    g = Rect((0.0, 0.25), (0.3, 0.15))
    out = draw_overlay(np.zeros((360, 640, 3), np.uint8), None, cam, {"top": g})
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
        u, v = np.round(cam.to_px(np.array([[g.center[0] + sx * 0.15, g.center[1] + sy * 0.075]]))[0]).astype(int)
        assert out[v - 1:v + 2, u - 1:u + 2].any(), (sx, sy)


def test_task_recorder_writes_one_video_per_task(tmp_path):
    """指示を受けてから終わるまで（と終わった後 hold_after_s）を、追跡の重ね描きつきで 1 本の動画に（2026-10-25）。記録していない間のフレームは書かない。"""
    import asyncio
    from types import SimpleNamespace
    from toy_sweeper.field.latest import Latest
    from toy_sweeper.field.record import TaskRecorder
    intr = intrinsics_from_fov((320, 180), 1.22)
    cam = CameraModel(intr, np.array([2.3, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0, (0.8, 0.8), 0.0)
    session = SimpleNamespace(rt=None, rig=SimpleNamespace(hom=cam, goal_rects=lambda: {}))
    frames: Latest = Latest("camera")
    rec = TaskRecorder(frames, tmp_path, fps=10.0, hold_after_s=0.2)
    rec.attach(session)
    t = [0.0]

    def push():
        t[0] += 0.1
        frames.set(SimpleNamespace(image=np.full((180, 320, 3), 120, np.uint8)), t[0])

    async def run():
        task = asyncio.create_task(rec.run())
        await asyncio.sleep(0)
        push()                                    # 記録の前: 書かない
        await asyncio.sleep(0.05)
        path = await rec.start("red lego -> top_right")
        for _ in range(5):
            push()
            await asyncio.sleep(0.03)
        stop = asyncio.create_task(rec.stop("success"))
        for _ in range(10):                       # 終わった後も hold_after_s の間は書く
            push()
            await asyncio.sleep(0.03)
        await stop
        push()                                    # 止めた後: 書かない
        await asyncio.sleep(0.05)
        task.cancel()
        return path

    path = asyncio.run(run())
    assert path.parent == tmp_path and path.name.startswith("001_") and path.suffix == ".mp4"
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 5 + 3 <= n <= 5 + 10, n
    assert rec.n_videos == 1 and not rec.recording
    # 重ね描きのない動画も同じフレーム数で残す（2026-10-28。重ね描きの
    # 印が検出の当て直しに混ざった）。中身はカメラのフレームそのまま（下の文字も無い）
    raw = path.with_name(path.stem + "_raw.mp4")
    cr = cv2.VideoCapture(str(raw))
    assert int(cr.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    ok, f = cr.read()
    assert ok and np.abs(f.astype(int) - 120).max() <= 8          # 圧縮の誤差だけ
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, g = cap.read()
    assert ok and np.abs(g.astype(int) - 120).max() > 100          # 重ね描きのほうは文字がある


def test_task_recorder_close_on_exit(tmp_path):
    """記録中にアプリが終わったら、書いた所までで閉じる（閉じないと mp4 が読めない）。"""
    import asyncio
    from types import SimpleNamespace
    from toy_sweeper.field.latest import Latest
    from toy_sweeper.field.record import TaskRecorder
    intr = intrinsics_from_fov((320, 180), 1.22)
    cam = CameraModel(intr, np.array([2.3, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0, (0.8, 0.8), 0.0)
    frames: Latest = Latest("camera")
    rec = TaskRecorder(frames, tmp_path, fps=10.0, hold_after_s=1.0)
    rec.attach(SimpleNamespace(rt=None, rig=SimpleNamespace(hom=cam, goal_rects=lambda: {})))

    async def run():
        task = asyncio.create_task(rec.run())
        path = await rec.start("x")
        for i in range(4):
            frames.set(SimpleNamespace(image=np.zeros((180, 320, 3), np.uint8)), 0.1 * (i + 1))
            await asyncio.sleep(0.03)
        task.cancel()
        rec.close("app_exit")
        return path
    path = asyncio.run(run())
    assert not rec.recording and int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT)) == 4


def test_attach_task_recorder_shared_by_apps(tmp_path):
    """指示ごとの動画をセッションにつなぐ（管理のアプリ field_app.py と、3090 の talk_sim.py の Gazebo の世界で共通。2026-10-28、
    Gazebo のシミュレーションも録画するため）。record.tasks が偽なら付けない。"""
    import dataclasses
    from toy_sweeper.config import load_all
    from toy_sweeper.field.config import load_field_cfg
    from toy_sweeper.field.latest import Latest
    from toy_sweeper.field.record import TaskRecorder, attach_task_recorder

    class Session:
        recorder = None

        def attach_recorder(self, rec):
            self.recorder = rec
            rec.attach(self)
    f = load_field_cfg(load_all())
    s = Session()
    rec = attach_task_recorder(s, Latest("camera"), f, 10.0, tmp_path / "videos_gazebo")
    assert isinstance(rec, TaskRecorder) and s.recorder is rec and rec.out_dir == tmp_path / "videos_gazebo"
    off = dataclasses.replace(f, record=dataclasses.replace(f.record, tasks=False))
    s2 = Session()
    assert attach_task_recorder(s2, Latest("camera"), off, 10.0, tmp_path / "x") is None and s2.recorder is None
