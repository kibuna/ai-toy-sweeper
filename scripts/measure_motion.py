"""BLE で決まった指令を送りながら、カメラでロボットの姿勢を記録して動きを測る。
ロボットは床（作業領域の中）に置く。カメラはこのスクリプトだけが開く（校正の画面は
止めておく。同じカメラを 2 つのプロセスで開くと、先に開いた側が読めなくなる）。

    # 1. マーカーの向き（ロボットの前とのずれ）: 前進と後退を交互に。data/calibration/robot_marker.yaml に保存する
    uv run python scripts/measure_motion.py yaw
    # 2. まっすぐ: 指令ごとに前進と後退を交互に（作業領域から出ないように）
    uv run python scripts/measure_motion.py straight --motors 20 30 40 --repeat 3
    # 3. その場の旋回: 左回りと右回りを交互に
    uv run python scripts/measure_motion.py spin --motors 30 --repeat 3
    # 4. 片輪の旋回（M,0,N で左回り、M,N,0 で右回り）
    uv run python scripts/measure_motion.py pivot --motors 22 --repeat 3
    # 5. 物を押す: 爪を開いて前進して押す。LEGO は爪を閉じて後退し、連れて戻れるかも見る（ミニカーは閉じない）。
    #    全フレームを data/camera/push_<日時>/ に保存し、物の動きは後で画像から出す
    uv run python scripts/measure_motion.py push --objects lego car --motors 20 --repeat 2

- 1 回ごとに Enter で始める（ほかのキーで止めて終わる）。走る前に、走る先が作業領域の中に収まるかを確かめ、
  収まらなければ置き直しを頼んで待つ。1 回分の解析ができない・動かなかったときは、理由を記録して次へ進む
  （止めるのは BLE・カメラ・マーカーが見えないときだけ）。
- 指令は TTL（0.5 s）で止まらないよう 0.25 s ごとに送り続け、終わりに S。Ctrl-C でも S を送って終わる。
- 出力: dev_log/motion/<日時>/run.log、run_NN.csv（姿勢の列）、report.json（1 回ごとの結果と、指令ごとの平均）。
"""
import argparse
import asyncio
import csv
import json
import logging
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from toy_sweeper.calibration import CameraModel
from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import (ArucoRobotPose, BleRobotLink, OpenCVCamera, RobotMarkerError,
                                        load_robot_marker_yaw, save_robot_marker_yaw)
from toy_sweeper.field.motion_analysis import (NoMotionError, Sample, analyze_spin, analyze_straight,
                                                marker_yaw_offset)
from toy_sweeper.markers import load_markers_cfg

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data" / "calibration" / "camera.yaml"
ROBOT_MARKER = ROOT / "data" / "calibration" / "robot_marker.yaml"
log = logging.getLogger("measure_motion")

# 指令を送る間隔（実機のランタイムの制御の周期と同じ 0.25 s、TTL 0.5 s の半分）
STREAM_PERIOD_S = 0.25
# push の物の名前
OBJECT_NAMES = {"lego": "LEGO 2×4", "car": "ミニカー"}
# 爪を閉じてから戻り始めるまで（開閉は目分量で 0.5 s より速い、組み立て後の確かめ）
GRIP_WAIT_S = 0.5
# マーカーが読めなかったフレームを 1 回の走りで何枚まで保存するか（原因を見るため。ぶれ・隠れ）
MISSED_FRAMES_PER_RUN = 3
# 指令の前と S の後に記録する時間。前: 止まっている姿勢の平均を取る（10 fps で 5 枚）。後: 止まりきるまで（未検証の目安）
PRE_S = 0.5
POST_S = 1.5
# 走る先が作業領域の内側に収まるかの確かめ: 端からの余白（車体の半分の長さ 約 5 cm ＋ 2 cm。作業領域の端はテープで、
# 壁ではない）と、指令 1 あたりの速さの見込み。2026-10-22 の最初の走り（指令 30、1.5 s で 42 cm、0.28 m/s）から、
# 指令に比例とみなした値（回り出す指令 15 があるので、低い指令では多めの見込みになる）
EDGE_MARGIN_M = 0.07
SPEED_PER_MOTOR_GUESS_MPS = 0.28 / 30


class Quit(Exception):
    pass


async def ask(prompt: str) -> str:
    """input を daemon のスレッドで待つ（scripts/check_ble.py と同じ）。"""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def run():
        try:
            r = input(prompt)
        except BaseException as e:   # EOF など（e は except の外で消えるので引数で渡す）
            loop.call_soon_threadsafe(lambda err=e: fut.done() or fut.set_exception(err))
            return
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(r))
    threading.Thread(target=run, daemon=True).start()
    return (await fut).strip().lower()


class Rig:
    def __init__(self, link: BleRobotLink, camera: OpenCVCamera, pose: ArucoRobotPose, half: tuple[float, float],
                 missed_dir: Path):
        self.link, self.camera, self.pose, self.half = link, camera, pose, half
        self.seq = 0
        self.missed = 0
        self.missed_dir = missed_dir   # マーカーが読めなかったフレーム（部屋が写るので data/camera/ に。git に入れない）
        self.saved_this_run = 0

    async def line(self, body: str) -> None:
        self.seq += 1
        await self.link.write_line(f"{body},{self.seq}")

    async def sample(self, frame) -> Sample | None:
        bgr = frame.image[:, :, ::-1].copy()
        try:
            center, my, side = await asyncio.to_thread(self.pose.observe, bgr)
        except RobotMarkerError:
            self.missed += 1
            if self.saved_this_run < MISSED_FRAMES_PER_RUN:
                self.saved_this_run += 1
                self.missed_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.missed_dir / f"run{self.run_no:02d}_missed{self.saved_this_run}.jpg"), bgr)
            return None
        p = self.pose.pose_from(center, my)
        return Sample(frame.stamp, p.x, p.y, p.yaw, my, side)

    async def current(self) -> Sample:
        item = await self.camera.frames.wait_newer(self.camera.frames.seq, 2.0)
        s = await self.sample(item.value)
        if s is None:
            raise RobotMarkerError("ロボットのマーカーが見えない（ロボットを作業領域に置き、マーカーを隠さない）")
        return s

    def check_inside(self, s: Sample, reach_m: float, what: str, direction: int | None) -> None:
        """今の位置から、これから走る向き（direction: 1 前、−1 後ろ）に reach_m 動いても、作業領域の端から EDGE_MARGIN_M
        内側に収まるか。向きが分からない（yaw を測る前、direction None）ときは、どの向きに reach_m 動いても収まるか。"""
        hx, hy = self.half[0] - EDGE_MARGIN_M, self.half[1] - EDGE_MARGIN_M
        angles = [s.yaw + (0.0 if direction > 0 else math.pi)] if direction is not None else [k * math.pi / 2 for k in range(4)]
        for ang in angles:
            x, y = s.x + reach_m * math.cos(ang), s.y + reach_m * math.sin(ang)
            if abs(x) > hx or abs(y) > hy:
                how = ("前に" if direction > 0 else "後ろに") if direction is not None else "どの向きに"
                move = [f"{'右' if s.x < 0 else '左'}へ {abs(s.x) * 100:.0f} cm" if abs(s.x) > 0.02 else "",
                        f"{'奥' if s.y < 0 else '手前'}へ {abs(s.y) * 100:.0f} cm" if abs(s.y) > 0.02 else ""]
                where = (f"{'右' if s.x >= 0 else '左'} {abs(s.x) * 100:.0f} cm・{'奥' if s.y >= 0 else '手前'} "
                         f"{abs(s.y) * 100:.0f} cm")
                raise ValueError(f"{what}: 今の位置（作業領域の中心から {where}）から"
                                 f"{how} {reach_m * 100:.0f} cm 動くと、作業領域の端（{EDGE_MARGIN_M * 100:.0f} cm の余白）を"
                                 f"越える。{'・'.join(x for x in move if x) or '真ん中'}動かして置き直す")

    async def run(self, body: str, duration_s: float, frames_dir: Path | None = None) -> tuple[list[Sample], float, float]:
        """body を duration_s 送り続けて S。前後の PRE_S / POST_S も含めた姿勢の列と、最初の指令・S の時刻。
        frames_dir: 全フレームを JPEG で保存する（物を押したときの物の動きを後で画像から出す。部屋が写るので data/camera/）。"""
        samples: list[Sample] = []
        done = asyncio.Event()
        self.saved_this_run = 0
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)

        async def record():
            seq = self.camera.frames.seq
            while not done.is_set():
                item = await self.camera.frames.wait_newer(seq, 2.0)
                seq = item.seq
                if frames_dir is not None:
                    await asyncio.to_thread(cv2.imwrite, str(frames_dir / f"{item.value.stamp:.4f}.jpg"),
                                            item.value.image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
                s = await self.sample(item.value)
                if s is not None:
                    samples.append(s)

        rec = asyncio.create_task(record())
        try:
            await asyncio.sleep(PRE_S)
            t_cmd = time.monotonic()
            n = 0
            while time.monotonic() - t_cmd < duration_s:
                await self.line(body)
                n += 1
                await asyncio.sleep(max(0.0, t_cmd + n * STREAM_PERIOD_S - time.monotonic()))
            t_stop = time.monotonic()
            await self.line("S")
            await asyncio.sleep(POST_S)
        finally:
            done.set()
            await rec
        return samples, t_cmd, t_stop


def save_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "marker_yaw", "side_m"])
        for s in samples:
            w.writerow([f"{s.t:.4f}", f"{s.x:.5f}", f"{s.y:.5f}", f"{s.yaw:.5f}", f"{s.marker_yaw:.5f}", f"{s.side_m:.5f}"])


async def main_async(a, out: Path) -> int:
    cfg = load_all()
    fcfg, m = load_field_cfg(cfg), load_markers_cfg()
    cam = CameraModel.load(CALIB)
    yaw_offset = 0.0 if a.mode == "yaw" else load_robot_marker_yaw(ROBOT_MARKER)
    camera = OpenCVCamera(fcfg.camera)
    link = BleRobotLink(fcfg.ble, cfg.sim.robot)
    rig = Rig(link, camera, ArucoRobotPose(cam, m, yaw_offset), (cam.work_area_size_m[0] / 2, cam.work_area_size_m[1] / 2),
              ROOT / "data" / "camera" / f"motion_{out.name}")
    log.info("モード %s、作業領域 %.3f × %.3f m、マーカーの向きのずれ %.1f°", a.mode, *cam.work_area_size_m,
             math.degrees(yaw_offset))
    cam_t = asyncio.create_task(camera.run(), name="camera")
    runs, code = [], 0
    yaw_offs: list[float] = []
    try:
        await link.connect()
        await rig.current()
        plan = []   # (指令の名前, 指令の数, 本文, 前後・左右の向き)
        motors = [a.motor] if a.mode == "yaw" else a.motors
        for mot in motors:
            for k in range(a.repeat):
                if a.mode == "spin":
                    plan += [(f"左回り {mot}", mot, f"M,-{mot},{mot}", 1), (f"右回り {mot}", mot, f"M,{mot},-{mot}", -1)]
                elif a.mode == "pivot":
                    plan += [(f"片輪 左回り {mot}", mot, f"M,0,{mot}", 1), (f"片輪 右回り {mot}", mot, f"M,{mot},0", -1)]
                elif a.mode == "push":
                    for obj in a.objects:
                        plan += [(f"押す {obj} {mot}", mot, f"M,{mot},{mot}", 1)]
                else:
                    plan += [(f"前進 {mot}", mot, f"M,{mot},{mot}", 1), (f"後退 {mot}", mot, f"M,-{mot},-{mot}", -1)]
        for i, (name, mot, body, d) in enumerate(plan, 1):
            print(f"\n[{i}/{len(plan)}] {name}: {body} を {a.duration} 秒")
            reach = SPEED_PER_MOTOR_GUESS_MPS * mot * a.duration + 0.05
            yaw_unknown = a.mode == "yaw" and not yaw_offs
            if yaw_unknown:   # 向きがまだ分からない最初の 1 回は、目で確かめてもらう（横まで見ると、走れる位置でも止めてしまう）
                print(f"ロボットの前と後ろに {reach * 100:.0f} cm ずつ空きがあるか、目で確かめてください")
            while not yaw_unknown:   # 端に近ければ、置き直してもらって確かめ直す（止めない）
                s0 = await rig.current()
                try:
                    if a.mode not in ("spin", "pivot"):
                        rig.check_inside(s0, reach, name, d)
                except ValueError as e:
                    print(e)
                    if await ask("置き直したら Enter（ほかのキーで止めて終わる）: ") != "":
                        raise Quit
                    continue
                break
            obj = name.split()[1] if a.mode == "push" else None
            if obj is not None:
                print(f"{OBJECT_NAMES[obj]}をロボットの前 約 5 cm、開いた爪の間に置く（爪はこれから開く）")
                await rig.line("G,1")
            if await ask("Enter で走る（ほかのキーで止めて終わる）: ") != "":
                raise Quit
            missed0 = rig.missed
            rig.run_no = i
            frames_dir = None if obj is None else ROOT / "data" / "camera" / f"push_{out.name}" / f"run_{i:02d}_forward"
            samples, t_cmd, t_stop = await rig.run(body, a.duration, frames_dir)
            save_csv(out / f"run_{i:02d}.csv", samples)
            back = None
            if obj == "lego":   # 挟んで戻る（ミニカーは爪より幅が広く、閉じると詰まって焼損のおそれがあるので閉じない）
                await rig.line("G,0")
                await asyncio.sleep(GRIP_WAIT_S)
                back = await rig.run(f"M,-{mot},-{mot}", a.duration, frames_dir.with_name(f"run_{i:02d}_back"))
                save_csv(out / f"run_{i:02d}_back.csv", back[0])
                await rig.line("G,1")
            r = {"run": i, "name": name, "motor": mot, "direction": d, "frames": len(samples),
                 "marker_missed": rig.missed - missed0}
            try:
                if a.mode == "yaw":   # 向きが分からないので、速さなどは出さない（straight で測る）
                    r["yaw_offset_rad"] = marker_yaw_offset(samples, d)
                    yaw_offs.append(r["yaw_offset_rad"])   # 次の回からは、ここまでの平均の向きで端を確かめる
                    rig.pose.yaw_offset = math.atan2(np.mean(np.sin(yaw_offs)), np.mean(np.cos(yaw_offs)))
                elif a.mode in ("straight", "push"):   # push: 押しているときのロボットの速さ（物の動きは画像から後で）
                    r.update(analyze_straight(samples, t_cmd, t_stop, d))
                    if back is not None:
                        r.update({f"back_{k}": v for k, v in analyze_straight(back[0], back[1], back[2], -1).items()})
                else:   # spin・pivot（pivot は車軸の中点が動くので axle_wander_m は大きい）
                    r.update(analyze_spin(samples, t_cmd, t_stop))
                r["moved"] = True
            except NoMotionError as e:   # 計測の結果として残して次へ
                r.update(moved=False, note=str(e))
                log.warning("%s: %s", name, e)
            except ValueError as e:      # 1 回分の解析ができない（フレームが足りない など）: 理由を残して次へ。
                r.update(moved=None, note=f"解析できない: {e}")   # 姿勢の列（run_NN.csv）は残るので後で解析し直せる
                log.warning("%s: 解析できない（%s）。次へ進む", name, e)
            runs.append(r)
            log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    except Quit:
        log.info("途中で止めた")
        code = 1
    except BaseException:
        log.exception("例外で止めた")
        code = 2
    finally:
        if link.status().connected:
            try:
                await rig.line("S")
            finally:
                await link.close()
        cam_t.cancel()
        try:
            await cam_t
        except asyncio.CancelledError:
            pass

    summary = {}
    keys = {"yaw": ["yaw_offset_rad"],
            "straight": ["speed_mps", "start_delay_s", "stop_distance_m", "yaw_rate_radps", "lateral_m"],
            "spin": ["yaw_rate_radps", "start_delay_s", "axle_wander_m"],
            "pivot": ["yaw_rate_radps", "start_delay_s", "axle_wander_m"],
            "push": ["speed_mps", "start_delay_s", "stop_distance_m", "yaw_rate_radps", "distance_m"]}[a.mode]
    for name in dict.fromkeys(r["name"] for r in runs):
        all_rs = [r for r in runs if r["name"] == name]
        rs = [r for r in all_rs if r["moved"] is True]
        if not rs:
            n_still = sum(r["moved"] is False for r in all_rs)
            summary[name] = {"moved": f"0/{len(all_rs)}", "not_moved": n_still, "not_analyzed": len(all_rs) - n_still}
            log.info("%s: 解析できた回なし（動かなかった %d 回、解析できない %d 回）", name, n_still, len(all_rs) - n_still)
            continue
        summary[name] = {"moved": f"{len(rs)}/{len(all_rs)}",
                         **{k: {"mean": statistics.fmean(r[k] for r in rs),
                                "sd": statistics.stdev(r[k] for r in rs) if len(rs) > 1 else None} for k in keys}}
        log.info("%s（動いた %d/%d 回）: %s", name, len(rs), len(all_rs),
                 {k: round(v["mean"], 4) for k, v in summary[name].items() if isinstance(v, dict)})
    if a.mode == "yaw" and code == 0:
        offs = [r["yaw_offset_rad"] for r in runs if r["moved"] is True]
        if not offs:
            raise RuntimeError("マーカーの向きを測れた回がない")
        mean = math.atan2(np.mean(np.sin(offs)), np.mean(np.cos(offs)))
        spread = max(abs(math.remainder(o - mean, math.tau)) for o in offs)
        if spread > math.radians(a.max_yaw_spread_deg):
            log.error("マーカーの向きのずれが回ごとに %.1f° ばらつく（上限 %.1f°）。保存しない", math.degrees(spread),
                      a.max_yaw_spread_deg)
            code = 3
        else:
            save_robot_marker_yaw(ROBOT_MARKER, mean, {"runs": len(offs), "spread_deg": math.degrees(spread),
                                                       "measured": time.strftime("%Y-%m-%d %H:%M"), "log": str(out)})
            log.info("マーカーの向きのずれ %.2f°（%d 回、最大のばらつき %.2f°）を %s に保存した", math.degrees(mean), len(offs),
                     math.degrees(spread), ROBOT_MARKER)
    (out / "report.json").write_text(json.dumps({"mode": a.mode, "args": vars(a), "runs": runs, "summary": summary,
                                                 "exit_code": code}, ensure_ascii=False, indent=2))
    print(f"記録: {out}")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["yaw", "straight", "spin", "pivot", "push"])
    ap.add_argument("--objects", nargs="+", choices=sorted(OBJECT_NAMES), default=["lego", "car"],
                    help="push: 押す物（lego は挟んで戻るまで、car は押すだけ）")
    ap.add_argument("--motor", type=int, default=20,
                    help="yaw の指令。向きが分かるだけ進めばよいので遅く（指令 30 は 0.28 m/s だった）")
    ap.add_argument("--motors", type=int, nargs="+", default=[30], help="straight / spin の指令")
    ap.add_argument("--repeat", type=int, default=3, help="指令ごとの往復の回数")
    ap.add_argument("--duration", type=float,
                    help="1 回に指令を送り続ける秒数。既定は yaw 0.8 s、straight / spin 1.0 s（速さの当てはめに 0.5 s 分の"
                         "フレームが残り、75 cm の作業領域に収まる長さ）")
    ap.add_argument("--max-yaw-spread-deg", type=float, default=3.0,
                    help="yaw: 回ごとのばらつきの上限。超えたら保存しない。前進と後退で、右への曲がりの分 1〜2° 違う"
                         "（2026-10-22 の実測。平均すると打ち消す）ので、それにマーカーの向きの誤差の分を足した 3°（2° では"
                         "2.0° のばらつきで保存されなかった）")
    a = ap.parse_args()
    if a.duration is None:
        a.duration = 0.8 if a.mode == "yaw" else 1.0
    for mot in [a.motor, *a.motors]:
        if not 0 < mot <= 100:
            ap.error(f"指令は 1〜100: {mot}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "dev_log" / "motion" / stamp
    k = 1
    while out.exists():   # 同じ秒に続けて起動したとき（コマンドをまとめて貼ったとき）
        k += 1
        out = ROOT / "dev_log" / "motion" / f"{stamp}_{k}"
    out.mkdir(parents=True)
    fh = logging.FileHandler(out / "run.log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[sh, fh])
    logging.getLogger("bleak").setLevel(logging.INFO)
    try:
        sys.exit(asyncio.run(main_async(a, out)))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        log.exception("起動の途中で止めた")   # 計測の前の失敗（設定・校正が無い など）もログに残す
        raise


if __name__ == "__main__":
    main()
