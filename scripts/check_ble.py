"""MacBook から BLE で micro:bit（Cutebot）を動かして確かめる（doc/real_robot_runbook.md の 1）。

    uv run python scripts/check_ble.py
    uv run python scripts/check_ble.py --from 9   # 9 番目（切断）から

- 先に micro:bit へ microbit/main.ts を入れ、**車輪を浮かせて**おく。
- 1 つずつ説明を出す。Enter で実行、Enter 以外（何か打って Enter）で S を送って終わる。実行のあと、見たとおりだったかを
  y / n で答える（n のときはメモも書ける）。Ctrl-C でも S を送って終わる。
- 走らせる指令は TTL（micro:bit 側 0.5 s）で止まらないよう、STREAM_PERIOD_S ごとに送り続ける。
- 出力: dev_log/ble/<日時>/run.log、report.json（各段階の答え、書き込み 1 回の時間の分布）。
"""
import argparse
import asyncio
import json
import logging
import statistics
import sys
import threading
import time
from pathlib import Path

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import BleRobotLink

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("check_ble")

# 指令 30（BLE の確かめで決めた確かめの値。床の上では 0.28 m/s、動きの計測）。組み立て後の確かめで 15 から回った
DRIVE = 30
# 送り続ける間隔。実機のランタイムの制御の周期（config/field.yaml control）と同じ 0.25 s。TTL 0.5 s の半分
STREAM_PERIOD_S = 0.25
# 走る・回る段階の長さ。目で向きを見分けられる長さ（未検証の目安）
MOVE_S = 2.0
# 長く送り続ける段階の長さ（手順書の「2 分間」）
LONG_S = 120.0
# 書き込み 1 回の時間の見積もり（config/field.yaml の command_latency の見積もり 50 ms）と比べる
LATENCY_ESTIMATE_S = 0.050


class Quit(Exception):
    pass


class Checker:
    def __init__(self, link: BleRobotLink):
        self.link = link
        self.seq = 0
        self.write_s: list[float] = []
        self.results: list[dict] = []

    async def line(self, body: str) -> None:
        self.seq += 1
        text = f"{body},{self.seq}"
        dt = await self.link.write_line(text)
        self.write_s.append(dt)
        log.debug("送った %s（%.1f ms）", text, dt * 1000)

    async def stream(self, body: str, duration_s: float) -> int:
        """body を STREAM_PERIOD_S ごとに duration_s のあいだ送る。送った回数を返す。"""
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < duration_s:
            await self.line(body)
            n += 1
            await asyncio.sleep(max(0.0, t0 + n * STREAM_PERIOD_S - time.monotonic()))
        return n

    async def drive_then_stop(self, body: str, duration_s: float) -> dict:
        n = await self.stream(body, duration_s)
        await self.line("S")
        return {"sent": n}

    async def step(self, name: str, explain: str, watch: str, action) -> None:
        print(f"\n=== {name} ===\n{explain}\n見ること: {watch}")
        if await ask("Enter で実行（ほかのキーで止めて終わる）: ") != "":
            raise Quit
        detail = await action()
        ans = ""
        while ans not in ("y", "n"):
            ans = await ask("見たとおりだった？ y / n: ")
        note = await ask("メモ（無ければ Enter）: ") if ans == "n" else ""
        self.results.append({"step": name, "ok": ans == "y", "note": note, "detail": detail})
        log.info("%s: %s %s %s", name, "OK" if ans == "y" else "NG", note, detail or "")


async def ask(prompt: str) -> str:
    """input を daemon のスレッドで待つ（Ctrl-C のとき、入力待ちのスレッドで終わりが止まらないように）。"""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def run():
        try:
            r = input(prompt)
        except BaseException as e:   # EOF など。待っている側で例外にする（e は except の外で消えるので引数で渡す）
            loop.call_soon_threadsafe(lambda err=e: fut.done() or fut.set_exception(err))
            return
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(r))
    threading.Thread(target=run, daemon=True).start()
    return (await fut).strip().lower()


async def main(start: int) -> int:
    out_dir = ROOT / "dev_log" / "ble" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    fh = logging.FileHandler(out_dir / "run.log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[sh, fh])
    logging.getLogger("bleak").setLevel(logging.INFO)

    cfg = load_all()
    link = BleRobotLink(load_field_cfg(cfg).ble, cfg.sim.robot)
    c = Checker(link)
    print("車輪を浮かせ、車体の電源を入れ、micro:bit に microbit/main.ts が入っていることを確かめてください。")
    print("macOS で初めてのときは、ターミナルに Bluetooth の許可を求めるダイアログが出ます（許可する）。")

    async def bad_command():
        n = await c.stream(f"M,{DRIVE},{DRIVE}", 1.0)
        await c.line("X")
        return {"sent_before_x": n}

    async def disconnect():
        n = await c.stream(f"M,{DRIVE},{DRIVE}", 1.0)
        await link.close()
        return {"sent_before_disconnect": n}

    steps = [
        ("止まる（S）", "S を 1 回送る。", "車輪が止まったまま。micro:bit の LED は接続のチェックの印のまま",
         lambda: c.line("S")),
        ("前進（M,30,30）", f"M,{DRIVE},{DRIVE} を {MOVE_S:.0f} 秒送り続けて、S。",
         "両輪が前進の向きに回り、S で止まる", lambda: c.drive_then_stop(f"M,{DRIVE},{DRIVE}", MOVE_S)),
        ("左回り（M,-30,30）", f"M,-{DRIVE},{DRIVE} を {MOVE_S:.0f} 秒送り続けて、S。",
         "左の車輪が後ろ、右の車輪が前に回る（床ならその場で左に回る向き）",
         lambda: c.drive_then_stop(f"M,-{DRIVE},{DRIVE}", MOVE_S)),
        ("爪を開く（G,1）", "G,1 を送る。", "爪が開く（角度 20）", lambda: c.line("G,1")),
        ("爪を閉じる（G,0）", "G,0 を送る。", "爪が閉じる（角度 60）。ジーと鳴らない", lambda: c.line("G,0")),
        ("TTL（1 回だけ M）", f"M,{DRIVE},{DRIVE} を 1 回だけ送る。続けて送らない。",
         "回り出して、0.5 秒ほどで自分で止まる", lambda: c.line(f"M,{DRIVE},{DRIVE}")),
        ("読めない指令（X）", f"M,{DRIVE},{DRIVE} を 1 秒送り続けたあと、読めない指令 X を送る。",
         "X で止まり、LED の真ん中の点が光る", bad_command),
        ("長く送り続ける", f"M,{DRIVE},{DRIVE} を {LONG_S:.0f} 秒送り続ける（{STREAM_PERIOD_S} 秒ごと）。",
         "途中で止まったり、つっかえたりせずに回り続ける。終わったら止まる",
         lambda: c.drive_then_stop(f"M,{DRIVE},{DRIVE}", LONG_S)),
        ("切断", f"M,{DRIVE},{DRIVE} を 1 秒送り続けたあと、PC 側から接続を切る。",
         "切れたところで止まり、LED が × になる", disconnect),
    ]

    if not 1 <= start <= len(steps):
        raise ValueError(f"--from は 1〜{len(steps)}: {start}")
    code = 0
    try:
        await link.connect()
        for name, explain, watch, action in steps[start - 1:]:
            await c.step(name, explain, watch, action)
            if link.status().connected:
                await c.line("S")   # 次の段階を止まった状態から始める
    except Quit:
        log.info("途中で止めた")
        code = 1
    except BaseException:
        log.exception("例外で止めた")
        code = 2
    finally:
        if link.status().connected:
            try:
                await c.line("S")
            finally:
                await link.close()

    ws = sorted(c.write_s)
    lat = None
    if ws:
        lat = {"n": len(ws), "median_ms": statistics.median(ws) * 1000, "p95_ms": ws[int(0.95 * (len(ws) - 1))] * 1000,
               "max_ms": ws[-1] * 1000, "estimate_ms": LATENCY_ESTIMATE_S * 1000}
        log.info("書き込み 1 回の時間: %d 回、中央値 %.1f ms、95%% %.1f ms、最大 %.1f ms（見積もり %.0f ms）",
                 lat["n"], lat["median_ms"], lat["p95_ms"], lat["max_ms"], lat["estimate_ms"])
    report = {"results": c.results, "write_latency": lat, "drive": DRIVE, "stream_period_s": STREAM_PERIOD_S,
              "exit_code": code}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for r in c.results:
        print(f"{'OK' if r['ok'] else 'NG'}  {r['step']}  {r['note']}")
    print(f"記録: {out_dir}")
    return code


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--from", dest="start", type=int, default=1, help="何番目の段階から始めるか（1 から）")
        sys.exit(asyncio.run(main(ap.parse_args().start)))
    except KeyboardInterrupt:
        sys.exit(130)
