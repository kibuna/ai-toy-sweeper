"""GPU 側（RTX 3090）の HTTP サーバー。pixi の vlm 環境で動かす（scripts/voice_server.py）。

- POST /transcribe: 本文は WAV（16 kHz・モノラル・16 bit）。Whisper で書き起こし {"ok", "text" | "reason", "seconds"}
- POST /instruction: 本文は {"text": ...}。VLM で JSON にし、parse_instruction と check_grounding で検証する。
  実行してはいけない指示（VLMOutputError / InstructionError）は HTTP 200 で {"ok": false, "reason", "raw"}
  （ロボットは言い直しを頼む。想定内の結果）。それ以外の例外は 500 のまま返す（隠さない）。
- POST /identify: 本文は {"image_png_b64", "boxes": [[x0, y0, x1, y1], ...]}。枠ごとに切り出して VLM に種類と色を
  聞く（指示を受けたときの物体の同定）。{"objects": [{"class", "color"} | {"error"}], "seconds"}。
  vlm_cfg を渡したときだけある。
- 入力の形の誤り（WAV でない、長すぎる、text が空、枠が画像の外）は 400。

モデルは起動時に 1 回だけ読む。要求はロックで 1 つずつ処理する（GPU のモデルは 1 つ）。
"""
from __future__ import annotations

import dataclasses
import io
import logging
import threading
import time
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from ..config import AllCfg
from ..task_manager import InstructionError, check_grounding, normalize_kana, parse_instruction
from ..vlm import VLMOutputError, extract_json, instruction_prompt
from .audio import VoiceInputError, read_wav
from .config import ServerCfg, VoiceCfg

log = logging.getLogger(__name__)


class ASROutputError(ValueError):
    """書き起こしが使えない（生成が上限で打ち切られた＝繰り返しに陥った、など）。"""


class InstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class IdentifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_png_b64: str
    boxes: list[list[int]]
    mode: str | None = None     # crops / grid（比べるため。None なら vlm.yaml の identify_mode）


OBJECT_CLASSES = ("lego", "toy_car", "other")


def _crops(image_rgb: np.ndarray, boxes: list[list[int]], vlm_cfg) -> list:
    from PIL import Image
    img = Image.fromarray(image_rgb)
    m, s = vlm_cfg.crop_margin_px, vlm_cfg.crop_size_px
    return [img.crop((max(0, x0 - m), max(0, y0 - m), min(img.width, x1 + m), min(img.height, y1 + m))).resize((s, s), Image.LANCZOS)
            for x0, y0, x1, y1 in boxes]


def _check_object(d: dict, raw: str) -> dict:
    """VLM の答えを確かめる。おもちゃ（lego・toy_car）は色も決まった言葉（task.COLORS）でなければ {"error"}。おもちゃでない物
    （other）の色は使わないので確かめず None にする（2026-10-22、実機でソファの脚の影に "brown" と答えて同定ごと止まった）。"""
    from ..task import COLORS
    if not isinstance(d, dict) or d.get("class") not in OBJECT_CLASSES:
        return {"error": f"決まった値でない: {d}", "raw": raw}
    if d["class"] == "other":
        return {"class": "other", "color": None}
    if d.get("color") not in COLORS:
        return {"error": f"決まった値でない: {d}", "raw": raw}
    return {"class": d["class"], "color": d["color"]}


def grid_image(crops: list, cols: int):
    """切り出しを番号（左上、1 から）つきのマスに並べた 1 枚の画像。"""
    from PIL import Image, ImageDraw
    s = crops[0].width
    rows = (len(crops) + cols - 1) // cols
    g = Image.new("RGB", (cols * s, rows * s), (255, 255, 255))
    d = ImageDraw.Draw(g)
    for k, c in enumerate(crops):
        x, y = (k % cols) * s, (k // cols) * s
        g.paste(c, (x, y))
        d.rectangle((x, y, x + s - 1, y + s - 1), outline=(0, 0, 0), width=2)
        d.rectangle((x + 2, y + 2, x + 34, y + 30), fill=(255, 255, 255))
        d.text((x + 6, y + 4), str(k + 1), fill=(0, 0, 0), font_size=22)
    return g


def identify_objects_grid(vlm, crops: list, vlm_cfg) -> list[dict]:
    """切り出しを 1 枚のマスの画像にまとめて、全部を 1 回で聞く（grid_max 個ずつ）。番号がそろわなければ全部 {"error"}。"""
    import math
    from ..vlm import object_grid_prompt
    out = []
    for start in range(0, len(crops), vlm_cfg.grid_max):
        part = crops[start:start + vlm_cfg.grid_max]
        cols = math.ceil(math.sqrt(len(part)))
        raw = vlm.ask(object_grid_prompt(len(part)), vlm_cfg.max_new_tokens_grid, grid_image(part, cols)).text
        try:
            arr = extract_json(raw, array=True)
        except VLMOutputError as e:
            out += [{"error": f"JSON でない: {e}", "raw": raw}] * len(part)
            continue
        ids = [a.get("id") for a in arr if isinstance(a, dict)]
        if sorted(ids) != list(range(1, len(part) + 1)):
            out += [{"error": f"番号が 1〜{len(part)} でない: {ids}", "raw": raw}] * len(part)
            continue
        by_id = {a["id"]: a for a in arr}
        out += [_check_object(by_id[k + 1], raw) for k in range(len(part))]
    return out


def identify_objects(vlm, image_rgb: np.ndarray, boxes: list[list[int]], vlm_cfg, mode: str | None = None) -> list[dict]:
    """枠ごとに切り出し（余白 crop_margin_px、一辺 crop_size_px に拡大）、VLM に種類と色を聞く。mode（None なら
    vlm_cfg.identify_mode）: crops は 1 個ずつ（scripts/eval_vlm.py の物体の分類と同じ）、grid はまとめて 1 回。
    決まった値でない答えは {"error"}（黙って使わない）。"""
    from ..vlm import object_prompt
    mode = mode or vlm_cfg.identify_mode
    crops = _crops(image_rgb, boxes, vlm_cfg)
    if mode == "grid":
        return identify_objects_grid(vlm, crops, vlm_cfg)
    prompt = object_prompt()
    if mode == "batch":
        out = []
        for start in range(0, len(crops), vlm_cfg.batch_size):
            part = crops[start:start + vlm_cfg.batch_size]
            for r in vlm.ask_batch([prompt] * len(part), part, vlm_cfg.max_new_tokens_object):
                try:
                    out.append(_check_object(extract_json(r.text), r.text))
                except VLMOutputError as e:
                    out.append({"error": f"JSON でない: {e}", "raw": r.text})
        return out
    if mode != "crops":
        raise ValueError(f"identify_mode は crops、grid、batch のどれか: {mode}")
    out = []
    for crop in crops:
        raw = vlm.ask(prompt, vlm_cfg.max_new_tokens_object, crop).text
        try:
            d = extract_json(raw)
        except VLMOutputError as e:
            out.append({"error": f"JSON でない: {e}", "raw": raw})
            continue
        out.append(_check_object(d, raw))
    return out


def structure_instruction(text: str, vlm, cfg: AllCfg, vocab: dict, max_new_tokens: int) -> dict:
    """指示文 → VLM の JSON → parse_instruction → check_grounding。/instruction の応答の形の dict を返す。
    実行してはいけない指示は ok: false（VLMOutputError / InstructionError だけ。他の例外はそのまま上げる）。"""
    raw = None
    t0 = time.perf_counter()
    said = text
    text = normalize_kana(text, vocab)   # ひらがなの言い方を置き換えてから VLM に渡し、根拠もその文で調べる
    if text != said:
        log.info("ひらがなを置き換えた: 「%s」→「%s」", said, text)
    try:
        raw = vlm.ask(instruction_prompt(text, cfg), max_new_tokens).text
        ins = parse_instruction(extract_json(raw), cfg)
        check_grounding(text, ins, vocab)
    except (VLMOutputError, InstructionError) as e:
        log.info("実行しない指示「%s」: %s: %s", text, type(e).__name__, e)
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "raw": raw}
    dt = time.perf_counter() - t0
    d = {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(ins).items()}
    log.info("指示「%s」→ %s（%.2f s）", text, d, dt)
    return {"ok": True, "instruction": d, "raw": raw, "seconds": dt}


def create_app(asr: Callable[[np.ndarray], str], vlm, vcfg: VoiceCfg, cfg: AllCfg, vocab: dict,
               max_new_tokens: int, vlm_cfg=None) -> FastAPI:
    """asr: int16 の音声 → 書き起こし（使えなければ ASROutputError）。vlm: QwenVL と同じ ask を持つもの。"""
    app = FastAPI(title="toy-sweeper voice")
    lock = threading.Lock()
    sr = vcfg.audio.sample_rate_hz

    def transcribe(audio: np.ndarray) -> dict:
        with lock:
            t0 = time.perf_counter()
            try:
                text = asr(audio)
            except ASROutputError as e:
                log.warning("書き起こせなかった: %s", e)
                return {"ok": False, "reason": f"ASROutputError: {e}", "seconds": time.perf_counter() - t0}
            dt = time.perf_counter() - t0
        log.info("書き起こし %.2f s（音声 %.2f s）: %s", dt, len(audio) / sr, text)
        return {"ok": True, "text": text, "seconds": dt}

    def instruction(text: str) -> dict:
        with lock:
            return structure_instruction(text, vlm, cfg, vocab, max_new_tokens)

    @app.post("/transcribe")
    async def post_transcribe(request: Request):
        body = await request.body()
        try:
            audio = read_wav(io.BytesIO(body), sr)
        except VoiceInputError as e:
            raise HTTPException(400, str(e)) from e
        if len(audio) == 0:
            raise HTTPException(400, "音声が空")
        if len(audio) / sr > vcfg.server.max_audio_s:
            raise HTTPException(400, f"音声が長すぎる: {len(audio) / sr:.1f} s > {vcfg.server.max_audio_s} s")
        return await run_in_threadpool(transcribe, audio)

    @app.post("/instruction")
    def post_instruction(req: InstructionRequest):
        if not req.text.strip():
            raise HTTPException(400, "text が空")
        return instruction(req.text)

    if vlm_cfg is not None:
        @app.post("/identify")
        def post_identify(req: IdentifyRequest):
            import base64
            import cv2
            buf = np.frombuffer(base64.b64decode(req.image_png_b64), np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(400, "画像を読めない")
            h, w = bgr.shape[:2]
            for b in req.boxes:
                if len(b) != 4 or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                    raise HTTPException(400, f"枠が画像（{w}x{h}）の外か不正: {b}")
            t0 = time.perf_counter()
            with lock:
                objs = identify_objects(vlm, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), req.boxes, vlm_cfg, req.mode)
            dt = time.perf_counter() - t0
            log.info("同定 %d 個（%.2f s）: %s", len(objs), dt, objs)
            return {"objects": objs, "seconds": dt}

    return app


class WhisperASR:
    """transformers の Whisper（GPU）。言語と作業（書き起こし）は明示する（自動判定・翻訳をさせない）。"""

    def __init__(self, cfg: ServerCfg, language: str, sample_rate_hz: int):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が使えない。pixi の vlm 環境で、GPU のある PC で実行すること")
        self.torch, self.cfg, self.language, self.sr = torch, cfg, language, sample_rate_hz
        self.dtype = getattr(torch, cfg.asr_dtype)
        self.proc = AutoProcessor.from_pretrained(cfg.asr_model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.asr_model_id, dtype=self.dtype).to("cuda").eval()
        # 生成の設定はモデルの generation_config に書く。generate に max_new_tokens を渡すと、Hub の設定の
        # max_length（448）と二重指定の警告が出るため（transformers 5.20）。Whisper の generate では max_length は
        # 先頭の指定トークン（言語・作業）を除いた生成の長さとして扱われる（2026-10-16 に 5 と 200 で確認）。
        # なお transformers 内部の呼び出し（generation_whisper.py の suppress_tokens の受け渡し）が出す
        # 「generation_config と generation-related arguments を一緒に渡すのは非推奨」の警告は、こちらでは消せない
        gc = self.model.generation_config
        gc.max_length = cfg.asr_max_new_tokens
        gc.num_beams = cfg.asr_num_beams
        # 出てきそうな言葉（config/voice.yaml の asr_prompt）。空なら渡さない
        self.prompt_ids = (self.proc.get_prompt_ids(cfg.asr_prompt, return_tensors="pt").to("cuda")
                           if cfg.asr_prompt else None)

    def __call__(self, audio: np.ndarray) -> str:
        feats = self.proc(audio.astype(np.float32) / 32768.0, sampling_rate=self.sr, return_tensors="pt")
        x = feats.input_features.to("cuda", self.dtype)
        with self.torch.inference_mode():
            kw = {} if self.prompt_ids is None else {"prompt_ids": self.prompt_ids}
            out = self.model.generate(x, language=self.language, task="transcribe", **kw)
        # transformers 5 の Whisper は、先頭の指定トークンと末尾の EOS を除いた列を返す（2026-10-16 に確認）。
        # なので長さが上限に達していれば、EOS の前に打ち切られたとみなす
        if out.shape[1] >= self.cfg.asr_max_new_tokens:
            raise ASROutputError(f"生成が上限 {self.cfg.asr_max_new_tokens} トークンで打ち切られた")
        # 日本語は空白で区切らないので、BPE の空白の後処理（clean_up_tokenization_spaces）はしない
        return self.proc.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
