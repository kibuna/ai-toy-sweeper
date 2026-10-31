"""VLM（Qwen3-VL）による指示の構造化と物体の分類。pixi の vlm 環境で使う。

VLM の自由文を直接使わない。出力は JSON として読み、指示は task_manager.parse_instruction で厳密に検証する。
JSON として読めない、生成が上限で打ち切られた、などは VLMOutputError にする（黙って補わない）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .config import CONFIG_DIR, AllCfg
from .task import COLORS
from .vlm_config import VLMCfg

PROMPT_DIR = CONFIG_DIR / "prompts"
# 行き先の名前と、プロンプトに書く言い換え（config/prompts/README.md）
DESTINATION_WORDS = {"top_left": "左上・左奥", "top_right": "右上・右奥", "bottom_left": "左下・左手前",
                     "bottom_right": "右下・右手前", "top": "上・奥", "bottom": "下・手前", "left": "左", "right": "右"}
# 指示の解析のプロンプトで、色の値に添える日本語（VLM が「黄緑」を lime に当てられるように）
COLOR_WORDS = {"red": "赤", "blue": "青", "yellow": "黄色", "green": "緑", "white": "白", "black": "黒", "lime": "黄緑",
               "light_blue": "水色", "pink": "ピンク", "orange": "オレンジ"}


class VLMOutputError(ValueError):
    pass


@dataclass(frozen=True)
class VLMReply:
    text: str
    n_tokens: int
    seconds: float


def extract_json(text: str, array: bool = False):
    """出力から JSON オブジェクト（array=True なら配列）を 1 つ取り出す。```json の囲みは外す。それ以外の文が混じれば
    VLMOutputError。"""
    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        d = json.loads(t)
    except json.JSONDecodeError as e:
        raise VLMOutputError(f"JSON として読めない: {text!r}") from e
    if array:
        if not isinstance(d, list):
            raise VLMOutputError(f"JSON の配列でない: {text!r}")
        return d
    if not isinstance(d, dict):
        raise VLMOutputError(f"JSON オブジェクトでない: {text!r}")
    return d


def instruction_prompt(instruction: str, cfg: AllCfg) -> str:
    for name in cfg.sim.goal_zones:
        if name not in DESTINATION_WORDS:
            raise ValueError(f"行き先 {name} の言い換えが DESTINATION_WORDS にない")
    dests = ", ".join(f'"{k}"（{DESTINATION_WORDS[k]}）' for k in cfg.sim.goal_zones)
    tpl = (PROMPT_DIR / "instruction.txt").read_text()
    for c in COLORS:
        if c not in COLOR_WORDS:
            raise ValueError(f"色 {c} の日本語が COLOR_WORDS にない")
    colors = ", ".join(f'"{c}"（{COLOR_WORDS[c]}）' for c in COLORS)
    return tpl.replace("{colors}", colors).replace("{destinations}", dests).replace(
        "{instruction}", instruction)


def object_grid_prompt(n: int) -> str:
    colors = ", ".join(f'"{c}"（{COLOR_WORDS[c]}）' for c in COLORS)
    return (PROMPT_DIR / "object_grid.txt").read_text().replace("{colors}", colors).replace("{n}", str(n))


def object_prompt() -> str:
    return (PROMPT_DIR / "object.txt").read_text().replace("{colors}", ", ".join(f'"{c}"' for c in COLORS))


class QwenVL:
    def __init__(self, cfg: VLMCfg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が使えない。pixi の vlm 環境で、GPU のある PC で実行すること")
        self.cfg = cfg
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=getattr(torch, cfg.dtype),
                                                                 device_map="cuda")

    def ask(self, text: str, max_new_tokens: int, image=None) -> VLMReply:
        content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=self.cfg.do_sample)
        self.torch.cuda.synchronize()
        new = out[0, inputs["input_ids"].shape[1]:]
        if len(new) >= max_new_tokens and new[-1].item() not in self._eos_ids():
            raise VLMOutputError(f"生成が上限 {max_new_tokens} トークンで打ち切られた")
        return VLMReply(self.proc.decode(new, skip_special_tokens=True), len(new), time.time() - t0)

    def ask_batch(self, texts: list[str], images: list, max_new_tokens: int) -> list[VLMReply]:
        """複数の（文、画像）を 1 回の generate でまとめて答える（GPU で並べて生成する。左詰めの padding）。
        2026-10-18、物体の同定の時間の大半は 1 文字ずつの生成で、1 個ずつ聞くと物体の数だけかかったため。"""
        if len(texts) != len(images) or not texts:
            raise ValueError("texts と images の数が違うか空")
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": t}]}]
                 for t, im in zip(texts, images)]
        self.proc.tokenizer.padding_side = "left"
        inputs = self.proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", padding=True).to("cuda")
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=self.cfg.do_sample)
        self.torch.cuda.synchronize()
        dt = time.time() - t0
        eos, pad = self._eos_ids(), self.proc.tokenizer.pad_token_id
        replies = []
        for row in out[:, inputs["input_ids"].shape[1]:]:
            toks = [t for t in row.tolist() if t != pad]
            if len(row) >= max_new_tokens and not (set(toks) & eos):
                raise VLMOutputError(f"生成が上限 {max_new_tokens} トークンで打ち切られた")
            replies.append(VLMReply(self.proc.decode(row, skip_special_tokens=True), len(toks), dt))
        return replies

    def _eos_ids(self) -> set[int]:
        e = self.model.generation_config.eos_token_id
        return set(e if isinstance(e, list) else [e])
