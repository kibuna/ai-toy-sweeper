"""確認の読み上げ。構造化した指示から日本語の文を作り、Speaker で話す。"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Protocol

from ..spaces import OBJECT_TYPE_NAMES
from ..task import COLORS
from ..task_manager import TidyInstruction

log = logging.getLogger(__name__)

REPHRASE = "すみません、その指示は実行できません。言い直してください。"
NOT_HEARD = "聞き取れませんでした。もう一度言ってください。"
TOO_LONG = "長すぎて聞き取れませんでした。短く言い直してください。"
STOPPING = "止まります。"

# 色が 1 つなら形容詞（赤いレゴ）、2 つ以上なら名詞を「と」でつなぐ（黄色と緑のレゴ）。順は COLORS の順
COLOR_ADJ = {"red": "赤い", "blue": "青い", "yellow": "黄色い", "green": "緑の", "white": "白い", "black": "黒い",
             "lime": "黄緑の", "light_blue": "水色の", "pink": "ピンクの", "orange": "オレンジの"}
COLOR_NOUN = {"red": "赤", "blue": "青", "yellow": "黄色", "green": "緑", "white": "白", "black": "黒",
              "lime": "黄緑", "light_blue": "水色", "pink": "ピンク", "orange": "オレンジ"}
CLASS_WORDS = {"lego": "レゴ", "toy_car": "ミニカー"}          # 順は文に並べる順
DEST_WORDS = {"top_left": "左上", "top_right": "右上", "bottom_left": "左下", "bottom_right": "右下",
              "top": "上", "bottom": "下", "left": "左", "right": "右"}

if set(COLOR_ADJ) != set(COLORS) or set(COLOR_NOUN) != set(COLORS):
    raise ValueError(f"確認の文の色の言葉が COLORS と合わない: {COLORS}")
if set(CLASS_WORDS) != set(OBJECT_TYPE_NAMES.values()):
    raise ValueError(f"確認の文の種類の言葉が物体の種類と合わない: {OBJECT_TYPE_NAMES}")


def _words(table: dict, keys) -> list[str]:
    unknown = set(keys) - set(table)
    if unknown:
        raise ValueError(f"確認の文にできない値: {sorted(unknown)}")
    return [w for k, w in table.items() if k in keys]


def confirmation_sentence(ins: TidyInstruction) -> str:
    if ins.destination not in DEST_WORDS:
        raise ValueError(f"確認の文にできない行き先: {ins.destination}")
    if not ins.target_classes:
        raise ValueError("片付ける種類がない")
    colors = [c for c in COLORS if c in ins.target_colors]
    if set(ins.target_colors) - set(colors):
        raise ValueError(f"確認の文にできない色: {ins.target_colors}")
    if len(colors) == 1:
        color = COLOR_ADJ[colors[0]]
    elif colors:
        color = "と".join(COLOR_NOUN[c] for c in colors) + "の"
    else:
        color = ""
    head = "と".join(_words(CLASS_WORDS, ins.excluded_classes)) + "はそのままにして、" if ins.excluded_classes else ""
    return f"{head}{color}{'と'.join(_words(CLASS_WORDS, ins.target_classes))}を{DEST_WORDS[ins.destination]}に運びます"


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class SaySpeaker:
    """macOS の say コマンドで話す（話し終えるまで待つ）。"""

    def __init__(self, voice: str, rate_wpm: int):
        if shutil.which("say") is None:
            raise RuntimeError("say コマンドがない（macOS 以外では --speaker print を使う）")
        self.voice, self.rate_wpm = voice, rate_wpm

    def say(self, text: str) -> None:
        log.info("発話: %s", text)
        subprocess.run(["say", "-v", self.voice, "-r", str(self.rate_wpm), text], check=True)


class PrintSpeaker:
    """話す代わりに標準出力に書く（Linux での動作確認用）。"""

    def say(self, text: str) -> None:
        log.info("発話: %s", text)
        print(f"[発話] {text}", flush=True)


class RecordingSpeaker:
    """テスト用。話した文を記録する。"""

    def __init__(self):
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)
