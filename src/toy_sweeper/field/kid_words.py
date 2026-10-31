"""話しかけるシミュレーションで、ロボットが子どもに返す言葉。

使う人は 5 歳くらいの子ども（2026-10-18 に決めた）。ひらがなとカタカナだけで短く言う（漢字は読めない前提）。
ブラウザが読み上げるので、文字が読めなくても分かる。理由の詳しい説明（InstructionError など）は画面に出さずログに残す。
"""
from __future__ import annotations

from ..spaces import OBJECT_TYPE_NAMES
from ..task import COLORS
from ..task_manager import TidyInstruction

COLOR_ADJ = {"red": "あかい", "blue": "あおい", "yellow": "きいろい", "green": "みどりの", "white": "しろい", "black": "くろい",
             "lime": "きみどりの", "light_blue": "みずいろの", "pink": "ピンクの", "orange": "オレンジの"}
COLOR_NOUN = {"red": "あか", "blue": "あお", "yellow": "きいろ", "green": "みどり", "white": "しろ", "black": "くろ",
              "lime": "きみどり", "light_blue": "みずいろ", "pink": "ピンク", "orange": "オレンジ"}
CLASS_WORDS = {"lego": "レゴ", "toy_car": "ミニカー"}
DEST_WORDS = {"top_left": "ひだりうえ", "top_right": "みぎうえ", "bottom_left": "ひだりした", "bottom_right": "みぎした",
              "top": "うえ", "bottom": "した", "left": "ひだり", "right": "みぎ"}

if set(COLOR_ADJ) != set(COLORS) or set(COLOR_NOUN) != set(COLORS):
    raise ValueError(f"子ども向けの色の言葉が COLORS と合わない: {COLORS}")
if set(CLASS_WORDS) != set(OBJECT_TYPE_NAMES.values()):
    raise ValueError(f"子ども向けの種類の言葉が物体の種類と合わない: {OBJECT_TYPE_NAMES}")

WAITING = "なにを おかたづけ する？"
THINKING = "かんがえてるよ…"
WORKING = "おかたづけ ちゅう！"
NOT_UNDERSTOOD = "ごめんね、わからなかった。「あかい レゴを みぎうえに」みたいに いってね"
NOT_HEARD = "ごめんね、きこえなかった。もういちど いってね"
BUSY = "いま おかたづけ ちゅう だよ。とめるときは「とまって」って いってね"
STOPPING = "とまるね！"
NOT_MOVING = "いまは うごいてないよ"
STOPPED = "とまったよ"
FAILED = "ごめんね、うまく できなかった。もういちど いってね"
DONE_BUT_LEFT = "おわったけど、まだ のこってる みたい。もういちど いってね"
RESET = "あたらしく ならべたよ！ なにを おかたづけ する？"
CANNOT_RESET = "ごめんね、いまは ならべなおせないよ"


def _words(table: dict, keys) -> list[str]:
    unknown = set(keys) - set(table)
    if unknown:
        raise ValueError(f"子ども向けの文にできない値: {sorted(unknown)}")
    return [w for k, w in table.items() if k in keys]


def _thing(ins: TidyInstruction) -> str:
    """例: 「あかい レゴ」「あかいの」（種類を問わない）「おもちゃ」（色も種類も問わない）"""
    every = set(ins.target_classes) == set(CLASS_WORDS)
    if every and not ins.target_colors:
        return "おもちゃ"
    if every and len(ins.target_colors) == 1:
        return _words(COLOR_ADJ, ins.target_colors)[0].removesuffix("の") + "の"
    colors = _words(COLOR_ADJ, ins.target_colors)
    if len(colors) == 1:
        color = colors[0]
    elif colors:
        color = "と".join(_words(COLOR_NOUN, ins.target_colors)) + "の"
    else:
        color = ""
    things = "と".join(_words(CLASS_WORDS, ins.target_classes))
    return f"{color} {things}" if color else things   # 子ども向けに、ことばの間を空ける（分かち書き）


def confirm(ins: TidyInstruction) -> str:
    """例: 「あかい レゴを みぎうえに はこぶね！」"""
    if ins.destination not in DEST_WORDS:
        raise ValueError(f"子ども向けの文にできない行き先: {ins.destination}")
    if not ins.target_classes:
        raise ValueError("はこぶ種類がない")
    head = "と".join(_words(CLASS_WORDS, ins.excluded_classes)) + "は そのままで、" if ins.excluded_classes else ""
    return f"{head}{_thing(ins)}を {DEST_WORDS[ins.destination]}に はこぶね！"


# 読み上げ用に置き換える言葉。ブラウザの音声合成は、空白の後の「は」を助詞として「わ」と読む
# （「はこんだよ」→「わこんだよ」、2026-10-18）。画面はひらがなのまま、読み上げだけ漢字にする
SPEECH_WORDS = {"はこぶね": "運ぶね", "はこんだよ": "運んだよ", "はこぶ": "運ぶ"}


def for_speech(text: str) -> str:
    """画面の文 → 読み上げの文（言葉の置き換えと、分かち書きの空白を抜く）。"""
    for k in sorted(SPEECH_WORDS, key=len, reverse=True):
        text = text.replace(k, SPEECH_WORDS[k])
    return text.replace(" ", "").replace("\u3000", "")


def done(ins: TidyInstruction, n_given_up: int = 0) -> str:
    """例: 「できたよ！ あかい レゴを ぜんぶ みぎうえに はこんだよ」。見つからずにあきらめた対象があれば（2026-10-23）
    「できたよ！ でも あかい レゴを 1こ みつけられなかった」"""
    if n_given_up:
        return f"できたよ！ でも {_thing(ins)}を {n_given_up}こ みつけられなかった"
    return f"できたよ！ {_thing(ins)}を ぜんぶ {DEST_WORDS[ins.destination]}に はこんだよ"


def already_done(ins: TidyInstruction) -> str:
    """例: 「あかい レゴは もう みぎうえに かたづいてるよ」（指示に合う物がみなゴールの中。2026-10-26）"""
    return f"{_thing(ins)}は もう {DEST_WORDS[ins.destination]}に かたづいてるよ"


# 止まった理由（FieldStop.reason）ごとの言葉（2026-10-22、「うまくできなかった」だけでなく理由を言う）。
# {thing} は指示の物（「みどりの レゴ」）。理由の詳しい説明はログと管理の画面に出す
_FAILED_WORDS = {
    "no_targets": "{thing}が みつからないよ",
    "target_lost": "{thing}を みうしなっちゃった",
    "target_hidden": "{thing}が かくれて みえなくなっちゃった",
    "robot_out_of_area": "ロボットが そとに でちゃった",
    "target_identity": "どれが {thing}か わからなくなっちゃった",
    "identify_failed": "おもちゃを みわけられなかった",
    "start_undetected": "おもちゃを みつけられなかった",
    "no_camera": "カメラが うつらないよ",
    "stale_camera": "カメラが とまっちゃった",
    "no_perception": "カメラで ロボットが みえないよ",
    "stale_perception": "カメラで ロボットが みえなくなっちゃった",
    "stale_observation": "カメラで ロボットが みえなくなっちゃった",
    "task_failed": "カメラの ようすが へんだよ",
    "link_lost": "ロボットと おはなし できなくなっちゃった",
    "command_expired": "ロボットと おはなし できなくなっちゃった",
    "control_overrun": "かんがえるのが まにあわなかった",
    "control_stalled": "かんがえるのが まにあわなかった",
    "timeout": "じかんが かかりすぎちゃった",
    "area_unfit": "あそぶ ばしょの ひろさが あわないよ。おとなの ひとに なおして もらってね",
    "internal_error": "なかで エラーが おきちゃった",
    "camera_moved": "カメラが うごいちゃったみたい。おとなの ひとに なおして もらってね",
}


def failed(reason: str, ins: TidyInstruction) -> str:
    """止まった理由を言う（例: 「ごめんね、みどりの レゴを みうしなっちゃった。もういちど いってね」）。知らない理由は ValueError
    （言葉を足す。test/test_session.py がランタイムの理由を全部確かめる）。"""
    if reason not in _FAILED_WORDS:
        raise ValueError(f"止まった理由 {reason} の子ども向けの言葉が無い")
    return f"ごめんね、{_FAILED_WORDS[reason].format(thing=_thing(ins))}。もういちど いってね"
