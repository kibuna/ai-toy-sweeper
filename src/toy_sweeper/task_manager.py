"""タスク管理。

VLM が出す構造化 JSON を厳密に検証し、該当する物体を 1 個ずつ Policy に渡す。
VLM の自由文を直接制御に使わない。曖昧・不正な指示は InstructionError で止める（黙って解釈を補わない）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import AllCfg
from .geometry import Pose2D, box_corners, dist
from .sim2d import ObjectSpec
from .spaces import OBJECT_TYPE_NAMES
from .task import COLORS, goal_rects

KEYS = {"task", "target_classes", "target_colors", "excluded_classes", "destination", "constraints",
        "needs_confirmation"}
KNOWN_CONSTRAINTS = {"do_not_move_excluded_objects"}


class InstructionError(ValueError):
    pass


@dataclass(frozen=True)
class TidyInstruction:
    target_classes: tuple[str, ...]
    target_colors: tuple[str, ...]   # 空なら色を問わない
    excluded_classes: tuple[str, ...]
    destination: str


def parse_instruction(d: dict, cfg: AllCfg) -> TidyInstruction:
    if not isinstance(d, dict) or set(d) != KEYS:
        raise InstructionError(f"キーは {sorted(KEYS)} ちょうど: {sorted(d) if isinstance(d, dict) else d}")
    if d["task"] != "collect":
        raise InstructionError(f"未対応の task: {d['task']}")
    if d["needs_confirmation"] is not False:
        raise InstructionError("needs_confirmation が false でない。人に確認するまで実行しない")
    classes = set(OBJECT_TYPE_NAMES.values())
    for key in ("target_classes", "excluded_classes", "target_colors", "constraints"):
        if not isinstance(d[key], list) or not all(isinstance(x, str) for x in d[key]):
            raise InstructionError(f"{key} は文字列のリスト")
    if not d["target_classes"]:
        raise InstructionError("target_classes が空")
    if unknown := set(d["target_classes"]) - classes:
        raise InstructionError(f"未知の target_classes: {unknown}")
    if unknown := set(d["excluded_classes"]) - classes:
        raise InstructionError(f"未知の excluded_classes: {unknown}")
    if both := set(d["target_classes"]) & set(d["excluded_classes"]):
        raise InstructionError(f"対象と除外の両方にある: {both}")
    if unknown := set(d["target_colors"]) - set(COLORS):
        raise InstructionError(f"未知の色: {unknown}（使える色: {COLORS}）")
    if unknown := set(d["constraints"]) - KNOWN_CONSTRAINTS:
        raise InstructionError(f"未知の constraints: {unknown}")
    if d["destination"] not in cfg.sim.goal_zones:
        raise InstructionError(f"未知の destination: {d['destination']}（ある: {sorted(cfg.sim.goal_zones)}）")
    return TidyInstruction(tuple(d["target_classes"]), tuple(d["target_colors"]), tuple(d["excluded_classes"]),
                           d["destination"])


def matches_instruction(o: ObjectSpec, ins: TidyInstruction) -> bool:
    """物体が指示の種類・色に合うか（ゴールの中かは問わない）。"""
    return OBJECT_TYPE_NAMES[o.type] in ins.target_classes and (not ins.target_colors or o.color in ins.target_colors)


def select_targets(objects: list[ObjectSpec], poses: dict[str, Pose2D], ins: TidyInstruction, robot: Pose2D,
                   cfg: AllCfg, goal=None) -> list[str]:
    """指示に合い、まだゴールに入っていない物体を、ロボットに近い順に返す。goal: 行き先の長方形（実機は校正した作業領域の大きさの
    ゴール、task.goal_rects_for_area）。None はシミュレータの決まった大きさのゴール。"""
    goal = goal_rects(cfg.sim)[ins.destination] if goal is None else goal
    out = []
    for o in objects:
        kind = OBJECT_TYPE_NAMES[o.type]
        if not matches_instruction(o, ins):
            continue
        if goal.contains_all(box_corners(poses[o.name], cfg.sim.objects[kind].size_m[:2])):
            continue
        out.append(o.name)
    return sorted(out, key=lambda n: dist(poses[n].xy, robot.xy))


# ---------- 指示文との突き合わせ（config/vocabulary.yaml） ----------
def load_vocabulary(config_dir=None) -> dict:
    from .config import CONFIG_DIR, load_yaml
    v = load_yaml("vocabulary.yaml", config_dir or CONFIG_DIR)
    if set(v) != {"classes", "all_classes", "colors", "destinations", "kana"}:
        raise InstructionError(f"vocabulary.yaml の項目が違う: {sorted(v)}")
    return v


def normalize_kana(text: str, vocab: dict) -> str:
    """ひらがなの言い方（vocabulary.yaml の kana）を漢字・カタカナの表記に置き換える。長い言葉から先に。"""
    for k in sorted(vocab["kana"], key=len, reverse=True):
        text = text.replace(k, vocab["kana"][k])
    return text


def _mentions(text: str, vocab: dict) -> dict[str, set[str]]:
    """指示文に現れた語彙の値（種類・色・行き先、全種類の語は "all"）。長い語から先に照合し、照合した部分に含まれる
    短い語は数えない（「黄緑」の「黄」「緑」、「右上」の「上」「右」。2026-10-18、色と方向を足したとき）。"""
    t = text.lower()
    entries = [(w.lower(), cat, key) for cat in ("classes", "colors", "destinations")
               for key, words in vocab[cat].items() for w in words]
    entries += [(w.lower(), "all", "all") for w in vocab["all_classes"]]
    taken = [False] * len(t)
    found: dict[str, set[str]] = {"classes": set(), "colors": set(), "destinations": set(), "all": set()}
    for w, cat, key in sorted(entries, key=lambda e: -len(e[0])):
        i = t.find(w)
        while i >= 0:
            if not any(taken[i:i + len(w)]):
                taken[i:i + len(w)] = [True] * len(w)
                found[cat].add(key)
            i = t.find(w, i + 1)
    return found


def check_grounding(text: str, ins: TidyInstruction, vocab: dict) -> None:
    """VLM の出力の各値に指示文の根拠があり、文中の値がすべて使われていることを確かめる。違えば InstructionError。"""
    m = _mentions(text, vocab)
    # 「おもちゃ全部」のように全種類を指す語があるか、種類を言わずに色だけ言った（「赤いの」、子どもの言い方）なら、
    # 対象の種類はどれでも根拠ありとする（2026-10-18）
    all_mentioned = bool(m["all"]) or (not m["classes"] and bool(m["colors"]))
    for c in ins.target_classes + ins.excluded_classes:
        if not (c in m["classes"] or (all_mentioned and c in ins.target_classes)):
            raise InstructionError(f"種類 {c} の根拠が指示文にない: {text}")
    for c in ins.target_colors:
        if c not in m["colors"]:
            raise InstructionError(f"色 {c} の根拠が指示文にない: {text}")
    if ins.destination not in m["destinations"]:
        raise InstructionError(f"行き先 {ins.destination} の根拠が指示文にない: {text}")
    if m["destinations"] != {ins.destination}:
        raise InstructionError(f"指示文の行き先が 1 つでない: {sorted(m['destinations'])}: {text}")
    unused = sorted(m["colors"] - set(ins.target_colors))
    if unused:
        raise InstructionError(f"指示文の色 {unused} が使われていない: {text}")
    if not ins.target_classes and not all_mentioned:
        raise InstructionError(f"片付ける物体の種類がない: {text}")


# ---------- 1 個を指す指示（「右にある赤いレゴ」）の解決 ----------
SIDE_AXIS = {"right": (0, 1.0), "left": (0, -1.0), "far": (1, 1.0), "near": (1, -1.0)}
# いちばん端・いちばん近いものが、2 番目とこれ以上離れていないときは選ばない（曖昧なので聞き返す）。
# 評価データの作り方（scripts/render_vlm_dataset.py の SIDE_MARGIN_M / NEAR_MARGIN_M）より小さく取った。未検証
SIDE_MARGIN_M = 0.05
NEAR_MARGIN_M = 0.02


def resolve_selector(query: dict, objects: list[dict]) -> int:
    """構造化した条件で、物体の一覧（id、class、color、x、y）から 1 個を選ぶ。決まらなければ InstructionError。"""
    if set(query) != {"target_class", "target_color", "selector"}:
        raise InstructionError(f"キーが違う: {sorted(query)}")
    color = query["target_color"]   # None は色の指定なし
    cand = [o for o in objects if o["class"] == query["target_class"] and (color is None or o["color"] == color)]
    if not cand:
        raise InstructionError(f"該当する物体がない: {query}")
    sel = query["selector"]
    if not isinstance(sel, dict) or "type" not in sel:
        raise InstructionError(f"selector が不正: {sel}")
    if sel["type"] == "none":
        if len(cand) != 1:
            raise InstructionError(f"該当が {len(cand)} 個あり、選び方の指定がない")
        return cand[0]["id"]
    if sel["type"] == "side":
        if sel.get("side") not in SIDE_AXIS:
            raise InstructionError(f"未知の side: {sel}")
        axis, sign = SIDE_AXIS[sel["side"]]
        key = (lambda o: sign * (o["x"] if axis == 0 else o["y"]))
        s = sorted(cand, key=key, reverse=True)
        if len(s) >= 2 and key(s[0]) - key(s[1]) < SIDE_MARGIN_M:
            raise InstructionError(f"いちばん{sel['side']}が決まらない（差 {key(s[0]) - key(s[1]):.3f} m）")
        return s[0]["id"]
    if sel["type"] == "nearest_to":
        refs = [o for o in objects if o["class"] == sel.get("ref_class") and o["color"] == sel.get("ref_color")]
        if len(refs) != 1:
            raise InstructionError(f"基準の物体が {len(refs)} 個（1 個でない）: {sel}")
        ref = refs[0]
        d = lambda o: ((o["x"] - ref["x"]) ** 2 + (o["y"] - ref["y"]) ** 2) ** 0.5
        s = sorted((o for o in cand if o["id"] != ref["id"]), key=d)
        if not s:
            raise InstructionError("基準以外に該当がない")
        if len(s) >= 2 and d(s[1]) - d(s[0]) < NEAR_MARGIN_M:
            raise InstructionError("いちばん近いものが決まらない")
        return s[0]["id"]
    raise InstructionError(f"未知の selector: {sel}")
