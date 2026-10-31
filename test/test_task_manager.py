import pytest

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import ObjectType
from toy_sweeper.task_manager import InstructionError, TidyInstruction, parse_instruction, select_targets


@pytest.fixture(scope="module")
def cfg():
    return load_all()


GOOD = {
    "task": "collect",
    "target_classes": ["lego"],
    "target_colors": ["red"],
    "excluded_classes": ["toy_car"],
    "destination": "top_right",
    "constraints": ["do_not_move_excluded_objects"],
    "needs_confirmation": False,
}


def test_parse_good(cfg):
    ins = parse_instruction(GOOD, cfg)
    assert ins == TidyInstruction(("lego",), ("red",), ("toy_car",), "top_right")


@pytest.mark.parametrize("bad", [
    {**GOOD, "destination": "kitchen"},          # 存在しないゴール
    {**GOOD, "target_classes": ["dinosaur"]},     # 未知のクラス
    {**GOOD, "target_colors": ["purple"]},        # 未知の色（ピンクは 2026-10-18 に足した）
    {**GOOD, "needs_confirmation": True},         # 確認が要る → 実行しない
    {**GOOD, "task": "throw"},                    # 未知のタスク
    {k: v for k, v in GOOD.items() if k != "destination"},  # 欠けたキー
    {**GOOD, "extra": 1},                         # 未知のキー
    {**GOOD, "target_classes": []},               # 対象なし
    {**GOOD, "target_classes": ["lego"], "excluded_classes": ["lego"]},  # 矛盾
])
def test_parse_rejects(cfg, bad):
    with pytest.raises(InstructionError):
        parse_instruction(bad, cfg)


def test_select_targets_filters_and_orders(cfg):
    objs = [
        ObjectSpec("a", ObjectType.LEGO, Pose2D(-0.3, -0.3, 0), "red"),
        ObjectSpec("b", ObjectType.LEGO, Pose2D(0.0, 0.1, 0), "red"),
        ObjectSpec("c", ObjectType.LEGO, Pose2D(0.0, 0.0, 0), "blue"),
        ObjectSpec("d", ObjectType.TOY_CAR, Pose2D(0.0, -0.1, 0), "red"),
        ObjectSpec("e", ObjectType.LEGO, Pose2D(0.3, 0.3, 0), "red"),   # すでに top_right ゴールの中
    ]
    ins = parse_instruction(GOOD, cfg)
    t = select_targets(objs, {o.name: o.pose for o in objs}, ins, Pose2D(0.0, 0.0, 0.0), cfg)
    assert t == ["b", "a"]   # 赤い LEGO のうちゴール外のもの、ロボットに近い順


@pytest.mark.parametrize("text,vlm,ok", [
    ("赤いレゴを右上のエリアに片付けて", ("lego",), True),
    ("おもちゃを全部右上に片付けて", ("lego", "toy_car"), True),
    ("ぬいぐるみを右上に片付けて", ("toy_car",), False),        # 種類の根拠がない
    ("キッチンにミニカーを持っていって", ("toy_car",), False),  # 行き先の根拠がない
    ("赤いレゴを右上に、青いレゴを左下に", ("lego",), False),    # 行き先が 2 つ・使われない色
    ("赤いレゴを右に片付けて", ("lego",), False),               # 四隅のどれか分からない
])
def test_check_grounding(cfg, text, vlm, ok):
    from toy_sweeper.task_manager import check_grounding, load_vocabulary
    colors = ("red",) if "赤" in text else ()
    ins = TidyInstruction(vlm, colors, (), "top_right")
    if ok:
        check_grounding(text, ins, load_vocabulary())
    else:
        with pytest.raises(InstructionError):
            check_grounding(text, ins, load_vocabulary())


def test_resolve_selector():
    from toy_sweeper.task_manager import resolve_selector
    objs = [{"id": 0, "class": "lego", "color": "red", "x": -0.2, "y": 0.0},
            {"id": 1, "class": "lego", "color": "red", "x": 0.25, "y": 0.1},
            {"id": 2, "class": "toy_car", "color": "white", "x": -0.15, "y": 0.05},
            {"id": 3, "class": "lego", "color": "red", "x": 0.23, "y": -0.2}]
    q = {"target_class": "lego", "target_color": "red"}
    with pytest.raises(InstructionError):   # 右端の 2 個は 2 cm 差で曖昧
        resolve_selector({**q, "selector": {"type": "side", "side": "right"}}, objs)
    assert resolve_selector({**q, "selector": {"type": "side", "side": "left"}}, objs) == 0
    assert resolve_selector({**q, "selector": {"type": "nearest_to", "ref_class": "toy_car", "ref_color": "white"}}, objs) == 0
    with pytest.raises(InstructionError):   # 3 個あるのに選び方がない
        resolve_selector({**q, "selector": {"type": "none"}}, objs)


def test_normalize_kana():
    """ひらがなの言い方を、VLM に渡す前に漢字・カタカナにする（長い言葉が先。「みぎうえ」は「右上」）。"""
    from toy_sweeper.task_manager import load_vocabulary, normalize_kana
    v = load_vocabulary()
    assert normalize_kana("あかい れごを みぎうえに かたづけて", v) == "赤い レゴを 右上に かたづけて"
    assert normalize_kana("あかと あおの ぶろっく ひだりした", v) == "赤と 青の ブロック 左下"
    assert normalize_kana("うしろの しろい れご", v) == "うしろの 白い レゴ"   # 「うしろ」は変えない


def test_color_only_means_any_class():
    """種類を言わずに色だけ（「赤いの」）なら、どの種類も対象にしてよい。種類を言ったら、その種類だけ。"""
    import pytest
    from toy_sweeper.task_manager import InstructionError, TidyInstruction, check_grounding, load_vocabulary
    v = load_vocabulary()
    both = TidyInstruction(("lego", "toy_car"), ("red",), (), "top_right")
    check_grounding("赤いの ぜんぶ 右上に", both, v)
    with pytest.raises(InstructionError):
        check_grounding("赤いレゴを右上に", both, v)


def test_grounding_matches_longest_word_first():
    """「黄緑」は黄色・緑と数えない。「右上」は「上」「右」と数えない（2026-10-18、色と方向を足したとき）。"""
    import pytest
    from toy_sweeper.task_manager import InstructionError, TidyInstruction, check_grounding, load_vocabulary
    v = load_vocabulary()
    check_grounding("黄緑のレゴを右上に", TidyInstruction(("lego",), ("lime",), (), "top_right"), v)
    check_grounding("水色のレゴを上に", TidyInstruction(("lego",), ("light_blue",), (), "top"), v)
    check_grounding("ピンクと黄色のレゴを右に", TidyInstruction(("lego",), ("yellow", "pink"), (), "right"), v)
    with pytest.raises(InstructionError):   # 黄緑と言ったのに黄色
        check_grounding("黄緑のレゴを右上に", TidyInstruction(("lego",), ("yellow",), (), "top_right"), v)
    with pytest.raises(InstructionError):   # 右上と言ったのに上
        check_grounding("赤いレゴを右上に", TidyInstruction(("lego",), ("red",), (), "top"), v)


def test_select_targets_uses_given_goal(cfg):
    """「もうゴールの中」は、渡したゴール（実機の校正した大きさのゴール、goal_rects_for_area）で決める（2026-10-26、シミュレータの
    決まった大きさのゴールで決めていて、実機の作業領域の大きさでは中・外を取り違えうる）。"""
    from toy_sweeper.geometry import Rect
    objs = [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.2, 0.1, 0), "red")]
    ins = parse_instruction(GOOD, cfg)
    poses = {o.name: o.pose for o in objs}
    assert select_targets(objs, poses, ins, Pose2D(0.0, 0.0, 0.0), cfg) == ["a"]
    assert select_targets(objs, poses, ins, Pose2D(0.0, 0.0, 0.0), cfg, goal=Rect((0.2, 0.1), (0.1, 0.1))) == []
