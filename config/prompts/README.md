# VLM のプロンプト

| ファイル | 用途 | 根拠・未検証の点 |
|---|---|---|
| `instruction.txt` | 指示文 → 構造化 JSON。`toy_sweeper.task_manager.parse_instruction` で検証する | 出力欄は task_manager の KEYS と一致させた。行き先の言い換え（奥＝top、手前＝bottom）は、カメラ画像の上を部屋の奥にする設置を仮定（設置時に確認）。言い回しの比較は未実施 |
| `object.txt` | 物体の切り出し画像 → 種類と色 | 種類と色の候補は `toy_sweeper.task.COLORS` と物体の種類に一致させた。比較は未実施 |
| `referring_text.txt` | 1 個を指す指示 → 物体 id。盤面の一覧（追跡結果）を文字で渡す LLM 方式 | 2026-10-16 に画像方式と比較 |
| `referring_image.txt` | 同じ。物体に番号を描き込んだ画像を渡す VLM 方式（Set-of-Mark） | 同上 |
| `referring_query.txt` | 1 個を指す指示 → 構造化した条件（種類、色、選び方）。1 個を選ぶのは `task_manager.resolve_selector`（幾何計算） | 2026-10-16 に ID を直接答えさせる 2 方式と比較。例を 2 つ入れた（評価の問題と同じ型の文なので、評価は甘めに出る） |

`{colors}` `{destinations}` `{instruction}` は `toy_sweeper.vlm` が埋める。
