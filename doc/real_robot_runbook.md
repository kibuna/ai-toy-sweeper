# 実機で片付けを動かす手順

GPU の PC（音声サーバー）と MacBook（カメラ・BLE・話しかける画面・方策）の二台で動かす。環境の作り方は
[setup.md](setup.md)、カメラとマーカーの準備は [camera_setup.md](camera_setup.md)。

## 1. ロボットの準備（初めてのときだけ）

- ロボットは ELECFREAKS Cutebot V3 に micro:bit V2 を挿し、サーボの口 S1 に爪（M5Stack Unit Catch 相当）をつないだもの。
  電池は単 4 を 3 本。
- micro:bit に [microbit/main.ts](../microbit/main.ts) を書き込む（MakeCode の JavaScript の画面に貼る。手順は
  [microbit/README.md](../microbit/README.md)）。
  - プロジェクトの設定で **「No Pairing Required: Anyone can connect via Bluetooth」** を選ぶ（PC はペアリングせずにつなぐ）。
  - 拡張に Cutebot と bluetooth を足す（radio は外れる）。
- 受け付ける指令は 1 行の文字: `M,左,右`（モーターの指令 −100〜100）、`G,1`（爪を開く）・`G,0`（閉じる）、`S`（止まる）。
  最後の指令から 0.5 s 届かないと、ロボットは自分で止まる。BLE が切れても止まる。
- 爪の角度は開く 20・閉じる 60（`OPEN_ANGLE`・`CLOSE_ANGLE`）。**20〜60 の外にしない。** 閉じきれずに押し続けると
  サーボが焼ける（ジーと鳴ったらすぐ開く）。
- 上に ID 4 のマーカーを貼り、向きを測る（[camera_setup.md](camera_setup.md) の 5）。
- 初めてつなぐときは、車輪を浮かせて `uv run python scripts/check_ble.py` で、走る・回る・止まる・開閉と、指令が途切れたとき
  に止まることを確かめてから床に置く。

## 2. GPU の PC: 音声サーバーを起動する

```bash
~/.pixi/bin/pixi run -e vlm python scripts/voice_server.py --host 127.0.0.1 --port 8765
```

- 声の書き起こし（`/transcribe`）、文から指示へ（`/instruction`）、画像の中の物の色と種類の見分け（`/identify`）を受け持つ。
- `--host 127.0.0.1` で家の LAN には開けず、MacBook からは ssh のトンネルでつなぐ。
- 止めるときは起動したプロセスの PID で止める。

## 3. MacBook: 話しかける画面のアプリを起動する

1. 音声サーバーへのトンネルを開いたままにする:
   ```bash
   ssh -N -L 8765:localhost:8765 <ユーザー>@<GPU の PC>
   ```
2. ロボットの電源を入れ、作業領域に置く。電池は新しいものを使う（速さは電池で ±3 割変わる。アプリは走りながら速さを
   測り直すが、減りすぎると回りきれない）。
3. アプリを起動する:
   ```bash
   uv run python scripts/field_app.py --model checkpoints/ppo_v36_s0_x2/final.zip \
       --voice-server http://127.0.0.1:8765 --port 8080 --gazebo-ssh none
   ```
   - 校正・背景・マーカーの向きが揃っていなければ、実機には切り替えられず、2D シミュレーションと校正の画面（カメラが開けたとき）で起動する。[camera_setup.md](camera_setup.md) の 4 で
     揃えてから起動し直す。カメラを動かした・照明が大きく変わったときも撮り直す。
   - カメラを開くのはこのアプリだけにする（`calibrate_app.py`・`measure_motion.py`・`check_detection.py` と同時に動かさない）。

## 4. 使う

| 画面 | 使う人 | できること |
|---|---|---|
| http://localhost:8080/ | 子ども | 録音ボタンを押して話す（「みどりのレゴを ひだりしたに かたづけて」）。返事を読み上げる。「とまって」でいつでも止まる |
| http://localhost:8080/admin | 大人 | 世界の切り替え（実機・2D シミュレーション・Gazebo）、状態、BLE、VLM の入出力、ログ、止めるボタン |
| http://localhost:8080/calibrate | 大人 | カメラの校正と背景の撮り直し |

- 指示は「何を（色・種類、ぜんぶ）」と「どこへ（四隅・辺・中央）」。あいまいな指示（「右に」で右上か右下か分からない）は
  聞き返す。
- 運ぶ物は 2×4 のブロック（6 色ほど）とミニカー。作業領域の外の物は運ばない。
- 止める: 画面の「とまって」、管理の画面の「止める」、停止語。アプリの終了は Ctrl+C（BLE が切れてロボットは止まる）。
- 行動の選び方は `config/field.yaml` の `control.policy_stochastic`（今は false: 決定的）。

## 5. 記録

- `dev_log/field_app/<日時>/`: `run.log`（判断と止まった理由）、`telemetry.jsonl`（送った指令・毎フレームの姿勢・方策の
  入力）、`results.json`（指示ごとの結果）。git には入れない。録音（`audio/`）とやりとり（`conversation_*.jsonl`）は子どもの
  声と言葉を含むので、この PC の外へ出さない。
- `data/camera/<日時>/videos_real/`: 指示ごとの動画。追跡とゴールを重ね描きしたものと、重ね描きなし（`_raw.mp4`）。
  部屋が写るので git に入れない。
- 走りの記録から動き方（遅れ・時定数・速さの倍率）を当てはめ直すには:
  ```bash
  uv run python scripts/fit_motion_from_telemetry.py dev_log/field_app/<日時> [ほかの記録 ...]
  ```

## 6. うまく動かないとき

| 様子 | 見るところ・直し方 |
|---|---|
| 「カメラから読めない」で止まる | カメラを挿し直す。ハブを通さず直接つなぐ。ほかのアプリがカメラを開いていないか |
| ロボットがほとんど動かない・回りきれない | 電池を替える。`run.log` の「先読みの速さの倍率（走りながら割り出した）」が 0.6 を下回っていたら電池切れが近い |
| 物の手前や奥で爪を閉じる | 電池の減りか、マーカーのずれ。マーカーを貼り直したら向きを測り直す |
| 床の影や家具を物と間違える | 物とロボットをどけて背景を撮り直す（/calibrate） |
| 同じ色の物を取り違える・くっついた 2 個が 1 個に見える | 物どうしを少し離して置く |
| ロボットの目印が読めない時間が長い | ロボットのマーカーに照明が反射していないか、カメラから見て隠れていないか |
