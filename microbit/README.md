# micro:bit のプログラム（ロボット側）

`main.ts` は Cutebot V3 に載せた micro:bit V2 のプログラム。PC から Bluetooth（BLE の UART）で届く指令で、車輪と爪のサーボを動かす。

## 書き込み方

1. https://makecode.microbit.org/ で新しいプロジェクトを作る。
2. 拡張に Cutebot（pxt-cutebot）と bluetooth を追加する（bluetooth を足すと radio は外れる）。
3. プロジェクトの設定（歯車 → Project Settings）で「No Pairing Required: Anyone can connect via Bluetooth」を選ぶ
   （PC 側の bleak はペアリングせずにつなぐ）。
4. JavaScript の画面に `main.ts` を貼り、micro:bit に書き込む。

## 指令（改行で終わる文字列）

| 指令 | 意味 |
|---|---|
| `M,左,右,seq` | 左右の車輪の速さ（−100〜100） |
| `G,1,seq` / `G,0,seq` | 爪を開く / 閉じる |
| `S,seq` | 止まる |

最後の指令から 0.5 秒たつと自分で止まる。切断されたときと、読めない指令が来たときも止まる。爪の角度（開く 20、閉じる 60）は
実機で決めた値で、この範囲の外にしない。PC 側は `src/toy_sweeper/field/hardware.py` の `BleRobotLink`。
