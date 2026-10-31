// toy-sweeper: PC からの BLE UART の指令で Cutebot を動かす（doc/real_robot_runbook.md の 1）。
// 指令（改行で終わる）: M,左,右,seq（-100〜100）/ G,1,seq（開く）/ G,0,seq（閉じる）/ S,seq（止まる）
// 最後の指令から TTL_MS を過ぎたら止まる（安全停止の 3 層目。config/field.yaml の control.command_ttl_s と同じ値）。
// 切断されたら止まる。読めない指令が来たら止まる（黙って無視しない）。
const TTL_MS = 500
const OPEN_ANGLE = 20    // 組み立て後の確かめで決めた（先の幅 63 mm。0〜10 は開く側の端に当たる）
const CLOSE_ANGLE = 60   // 組み立て後の確かめで決めた（輪ゴムを巻いて LEGO を挟める。70 は詰まって鳴る）
let lastCmd = 0
let moving = false

function stopWheels() {
    cuteBot.stopcar()
    moving = false
}

function bad() {
    stopWheels()
    led.plot(2, 2)    // 真ん中が光ったら、読めない指令が来た
}

bluetooth.startUartService()
bluetooth.onBluetoothConnected(function () {
    basic.showIcon(IconNames.Yes)
})
bluetooth.onBluetoothDisconnected(function () {
    stopWheels()
    basic.showIcon(IconNames.No)
})
bluetooth.onUartDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    const p = bluetooth.uartReadUntil(serial.delimiters(Delimiters.NewLine)).split(",")
    if (p[0] == "M" && p.length == 4) {
        const l = parseInt(p[1])
        const r = parseInt(p[2])
        if (isNaN(l) || isNaN(r) || Math.abs(l) > 100 || Math.abs(r) > 100) {
            bad()
            return
        }
        cuteBot.motors(l, r)
        moving = l != 0 || r != 0
    } else if (p[0] == "G" && p.length == 3 && (p[1] == "0" || p[1] == "1")) {
        cuteBot.setServo(cuteBot.ServoList.S1, p[1] == "1" ? OPEN_ANGLE : CLOSE_ANGLE)
    } else if (p[0] == "S" && p.length == 2) {
        stopWheels()
    } else {
        bad()
        return
    }
    lastCmd = input.runningTime()
})
basic.forever(function () {
    if (moving && input.runningTime() - lastCmd > TTL_MS) {
        stopWheels()
    }
    basic.pause(20)
})
