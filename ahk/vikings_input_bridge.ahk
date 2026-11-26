#Requires AutoHotkey v2.0
#SingleInstance Off
Persistent
SetWorkingDir A_ScriptDir
CoordMode "Mouse", "Screen"
SendMode "Event"
SetKeyDelay 15, 25
SetMouseDelay 8
SetControlDelay -1
SetWinDelay -1

; =============================================================================
; Vikings Bot v2 - Persistent AutoHotkey v2 Input Bridge (stdin / stdout pipe)
; =============================================================================
; Protocol (1 line per command from Python -> stdin, 1 line reply -> stdout):
;   PING
;   CLICK|<mode>|<hwnd>|<x>|<y>|<button>|<hold_ms>
;   MOVE|<mode>|<hwnd>|<x>|<y>
;   DRAG_MMB|<mode>|<hwnd>|<sx>|<sy>|<dx>|<dy>|<steps>
;   KEY_TAP|<mode>|<hwnd>|<key>|<hold_ms>
;   KEY_DOWN|<mode>|<hwnd>|<key>
;   KEY_UP|<mode>|<hwnd>|<key>
;   CHAT_CMD|<mode>|<hwnd>|<text>
;   RELEASE_ALL|<hwnd>
;   QUIT
;
; Modes:
;   vm_foreground   -> Exact absolute screen coordinates (VMware / Guest AFK)
;   host_background -> Win32 ControlClick / ControlSend to target HWND (no focus)
; =============================================================================

global gHeldKeys := Map()
global gHeldButtons := Map()
global gStdIn := FileOpen("*", "r")
global gStdOut := FileOpen("*", "w")

OnExit(CleanupAllInputs)

WriteReply("READY|AHK_V2")

while true {
    if gStdIn.AtEOF {
        Sleep 10
        break
    }
    line := ""
    try {
        line := Trim(gStdIn.ReadLine(), "`r`n `t")
    } catch {
        break
    }
    if (line = "") {
        continue
    }
    if (line = "QUIT") {
        CleanupAllInputs()
        WriteReply("OK|QUIT")
        break
    }
    try {
        resp := HandleCommand(line)
        WriteReply(resp)
    } catch as err {
        WriteReply("ERR|" . StrReplace(err.Message, "`n", " "))
    }
}
ExitApp 0

WriteReply(msg) {
    global gStdOut
    gStdOut.WriteLine(msg)
    gStdOut.Read(0) ; flush stdout immediately
}

MapKeyName(k) {
    lk := StrLower(Trim(k))
    switch lk {
        case "esc", "escape": return "Escape"
        case "enter", "return": return "Enter"
        case "space": return "Space"
        case "tab": return "Tab"
        case "shift": return "LShift"
        case "ctrl": return "LCtrl"
        case "alt": return "LAlt"
        case "slash", "/": return "/"
        default: return lk
    }
}

MapButtonName(btn) {
    lb := StrLower(Trim(btn))
    switch lb {
        case "right", "r": return "Right"
        case "middle", "m", "mmb": return "Middle"
        default: return "Left"
    }
}

HandleCommand(line) {
    parts := StrSplit(line, "|")
    cmd := StrUpper(parts[1])

    if (cmd = "PING") {
        return "PONG"
    }

    if (cmd = "RELEASE_ALL") {
        hwnd := (parts.Length >= 2) ? Integer(parts[2]) : 0
        CleanupAllInputs(hwnd)
        return "OK|RELEASE_ALL"
    }

    if (cmd = "MOVE" && parts.Length >= 5) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        x := Integer(parts[4])
        y := Integer(parts[5])
        if (mode = "host_background" && hwnd > 0) {
            lParam := ((y & 0xFFFF) << 16) | (x & 0xFFFF)
            DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0200, "Ptr", 0, "Ptr", lParam)
        } else {
            DllCall("SetCursorPos", "Int", x, "Int", y)
        }
        return "OK|MOVE"
    }

    if (cmd = "CLICK" && parts.Length >= 7) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        x := Integer(parts[4])
        y := Integer(parts[5])
        btn := MapButtonName(parts[6])
        holdMs := Max(10, Integer(parts[7]))

        if (mode = "host_background" && hwnd > 0) {
            try {
                ControlClick("x" . x . " y" . y, "ahk_id " . hwnd, , btn, 1, "NA Pos")
            } catch {
                ; Fallback via WM_LBUTTONDOWN/UP when ControlClick cannot resolve control
                lParam := ((y & 0xFFFF) << 16) | (x & 0xFFFF)
                downMsg := (btn = "Right") ? 0x0204 : ((btn = "Middle") ? 0x0207 : 0x0201)
                upMsg := (btn = "Right") ? 0x0205 : ((btn = "Middle") ? 0x0208 : 0x0202)
                wParam := (btn = "Right") ? 0x0002 : ((btn = "Middle") ? 0x0010 : 0x0001)
                DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0200, "Ptr", 0, "Ptr", lParam)
                DllCall("PostMessageW", "Ptr", hwnd, "UInt", downMsg, "Ptr", wParam, "Ptr", lParam)
                Sleep holdMs
                DllCall("PostMessageW", "Ptr", hwnd, "UInt", upMsg, "Ptr", 0, "Ptr", lParam)
            }
        } else {
            DllCall("SetCursorPos", "Int", x, "Int", y)
            Sleep 12
            Click(x . " " . y . " " . btn . " Down")
            Sleep holdMs
            Click(btn . " Up")
        }
        return "OK|CLICK"
    }

    if (cmd = "DRAG_MMB" && parts.Length >= 8) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        sx := Integer(parts[4])
        sy := Integer(parts[5])
        dx := Integer(parts[6])
        dy := Integer(parts[7])
        steps := Max(2, Integer(parts[8]))

        if (mode = "host_background" && hwnd > 0) {
            lStart := ((sy & 0xFFFF) << 16) | (sx & 0xFFFF)
            DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0200, "Ptr", 0, "Ptr", lStart)
            DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0207, "Ptr", 0x0010, "Ptr", lStart)
            Loop steps {
                cx := Round(sx + (dx * A_Index) / steps)
                cy := Round(sy + (dy * A_Index) / steps)
                lCur := ((cy & 0xFFFF) << 16) | (cx & 0xFFFF)
                DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0200, "Ptr", 0x0010, "Ptr", lCur)
                Sleep 12
            }
            lEnd := (((sy + dy) & 0xFFFF) << 16) | ((sx + dx) & 0xFFFF)
            DllCall("PostMessageW", "Ptr", hwnd, "UInt", 0x0208, "Ptr", 0, "Ptr", lEnd)
        } else {
            DllCall("SetCursorPos", "Int", sx, "Int", sy)
            Sleep 14
            Click("Middle Down")
            gHeldButtons["Middle"] := true
            Sleep 22
            prevX := 0
            prevY := 0
            Loop steps {
                ; Smooth horizontal step with zero pitch drift when dy == 0
                curX := Round((dx * A_Index) / steps)
                curY := Round((dy * A_Index) / steps)
                stepX := curX - prevX
                stepY := curY - prevY
                prevX := curX
                prevY := curY
                if (stepX != 0 || stepY != 0) {
                    DllCall("mouse_event", "UInt", 0x0001, "Int", stepX, "Int", stepY, "UInt", 0, "UPtr", 0)
                }
                Sleep 12
            }
            Sleep 18
            Click("Middle Up")
            gHeldButtons.Delete("Middle")
        }
        return "OK|DRAG_MMB"
    }

    if (cmd = "KEY_TAP" && parts.Length >= 5) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        key := MapKeyName(parts[4])
        holdMs := Max(15, Integer(parts[5]))

        if (mode = "host_background" && hwnd > 0) {
            ControlSend("{" . key . " down}", , "ahk_id " . hwnd)
            Sleep holdMs
            ControlSend("{" . key . " up}", , "ahk_id " . hwnd)
        } else {
            SendEvent("{" . key . " down}")
            Sleep holdMs
            SendEvent("{" . key . " up}")
        }
        return "OK|KEY_TAP"
    }

    if (cmd = "KEY_DOWN" && parts.Length >= 4) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        key := MapKeyName(parts[4])

        if (mode = "host_background" && hwnd > 0) {
            ControlSend("{" . key . " down}", , "ahk_id " . hwnd)
        } else {
            SendEvent("{" . key . " down}")
        }
        gHeldKeys[key] := mode . "|" . hwnd
        return "OK|KEY_DOWN"
    }

    if (cmd = "KEY_UP" && parts.Length >= 4) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        key := MapKeyName(parts[4])

        if (mode = "host_background" && hwnd > 0) {
            ControlSend("{" . key . " up}", , "ahk_id " . hwnd)
        } else {
            SendEvent("{" . key . " up}")
        }
        if gHeldKeys.Has(key) {
            gHeldKeys.Delete(key)
        }
        return "OK|KEY_UP"
    }

    if (cmd = "CHAT_CMD" && parts.Length >= 4) {
        mode := StrLower(parts[2])
        hwnd := Integer(parts[3])
        text := parts[4]
        if (mode = "host_background" && hwnd > 0) {
            ControlSend("{Enter}", , "ahk_id " . hwnd)
            Sleep 120
            ControlSendText(text, , "ahk_id " . hwnd)
            Sleep 90
            ControlSend("{Enter}", , "ahk_id " . hwnd)
        } else {
            SendEvent("{Enter}")
            Sleep 120
            SendText(text)
            Sleep 90
            SendEvent("{Enter}")
        }
        return "OK|CHAT_CMD"
    }

    return "ERR|UNKNOWN_CMD:" . cmd
}

CleanupAllInputs(reason := "", code := 0) {
    global gHeldKeys, gHeldButtons
    for key, meta in gHeldKeys {
        try {
            m := StrSplit(meta, "|")
            if (m.Length >= 2 && m[1] = "host_background" && Integer(m[2]) > 0) {
                ControlSend("{" . key . " up}", , "ahk_id " . Integer(m[2]))
            } else {
                SendEvent("{" . key . " up}")
            }
        }
    }
    gHeldKeys.Clear()

    for btn, _ in gHeldButtons {
        try {
            Click(btn . " Up")
        }
    }
    gHeldButtons.Clear()
}
