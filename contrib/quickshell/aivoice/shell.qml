import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import Quickshell
import Quickshell.Io

// Three systems, one virtual mic:
//   VOICE CONVERSION  - your speech -> a target voice (X-VC / RVC / MeanVC2 / ...)
//   VOICE MODULATION  - your own voice with altered attributes (DSP, no target voice)
//   SPEECH-TO-SPEECH  - ASR -> optional LLM -> TTS (aivoice experimental)
// Engine buttons come from `aivoice vc engines --json`: only runnable engines are clickable.

ShellRoot {
    id: root
    property bool opened: true
    property string major: "vc"           // vc | mod | s2s | res
    property var st: ({ running: false, pid: "", version: "" })
    property var live: ({ running: false })   // `aivoice vc status --json`
    property var engines: []
    property var presets: []
    property var s2s: ({})
    property var voices: []
    property var results: []
    property string voice: ""
    property string mode: "balanced"
    property string vcEngine: "default"   // default = the voice's saved engine (RVC when it has a .pth)
    property real denoise: 0
    property string preset: "masculine"
    property real strength: 65
    property bool advanced: false
    property bool showExperimental: false
    property var adv: ({ pitch: 0, formants: 0, resonance: 0, body: 0, brightness: 0, roughness: 0, breathiness: 0, intonation: 1 })
    property string message: ""
    property string logText: ""
    property string tab: "voices"
    property var doctor: []
    property var inputs: []
    property var rt: ({ engine: "rvc", pitch: 0, index_rate: 0.5, rvc_model: null })
    onVoiceChanged: if (voice) sh(rtP, ["rt", voice])
    property real level: 0
    property bool fixing: false
    readonly property var modes: ["lowest-latency", "balanced", "best-quality"]
    readonly property var mainEngines: ["auto", "tvtsyn", "rvc", "meanvc2", "xvc"]
    readonly property var expEngines: ["stylestream", "streamvoiceanon", "vossa", "rtvc", "zerovc-repro", "tvtsyn-repro", "tvtsyn-aivoice"]
    property var models: []
    // a session is live when the widget started it OR any `aivoice live/virtualmic/modulate` session reports in vc.json
    readonly property bool isLive: !!st.running || !!live.running
    property var repro: []
    onMajorChanged: if (major === "res") sh(reproP, ["repro"])
    readonly property var mainPresets: ["natural", "masculine", "feminine", "deep", "warm", "bright", "broadcast", "robotic"]
    function sh(p, args) { const h = Quickshell.env("HOME"), u = Quickshell.env("USER"); p.environment = ({ HOME: h, USER: u, PATH: h + "/.local/bin:" + h + "/.nix-profile/bin:/etc/profiles/per-user/" + u + "/bin:/run/current-system/sw/bin" }); p.command = ["/run/current-system/sw/bin/bash", Quickshell.shellDir + "/run.sh"].concat(args); p.running = true }
    function vname(v) { return v.name || v.display_name || v.id || String(v) }
    function eng(name) { for (let i = 0; i < engines.length; i++) if (engines[i].name === name) return engines[i]; return null }
    function engState(name) { if (name === "auto" || name === "default") return "READY"; const e = eng(name); return e ? e.state : "UNKNOWN" }
    function engRunnable(name) { if (name === "auto" || name === "default") return true; const e = eng(name); return !!e && e.runnable }
    function stateText(s) { return ({ READY: "ready", INSTALLED: "installed", EXPERIMENTAL: "experimental", MISSING_WEIGHTS: "missing model", MISSING_RUNTIME: "missing runtime", UNSUPPORTED_HARDWARE: "unavailable", RESEARCH_ONLY: "research only", BLOCKED_LICENSE: "license restriction", BLOCKED_CODE: "no public code/weights" })[s] || s.toLowerCase() }
    function advArgs() {
        const a = [], d = root.adv
        if (d.pitch) a.push("--pitch", String(d.pitch)); if (d.formants) a.push("--formants", String(d.formants))
        if (d.resonance) a.push("--resonance", String(d.resonance)); if (d.body) a.push("--body", String(d.body))
        if (d.brightness) a.push("--brightness", String(d.brightness)); if (d.roughness) a.push("--roughness", String(d.roughness))
        if (d.breathiness) a.push("--breathiness", String(d.breathiness)); if (d.intonation !== 1) a.push("--intonation", String(d.intonation))
        return a
    }
    function advParams() {   // the same overrides as `vc set --param NAME=VALUE`
        const a = [], d = root.adv
        if (d.pitch) a.push("pitch_semitones=" + d.pitch); if (d.formants) a.push("formant_ratio=" + (1 + d.formants / 100))
        if (d.resonance) a.push("resonance_db=" + d.resonance); if (d.body) a.push("body_db=" + d.body)
        if (d.brightness) a.push("brightness_db=" + d.brightness); if (d.roughness) a.push("roughness=" + d.roughness)
        if (d.breathiness) a.push("breathiness=" + d.breathiness); if (d.intonation !== 1) a.push("intonation=" + d.intonation)
        return a
    }
    function setAdv(k, v) { const d = Object.assign({}, root.adv); d[k] = v; root.adv = d; root.pushMod() }
    function pushMod() { if (root.live.running && root.live.mode === "modulation") sh(modsetP, ["modset", root.preset, String(Math.round(root.strength))].concat(advParams())) }
    function fmt(v, unit) { return (v === undefined || v === null) ? "–" : (Math.round(v * 10) / 10) + (unit || "") }

    IpcHandler { target: "aivoice"; function show(): void { root.opened = true; root.refresh() } function open(): void { root.opened = true; root.refresh(); root.sh(doctorP, ["doctor"]) } function toggle(): void { root.opened = !root.opened } function tab(name: string): void { root.opened = true; root.tab = name; if (name === "doctor") root.sh(doctorP, ["doctor"]) } function mode(name: string): void { root.opened = true; root.major = name } function bypass(): void { root.sh(actP, ["bypass"]) } }
    Process { id: statusP; stdout: StdioCollector { onStreamFinished: { try { root.st = JSON.parse(text) } catch (e) {} } } }
    Process { id: liveP; stdout: StdioCollector { onStreamFinished: { try { root.live = JSON.parse(text) } catch (e) { root.live = ({ running: false }) } } } }
    Process { id: enginesP; stdout: StdioCollector { onStreamFinished: { try { root.engines = JSON.parse(text) } catch (e) { root.engines = [] } } } }
    Process { id: presetsP; stdout: StdioCollector { onStreamFinished: { try { root.presets = JSON.parse(text) } catch (e) { root.presets = [] } } } }
    Process { id: s2sP; stdout: StdioCollector { onStreamFinished: { try { root.s2s = JSON.parse(text) } catch (e) { root.s2s = ({}) } } } }
    Process { id: reproP; stdout: StdioCollector { onStreamFinished: { try { root.repro = JSON.parse(text) } catch (e) { root.repro = [] } } } }
    Process { id: modsetP; stdout: StdioCollector { onStreamFinished: {} } }
    Process { id: voicesP; stdout: StdioCollector { onStreamFinished: { try { root.voices = JSON.parse(text); if (!root.voice && root.voices.length) root.voice = root.vname(root.voices[0]) } catch (e) { root.voices = [] } } } }
    Process { id: searchP; stdout: StdioCollector { onStreamFinished: { try { root.results = JSON.parse(text).results || []; root.message = root.results.length + " results" } catch (e) { root.results = []; root.message = "Search failed" } } } }
    Process { id: actP; stdout: StdioCollector { onStreamFinished: { root.message = text.trim().split("\n").slice(-3).join("  ·  "); root.refresh(); if (root.fixing) { root.fixing = false; root.sh(doctorP, ["doctor"]) } } } }
    Process { id: doctorP; stdout: StdioCollector { onStreamFinished: root.doctor = text.split("\n").filter(l => /[✓✗]/.test(l)) } }
    Process { id: inputsP; stdout: StdioCollector { onStreamFinished: { try { root.inputs = JSON.parse(text) } catch (e) { root.inputs = [] } } } }
    Process { id: levelP; stdout: StdioCollector { onStreamFinished: root.level = parseFloat(text) || 0 } }
    // input meter: from the running session's status when live (no extra capture), else a short parec probe
    Timer { interval: 700; running: root.opened; repeat: true; onTriggered: { if (root.live.running) root.level = root.live.level_in || 0; else if (!levelP.running) root.sh(levelP, ["level"]) } }
    Process { id: rtP; stdout: StdioCollector { onStreamFinished: { try { const j = JSON.parse(text); if (j.engine) root.rt = j } catch (e) {} } } }
    function setRt(args) { sh(rtP, ["rt", root.voice].concat(args)) }
    Process { id: logP; stdout: StdioCollector { onStreamFinished: root.logText = text } }
    Process { id: modelsP; stdout: StdioCollector { onStreamFinished: { try { root.models = JSON.parse(text) } catch (e) { root.models = [] } } } }
    function refresh() { sh(statusP, ["status"]); sh(voicesP, ["voices"]); sh(liveP, ["vcstatus"]); if (root.major === "s2s") sh(s2sP, ["s2sstatus"]) }
    // live metrics every 2 s (the session itself writes them once a second; nothing expensive is polled here)
    Timer { interval: 2000; running: root.opened; repeat: true; triggeredOnStart: true; onTriggered: { root.sh(statusP, ["status"]); if (!liveP.running) root.sh(liveP, ["vcstatus"]); if (root.major === "s2s" && !s2sP.running) root.sh(s2sP, ["s2sstatus"]); if (root.major === "res" && !reproP.running) root.sh(reproP, ["repro"]); if (root.tab === "logs") root.sh(logP, ["log"]) } }
    // engine availability changes rarely: every 60 s
    Timer { interval: 60000; running: root.opened; repeat: true; triggeredOnStart: true; onTriggered: if (!enginesP.running) root.sh(enginesP, ["vcengines"]) }
    Component.onCompleted: {
        sh(inputsP, ["inputs"]); sh(voicesP, ["voices"]); sh(doctorP, ["doctor"]); sh(presetsP, ["presets"]); sh(modelsP, ["models"])
        // test hooks (offscreen screenshots): AIVOICE_UI_MAJOR=vc|mod|s2s|res AIVOICE_UI_TAB=voices|models|...
        if (Quickshell.env("AIVOICE_UI_MAJOR")) major = Quickshell.env("AIVOICE_UI_MAJOR")
        if (Quickshell.env("AIVOICE_UI_TAB")) tab = Quickshell.env("AIVOICE_UI_TAB")
    }

    component Card: GalaxyPanel { radius: Galaxy.radiusTile; Layout.fillWidth: true }
    component Label2: Text { font.family: "Cellforge"; color: "#8a84a8"; font.pixelSize: 10; font.letterSpacing: 1.5 }
    component Metric: ColumnLayout { property string k: ""; property string v: "–"; spacing: 0; Layout.fillWidth: true
        Label2 { text: parent.k }
        Text { text: parent.v; font.family: "NullType"; font.pixelSize: 12; color: "#f2f0ff" } }
    component SliderRow: RowLayout { id: sr; property string label: ""; property real from: 0; property real to: 1; property real step: 0.05; property real value: 0; property string unit: ""; signal moved(real v)
        Layout.fillWidth: true; spacing: 8
        Label2 { Layout.preferredWidth: 96; text: sr.label + " " + (Math.round(sr.value * 100) / 100) + sr.unit }
        GalaxySlider { Layout.fillWidth: true; from: sr.from; to: sr.to; stepSize: sr.step; value: sr.value; onPressedChanged: if (!pressed) sr.moved(value) } }

    FloatingWindow {
        id: window
        title: "Galaxy AIVoice"
        visible: root.opened
        implicitWidth: 640; implicitHeight: 860
        minimumSize: Qt.size(500, 600)
        color: "transparent"
        onClosed: root.opened = false
        Timer { interval: 3000; running: !!Quickshell.env("AIVOICE_UI_SCREENSHOT"); onTriggered: frame.grabToImage(function(r) { r.saveToFile(Quickshell.env("AIVOICE_UI_SCREENSHOT")); if (Quickshell.env("AIVOICE_UI_SCREENSHOT_QUIT")) Qt.quit() }) }
        Item {
            id: frame
            anchors.fill: parent
            // gradient + stars painted by a Canvas clipped to the rounded rect: a gradient on a rounded Rectangle
            // is interpolated per triangle and shows a seam at the diagonal
            Starfield { anchors.fill: parent; cornerRadius: Galaxy.radiusWindow; seed: 3; paintBase: true }
            Rectangle { anchors.fill: parent; radius: Galaxy.radiusWindow; color: "transparent"; border.color: Galaxy.edge; border.width: 1 }
        ColumnLayout {
            anchors.fill: parent; anchors.margins: 20; spacing: 10
            RowLayout {
                Layout.fillWidth: true; spacing: 10
                Text {
                    Layout.fillWidth: true; font.family: "Cellforge"; text: "✦  O R B I T  /  A I V O I C E"; font.pixelSize: 12; color: "#d4b8ff"
                    MouseArea { anchors.fill: parent; cursorShape: Qt.SizeAllCursor; onPressed: window.startSystemMove() }
                }
                Rectangle {
                    implicitWidth: lr.implicitWidth + 18; implicitHeight: 24; radius: 12
                    color: root.isLive ? "#1d3a2a" : "#1a1630"; border.color: root.isLive ? "#7dffa8" : "#3a2f5c"
                    RowLayout { id: lr; anchors.centerIn: parent; spacing: 6
                        Rectangle { width: 8; height: 8; radius: 4; color: root.isLive ? "#7dffa8" : "#5d5878"
                            SequentialAnimation on opacity { running: root.isLive; loops: Animation.Infinite; NumberAnimation { to: 0.25; duration: 700 } NumberAnimation { to: 1; duration: 700 } } }
                        Text { font.family: "Cellforge"; font.pixelSize: 10; color: "#f2f0ff"; text: root.isLive ? (root.live.muted ? "MUTED" : root.live.bypass ? "BYPASS" : "LIVE") : "OFFLINE" } }
                }
                GalaxyButton { text: "×"; hint: "Hide · virtual mic keeps running"; onClicked: root.opened = false }
            }
            // the three systems
            RowLayout { Layout.fillWidth: true; spacing: 6
                Repeater { model: [["vc", "󰗅 VOICE CONVERSION"], ["mod", "󰎈 VOICE MODULATION"], ["s2s", "󰗋 SPEECH-TO-SPEECH"], ["res", "󰙨 RESEARCH"]]
                    GalaxyButton { required property var modelData; text: modelData[1]; selected: root.major === modelData[0]; Layout.fillWidth: true; font.pixelSize: 11
                        onClicked: { root.major = modelData[0]; if (root.major === "s2s") root.sh(s2sP, ["s2sstatus"]) } } }
            }
            // hero + live metrics (shared by VC and modulation: both use aivoice-virtual-mic)
            Card {
                implicitHeight: heroCol.implicitHeight + 28; hot: root.isLive; visible: root.major !== "s2s"
                ColumnLayout { id: heroCol; anchors.fill: parent; anchors.margins: 14; spacing: 8
                    RowLayout { spacing: 14
                        Text { font.family: "JetBrainsMono Nerd Font"; text: root.isLive ? "󰍬" : "󰍭"; font.pixelSize: 44; color: "#b46bff" }
                        ColumnLayout { Layout.fillWidth: true; spacing: 3
                            Label2 { text: root.major === "vc" ? "TARGET VOICE" : "YOUR VOICE · PRESET" }
                            Text { Layout.fillWidth: true; elide: Text.ElideRight; font.family: "Zerodot"; font.pixelSize: 18; color: "#f2f0ff"
                                text: root.major === "vc" ? ((root.live.running && root.live.voice) || root.voice || "no voice installed") : (root.preset + " · " + Math.round(root.strength) + "%") }
                            Text { font.family: "NullType"; font.pixelSize: 11; color: "#9a96b3"
                                text: (root.live.running ? ("engine " + root.live.engine + (root.live.engine !== root.live.selected_engine ? " (fallback from " + root.live.selected_engine + ")" : "")) : ("mode " + root.mode)) + "   ·   v" + (root.st.version || "?") + (root.st.running ? "   ·   pid " + root.st.pid : "") }
                        }
                    }
                    GridLayout { Layout.fillWidth: true; columns: 4; columnSpacing: 8; rowSpacing: 6; visible: !!root.live.running
                        Metric { k: "ALGORITHMIC"; v: root.fmt(root.live.algorithmic_latency_ms, " ms") }
                        Metric { k: "ENGINE P95"; v: root.fmt(root.live.processing ? (root.live.processing.engine_ms || {}).p95 : null, " ms") }
                        Metric { k: "RTF"; v: root.fmt(root.live.rtf) }
                        Metric { k: "CHUNK"; v: root.fmt(root.live.chunk_ms, " ms") }
                        Metric { k: "GPU"; v: root.live.gpu && root.live.gpu.util_pct !== undefined ? root.live.gpu.util_pct + " %" : "–" }
                        Metric { k: "VRAM"; v: root.live.worker && root.live.worker.vram_allocated_mib !== undefined ? root.fmt(root.live.worker.vram_allocated_mib, " MiB") : (root.live.gpu && root.live.gpu.mem_used_mib !== undefined ? root.live.gpu.mem_used_mib + " MiB" : "–") }
                        Metric { k: "UNDERRUNS / DROPS"; v: (root.live.underruns || 0) + " / " + (root.live.dropped_blocks || 0) }
                        Metric { k: "RATE"; v: (root.live.sample_rate || "–") + " / " + (root.live.engine_sample_rate || "–") }
                        Metric { k: "OUT BUFFER"; v: root.fmt(root.live.prebuffer_ms, " ms") }
                        Metric { k: "MIC→VIRTUAL EST."; v: root.fmt(root.live.end_to_end_estimate_ms, " ms") }
                        Metric { k: "FALLBACKS"; v: String((root.live.fallbacks || []).length) }
                        Metric { k: "UPTIME"; v: root.fmt(root.live.uptime_s, " s") }
                    }
                    GalaxyPill { visible: !!root.live.running && ((root.live.fallbacks || []).length > 0 || (root.live.underruns || 0) > 0)
                        state_: (root.live.fallbacks || []).length ? "err" : "warn"
                        text: (root.live.fallbacks || []).length ? "FALLBACK: " + root.live.engine + " (from " + root.live.selected_engine + ")"
                                                                : "DROPOUTS: " + root.live.underruns + " underruns" }
                    RowLayout { Layout.fillWidth: true; spacing: 8
                        Label2 { text: "IN" }
                        Rectangle { Layout.fillWidth: true; implicitHeight: 8; radius: 4; color: "#1a1630"; border.color: "#26223a"
                            Rectangle { height: parent.height; radius: 4; width: parent.width * Math.min(1, root.level); color: root.level > 0.8 ? "#ff8fb1" : "#7dffa8"; Behavior on width { NumberAnimation { duration: 150 } } } }
                        Label2 { text: "OUT"; visible: !!root.live.running }
                        Rectangle { Layout.fillWidth: true; implicitHeight: 8; radius: 4; color: "#1a1630"; border.color: "#26223a"; visible: !!root.live.running
                            Rectangle { height: parent.height; radius: 4; width: parent.width * Math.min(1, root.live.level_out || 0); color: "#b46bff"; Behavior on width { NumberAnimation { duration: 150 } } } }
                    }
                }
            }
            // input picker (all modes record from your real mic; aivoice's own devices are never listed)
            RowLayout { Layout.fillWidth: true; spacing: 8; visible: root.major !== "s2s"
                Text { font.family: "JetBrainsMono Nerd Font"; text: "󰍬"; color: "#b46bff"; font.pixelSize: 16 }
                ComboBox { id: inPick; Layout.fillWidth: true; implicitHeight: 34; model: root.inputs; textRole: "desc"; font.family: "Cellforge"; font.pixelSize: 11
                    currentIndex: root.inputs.findIndex(i => i.current)
                    onActivated: (i) => { root.sh(actP, ["setinput", root.inputs[i].name]); root.sh(inputsP, ["inputs"]) }
                    contentItem: Text { leftPadding: 10; text: inPick.displayText || "default input"; color: "#d4b8ff"; font: inPick.font; verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight }
                    background: Rectangle { radius: 10; color: "#0b0914"; border.color: "#26223a" } }
            }

            // ============================ VOICE CONVERSION ============================
            ColumnLayout { Layout.fillWidth: true; spacing: 8; visible: root.major === "vc"
                Label2 { text: "ENGINE" }
                RowLayout { spacing: 6; Layout.fillWidth: true
                    GalaxyButton { implicitHeight: 30; text: "Voice default"; hint: "The voice's saved engine: RVC v2 when it has a .pth, else MeanVC2 (previous runtime path)"
                        selected: root.vcEngine === "default"; enabled: !root.isLive; onClicked: root.vcEngine = "default" }
                    Repeater { model: root.mainEngines
                        GalaxyButton { required property string modelData; implicitHeight: 30; Layout.fillWidth: true
                            text: ({ auto: "Auto", tvtsyn: "TVTSyn (NC)", xvc: "X-VC", rvc: "RVC", meanvc2: "MeanVC2" })[modelData] + (root.engRunnable(modelData) ? "" : " ·")
                            hint: modelData === "auto" ? "Pick from LOCAL benchmark results (aivoice benchmark vc); never from paper numbers"
                                : root.stateText(root.engState(modelData)) + (root.eng(modelData) ? " — " + root.eng(modelData).caps.license + " — " + root.eng(modelData).detail : "")
                            selected: root.vcEngine === modelData; enabled: !root.isLive && root.engRunnable(modelData)
                            onClicked: root.vcEngine = modelData } }
                }
                GalaxyButton { implicitHeight: 26; text: (root.showExperimental ? "▾" : "▸") + " Experimental engines"; onClicked: root.showExperimental = !root.showExperimental }
                Flow { Layout.fillWidth: true; spacing: 6; visible: root.showExperimental
                    Repeater { model: root.expEngines
                        GalaxyButton { required property string modelData; implicitHeight: 28
                            text: modelData + "  ·  " + root.stateText(root.engState(modelData))
                            hint: root.eng(modelData) ? (root.eng(modelData).caps.notes || root.eng(modelData).detail) : ""
                            selected: root.vcEngine === modelData; enabled: !root.isLive && root.engRunnable(modelData)
                            onClicked: root.vcEngine = modelData } }
                }
                RowLayout { Layout.fillWidth: true; spacing: 10; visible: root.vcEngine === "rvc" || (root.vcEngine === "default" && root.rt.engine === "rvc")
                    Label2 { text: "PITCH " + (root.rt.pitch > 0 ? "+" : "") + Math.round(root.rt.pitch) }
                    GalaxySlider { Layout.fillWidth: true; from: -24; to: 24; stepSize: 1; value: root.rt.pitch; enabled: !root.isLive
                        onPressedChanged: if (!pressed) root.setRt(["--pitch", String(value)]) }
                    Label2 { text: "INDEX " + Number(root.rt.index_rate).toFixed(2) }
                    GalaxySlider { Layout.preferredWidth: 140; from: 0; to: 1; stepSize: 0.05; value: root.rt.index_rate; enabled: !root.isLive && !!root.rt.rvc_index
                        onPressedChanged: if (!pressed) root.setRt(["--index-rate", String(value)]) }
                }
                SliderRow { label: "NOISE SUPPR."; from: 0; to: 1; step: 0.05; value: root.denoise; visible: root.vcEngine !== "default" && root.vcEngine !== "rvc" && root.vcEngine !== "meanvc2"
                    onMoved: (v) => root.denoise = v }
                RowLayout { Layout.fillWidth: true; spacing: 6
                    Repeater { model: root.modes
                        GalaxyButton { required property string modelData; text: modelData; selected: root.mode === modelData; Layout.fillWidth: true; implicitHeight: 30; onClicked: root.mode = modelData } }
                }
            }

            // ============================ VOICE MODULATION ============================
            ColumnLayout { Layout.fillWidth: true; spacing: 8; visible: root.major === "mod"
                Label2 { text: "PRESET  (your own voice: no target voice, no consent prompt, nothing leaves this machine)" }
                Flow { Layout.fillWidth: true; spacing: 6
                    Repeater { model: root.mainPresets
                        GalaxyButton { required property string modelData; implicitHeight: 30; text: modelData; selected: root.preset === modelData
                            onClicked: { root.preset = modelData; root.pushMod() } } }
                    ComboBox { id: morePick; implicitHeight: 30; implicitWidth: 160; font.family: "Cellforge"; font.pixelSize: 11
                        model: root.presets.map(p => p.name).filter(n => root.mainPresets.indexOf(n) < 0)
                        displayText: root.mainPresets.indexOf(root.preset) < 0 ? root.preset : "more…"
                        onActivated: (i) => { root.preset = morePick.model[i]; root.pushMod() }
                        contentItem: Text { leftPadding: 10; text: morePick.displayText; color: "#d4b8ff"; font: morePick.font; verticalAlignment: Text.AlignVCenter }
                        background: Rectangle { radius: 10; color: "#0b0914"; border.color: "#26223a" } }
                }
                SliderRow { label: "STRENGTH"; from: 0; to: 100; step: 1; value: root.strength; unit: "%"; onMoved: (v) => { root.strength = v; root.pushMod() } }
                GalaxyButton { implicitHeight: 26; text: (root.advanced ? "▾" : "▸") + " Advanced"; onClicked: root.advanced = !root.advanced }
                ColumnLayout { Layout.fillWidth: true; spacing: 2; visible: root.advanced
                    SliderRow { label: "PITCH"; from: -12; to: 12; step: 0.5; value: root.adv.pitch; unit: " st"; onMoved: (v) => root.setAdv("pitch", v) }
                    SliderRow { label: "FORMANTS"; from: -30; to: 30; step: 1; value: root.adv.formants; unit: " %"; onMoved: (v) => root.setAdv("formants", v) }
                    SliderRow { label: "RESONANCE"; from: -6; to: 6; step: 0.5; value: root.adv.resonance; unit: " dB"; onMoved: (v) => root.setAdv("resonance", v) }
                    SliderRow { label: "BODY"; from: -6; to: 6; step: 0.5; value: root.adv.body; unit: " dB"; onMoved: (v) => root.setAdv("body", v) }
                    SliderRow { label: "BRIGHTNESS"; from: -6; to: 6; step: 0.5; value: root.adv.brightness; unit: " dB"; onMoved: (v) => root.setAdv("brightness", v) }
                    SliderRow { label: "ROUGHNESS"; from: 0; to: 1; step: 0.05; value: root.adv.roughness; onMoved: (v) => root.setAdv("roughness", v) }
                    SliderRow { label: "BREATHINESS"; from: 0; to: 1; step: 0.05; value: root.adv.breathiness; onMoved: (v) => root.setAdv("breathiness", v) }
                    SliderRow { label: "INTONATION"; from: 0; to: 2; step: 0.05; value: root.adv.intonation; unit: "x"; onMoved: (v) => root.setAdv("intonation", v) }
                    Text { Layout.fillWidth: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"
                        text: "Advanced values override the preset. Identity is preserved best by small formant + pitch changes; see docs/masculine-voice.md." }
                }
                RowLayout { Layout.fillWidth: true; spacing: 6
                    GalaxyButton { text: "Reset"; onClicked: { root.adv = ({ pitch: 0, formants: 0, resonance: 0, body: 0, brightness: 0, roughness: 0, breathiness: 0, intonation: 1 }); root.strength = 65; root.pushMod() } }
                    GalaxyButton { text: "A/B bypass"; hint: "Hear your unprocessed voice (delay-matched)"; enabled: !!root.live.running; selected: !!root.live.bypass; onClicked: root.sh(actP, ["bypass"]) }
                    TextField { id: pname; Layout.fillWidth: true; placeholderText: "preset name"; font.family: "NullType"; color: "#f2f0ff"; placeholderTextColor: "#5d5878"
                        background: Rectangle { radius: 10; color: "#0b0914"; border.color: pname.activeFocus ? "#b46bff" : "#26223a" } }
                    GalaxyButton { text: "Save preset"; enabled: pname.text.length > 0 && !actP.running
                        onClicked: { root.sh(actP, ["presetsave", pname.text, root.preset, String(Math.round(root.strength))].concat(root.advArgs())); root.sh(presetsP, ["presets"]) } }
                }
            }

            // ============================ RESEARCH (paper reproductions) ============================
            Card { visible: root.major === "res"; implicitHeight: resCol.implicitHeight + 28
                ColumnLayout { id: resCol; anchors.fill: parent; anchors.margins: 14; spacing: 8
                    Label2 { text: "PAPER REPRODUCTIONS · EXPERIMENTAL · NEVER OFFICIAL WEIGHTS" }
                    Repeater { model: root.repro
                        Rectangle { required property var modelData; Layout.fillWidth: true; radius: 8; color: "#151127"
                            border.color: modelData.state === "Training" ? "#5b3d99" : modelData.state === "Failed" ? "#7a2f45" : "#2a2344"
                            implicitHeight: rcol.implicitHeight + 16
                            ColumnLayout { id: rcol; anchors.fill: parent; anchors.margins: 8; spacing: 3
                                RowLayout { spacing: 8
                                    Text { text: modelData.name; font.family: "NullType"; font.pixelSize: 12; color: "#e8e4ff" }
                                    Text { text: modelData.state + (modelData.stale ? " (no update > 10 min)" : ""); font.family: "NullType"; font.pixelSize: 11
                                        color: ({ "Ready": "#7fd1a8", "Training": "#b79cff", "Failed": "#ff8fa8", "Blocked": "#ff8fa8", "Checkpoint available": "#e6c07b" })[modelData.state] || "#9a96b3" }
                                    Item { Layout.fillWidth: true }
                                    Text { text: modelData.provenance + " · " + modelData.source_status; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8" } }
                                Text { visible: !!modelData.run && modelData.run.step !== undefined; font.family: "NullType"; font.pixelSize: 10; color: "#c9c3e6"
                                    text: { const r = modelData.run || {}; const l = r.loss || {}; const k = Object.keys(l)[0];
                                        return "step " + r.step + "/" + r.max_steps + (k ? "  ·  " + k + " " + l[k] : "") + "  ·  " + (r.samples_per_s || 0) + " samples/s"
                                            + "  ·  ETA " + (r.eta_s ? Math.round(r.eta_s / 60) + " min" : "–") + "  ·  " + (r.gpu || "cpu") + (r.vram_gib ? " " + r.vram_gib + " GiB" : "") } }
                                Text { visible: !!modelData.checkpoint || !!modelData.why; Layout.fillWidth: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"
                                    text: (modelData.checkpoint ? "checkpoint " + modelData.checkpoint + " · " + (modelData.weights || "") : (modelData.why || "")) }
                                RowLayout { visible: !!modelData.output && (modelData.state === "Training"); spacing: 6
                                    GalaxyButton { implicitHeight: 26; text: (modelData.run || {}).state === "paused" ? "resume" : "pause"; enabled: !actP.running
                                        onClicked: root.sh(actP, ["reprocontrol", (modelData.run || {}).state === "paused" ? "resume" : "pause", modelData.output]) }
                                    GalaxyButton { implicitHeight: 26; text: "stop"; enabled: !actP.running; onClicked: root.sh(actP, ["reprocontrol", "stop", modelData.output]) } }
                            } } }
                    Text { Layout.fillWidth: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"
                        text: "Training runs in its own background process, never in this widget: `aivoice reproduce train NAME ... --background`. Pause/stop write the run's control file; a checkpoint is saved first." }
                }
            }

            // ============================ SPEECH-TO-SPEECH ============================
            Card { visible: root.major === "s2s"; implicitHeight: s2sCol.implicitHeight + 28
                ColumnLayout { id: s2sCol; anchors.fill: parent; anchors.margins: 14; spacing: 8
                    Label2 { text: "MIC → VAD → ASR → (LLM) → TTS → VIRTUAL MIC   ·   separate from voice conversion" }
                    Text { font.family: "NullType"; font.pixelSize: 11; color: "#9a96b3"
                        text: "state " + (root.s2s.engine ? root.s2s.engine.state : "not running") + "   ·   TTS " + (root.s2s.settings ? root.s2s.settings.tts.backend + " (" + root.s2s.settings.tts.server_model + ")" : "–") + "   ·   voice " + (root.s2s.settings && root.s2s.settings.voice.active ? root.s2s.settings.voice.active : "default") }
                    Label2 { text: "ASR ENGINE" }
                    Flow { Layout.fillWidth: true; spacing: 6
                        Repeater { model: ["auto", "moonshine-tiny", "moonshine-small", "moonshine-medium", "parakeet"]
                            GalaxyButton { required property string modelData; implicitHeight: 28; text: modelData
                                hint: modelData === "auto" ? "" : ((root.s2s.asr_available || {})[modelData] || "ready to load")
                                enabled: !actP.running && (modelData === "auto" || (root.s2s.asr_available || {})[modelData] === null)
                                selected: !!root.s2s.settings && root.s2s.settings.asr.backend === modelData
                                onClicked: { root.sh(actP, ["s2sasr", modelData]); root.sh(s2sP, ["s2sstatus"]) } } }
                    }
                    RowLayout { spacing: 6
                        Label2 { text: "LLM REWRITE" }
                        GalaxyButton { implicitHeight: 28; text: "on"; selected: !!root.s2s.settings && !!root.s2s.settings.llm.enabled; onClicked: { root.sh(actP, ["s2sllm", "on"]); root.sh(s2sP, ["s2sstatus"]) } }
                        GalaxyButton { implicitHeight: 28; text: "off"; selected: !!root.s2s.settings && !root.s2s.settings.llm.enabled; onClicked: { root.sh(actP, ["s2sllm", "off"]); root.sh(s2sP, ["s2sstatus"]) } }
                    }
                    Label2 { text: "LATENCY / EXPRESSION PRESET" }
                    Flow { Layout.fillWidth: true; spacing: 6
                        Repeater { model: ["ultra-low-latency", "balanced", "expressive", "character", "accuracy"]
                            GalaxyButton { required property string modelData; implicitHeight: 28; text: modelData
                                selected: !!root.s2s.settings && root.s2s.settings.experimental.preset === modelData
                                onClicked: { root.sh(actP, ["s2spreset", modelData]); root.sh(s2sP, ["s2sstatus"]) } } }
                    }
                    Text { Layout.fillWidth: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"
                        text: "TTS engine, voice and expression: `aivoice experimental tts use …`, `aivoice voice use …`. Remote TTS needs `aivoice experimental remote allow`." }
                }
            }

            RowLayout { Layout.fillWidth: true; spacing: 6; visible: root.major === "vc"
                Repeater { model: [["voices", "󰗅 Voices"], ["search", "󰍉 Search"], ["models", "󰆼 Models"], ["doctor", "󰓙 Health"], ["logs", "󰦪 Logs"]]
                    GalaxyButton { required property var modelData; text: modelData[1]; selected: root.tab === modelData[0]; Layout.fillWidth: true; implicitHeight: 30
                        onClicked: { root.tab = modelData[0]; if (root.tab === "logs") root.sh(logP, ["log"]); if (root.tab === "doctor") root.sh(doctorP, ["doctor"]); if (root.tab === "models") root.sh(modelsP, ["models"]) } } }
            }
            Card {
                Layout.fillHeight: true; clip: true
                // voices
                ListView {
                    visible: root.major === "vc" && root.tab === "voices"; anchors.fill: parent; anchors.margins: 10; spacing: 6; model: root.voices
                    delegate: Rectangle { required property var modelData; width: ListView.view.width; height: 40; radius: 10
                        color: root.voice === root.vname(modelData) ? "#1a1630" : "transparent"; border.color: root.voice === root.vname(modelData) ? "#b46bff" : "#26223a"
                        Text { anchors.verticalCenter: parent.verticalCenter; x: 12; width: parent.width - 24; elide: Text.ElideRight; font.family: "Cellforge"; font.pixelSize: 12; color: "#d4b8ff"; text: root.vname(modelData) }
                        MouseArea { anchors.fill: parent; onClicked: root.voice = root.vname(modelData) } }
                    Text { anchors.centerIn: parent; visible: !root.voices.length; font.family: "Cellforge"; color: "#9a96b3"; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter
                        text: "No voices installed yet\nUse 󰍉 Search to install one" }
                }
                // search
                ColumnLayout {
                    visible: root.major === "vc" && root.tab === "search"; anchors.fill: parent; anchors.margins: 10; spacing: 8
                    RowLayout { Layout.fillWidth: true
                        TextField { id: q; Layout.fillWidth: true; placeholderText: "search voice-models.com…"; font.family: "NullType"; color: "#f2f0ff"; placeholderTextColor: "#5d5878"
                            background: Rectangle { radius: 10; color: "#0b0914"; border.color: q.activeFocus ? "#b46bff" : "#26223a" }
                            onAccepted: if (text) { root.message = "Searching…"; root.sh(searchP, ["search", text]) } }
                        GalaxyButton { text: "󰍉"; primary: true; enabled: !searchP.running && q.text; onClicked: q.accepted() } }
                    ListView { Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 6; model: root.results
                        delegate: Rectangle { required property var modelData; width: ListView.view.width; height: 48; radius: 10; color: "#0f0d1a"; border.color: "#26223a"
                            RowLayout { anchors.fill: parent; anchors.margins: 8; spacing: 8
                                ColumnLayout { Layout.fillWidth: true; spacing: 2
                                    Text { Layout.fillWidth: true; elide: Text.ElideRight; font.family: "Cellforge"; font.pixelSize: 12; color: "#d4b8ff"; text: modelData.name }
                                    Text { font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"; text: (modelData.architecture || "") + "  ·  " + (modelData.size_label || "") } }
                                GalaxyButton { text: "󰇚 Install"; enabled: !actP.running; onClicked: { root.message = "Installing " + modelData.name + "…"; root.sh(actP, ["install", modelData.id]) } } } } }
                }
                // models (catalog engines: pinned sources, verify re-hashes every file)
                ListView { visible: root.major === "vc" && root.tab === "models"; anchors.fill: parent; anchors.margins: 10; spacing: 6; clip: true; model: root.models
                    delegate: Rectangle { required property var modelData
                        width: ListView.view.width; height: mcol.implicitHeight + 16; radius: 10; color: Galaxy.surface; border.color: modelData.installed ? Galaxy.edge : Galaxy.line
                        RowLayout { anchors.fill: parent; anchors.margins: 8; spacing: 8
                            ColumnLayout { id: mcol; Layout.fillWidth: true; spacing: 2
                                RowLayout { spacing: 8
                                    Text { font.family: Galaxy.font; font.pixelSize: 12; color: Galaxy.ink; text: modelData.name }
                                    GalaxyPill { state_: modelData.installed ? "ok" : "idle"; text: modelData.installed ? "INSTALLED" : "AVAILABLE" } }
                                Text { Layout.fillWidth: true; elide: Text.ElideRight; font.family: "NullType"; font.pixelSize: 10; color: Galaxy.muted; text: modelData.size_hint || "" }
                                Text { Layout.fillWidth: true; elide: Text.ElideRight; font.family: "NullType"; font.pixelSize: 10
                                    color: /NON-COMMERCIAL|NC/.test(String(modelData.weights_license)) ? Galaxy.warn : Galaxy.label; text: "weights: " + (modelData.weights_license || "?") } }
                            GalaxyButton { implicitHeight: 28; text: "Verify"; visible: !!modelData.installed && modelData.name !== "xvc"; enabled: !actP.running
                                hint: "Re-hash every file against its pinned SHA-256"; onClicked: { root.message = "Verifying " + modelData.name + "…"; root.sh(actP, ["verify", modelData.name]) } }
                            GalaxyButton { implicitHeight: 28; text: modelData.installed ? "Repair" : "Install"; primary: !modelData.installed; enabled: !actP.running
                                hint: (modelData.size_hint || "") + " · " + (modelData.weights_license || ""); onClicked: { root.message = "Installing " + modelData.name + "… (see Logs)"; root.sh(actP, ["install-model", modelData.name]) } } } } }
                // doctor
                ListView { visible: root.major === "vc" && root.tab === "doctor"; anchors.fill: parent; anchors.margins: 10; spacing: 5; clip: true; model: root.doctor
                    delegate: Rectangle { required property string modelData
                        readonly property bool ok: modelData.indexOf("✓") >= 0
                        readonly property bool opt: modelData.indexOf("optional") >= 0
                        readonly property string key: modelData.split(/[✓✗]/)[0].trim()
                        width: ListView.view.width; height: 44; radius: 10; color: "#0f0d1a"; border.color: ok ? "#26223a" : opt ? "#2a2344" : "#5a2a44"
                        RowLayout { anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 4; spacing: 8
                            Rectangle { width: 8; height: 8; radius: 4; color: ok ? "#7dffa8" : opt ? "#5d5878" : "#ff8fb1" }
                            Text { Layout.preferredWidth: 150; elide: Text.ElideRight; font.family: "Cellforge"; font.pixelSize: 11; color: "#d4b8ff"; text: key }
                            Text { Layout.fillWidth: true; wrapMode: Text.Wrap; maximumLineCount: 2; elide: Text.ElideRight; font.family: "NullType"; font.pixelSize: 10; color: "#8a84a8"; text: modelData.split(/[✓✗]/).slice(1).join("").trim() }
                            GalaxyButton { visible: !ok; implicitHeight: 28; text: "󰁨 Fix"; primary: !opt; enabled: !actP.running
                                onClicked: { root.message = "Fixing " + key + "…"; root.fixing = true; root.sh(actP, ["fix", key]) } } } } }
                // logs (all modes)
                ScrollView { visible: root.major !== "vc" || root.tab === "logs"; anchors.fill: parent; anchors.margins: 12
                    TextArea { readOnly: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#c9c3e6"; text: root.logText || (root.live.errors && root.live.errors.length ? root.live.errors.join("\n") : "Logs: switch to Voice conversion → Logs for the session log"); background: null } }
            }
            Text { Layout.fillWidth: true; font.family: "Cellforge"; textFormat: Text.PlainText; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideRight; color: "#8a84a8"; font.pixelSize: 11
                text: root.message || (root.major === "mod" ? "Modulating your own voice · same speaker, altered attributes" : "Disclosed synthetic media only · use voices you have rights to") }
            RowLayout { Layout.fillWidth: true; spacing: 8
                GalaxyButton { text: "󰑐"; hint: "Refresh"; onClicked: { root.refresh(); root.sh(enginesP, ["vcengines"]); root.sh(presetsP, ["presets"]) } }
                GalaxyButton { visible: root.major === "vc"; text: "󰕾 Setup mic"; hint: "Create the aivoice null sink / virtual source only"; enabled: !actP.running; onClicked: root.sh(actP, ["setup"]) }
                GalaxyButton { visible: root.major === "vc"; text: "󰑓 Resample"; hint: "Regenerate the selected voice's reference sample (Piper TTS -> its RVC model)"; enabled: !!root.voice && !actP.running && !root.isLive; onClicked: { root.message = "Regenerating sample for " + root.voice + "…"; root.sh(actP, ["resynth", root.voice]) } }
                GalaxyButton { visible: root.major === "vc"; text: "󰇚 MeanVC2"; hint: "Download MeanVC2 runtime + models (multi-GB, needs PyTorch)"; enabled: !actP.running; onClicked: { root.message = "Downloading MeanVC2…"; root.sh(actP, ["models-install"]) } }
                GalaxyButton { visible: root.major !== "s2s" && root.isLive; text: root.live.bypass ? "󰕿 Bypassed" : "󰕾 Bypass"; selected: !!root.live.bypass; onClicked: root.sh(actP, ["bypass"]) }
                Item { Layout.fillWidth: true }
                GalaxyButton { visible: root.major === "vc" && !root.isLive; primary: true; text: "󰍬 Start"; enabled: !!root.voice && !actP.running && root.engRunnable(root.vcEngine)
                    onClicked: root.sh(actP, ["start", root.voice, root.mode, root.vcEngine, String(root.denoise)]) }
                GalaxyButton { visible: root.major === "mod" && !root.isLive; primary: true; text: "󰍬 Start modulation"; enabled: !actP.running
                    onClicked: root.sh(actP, ["modstart", root.preset, String(Math.round(root.strength))].concat(root.advArgs())) }
                GalaxyButton { visible: root.major !== "s2s" && root.isLive; primary: true; text: "󰓛 Stop"; enabled: !actP.running
                    hint: root.st.running ? "" : "Stops the running aivoice session (started outside the widget)"
                    onClicked: root.sh(actP, [root.st.running ? "stop" : "vcstop"]) }
                GalaxyButton { visible: root.major === "s2s" && !(root.s2s.engine); primary: true; text: "󰍬 Start S2S"; enabled: !actP.running && !root.isLive; hint: root.isLive ? "Stop the live VC / modulation session first (one virtual mic)" : ""
                    onClicked: root.sh(actP, ["s2sstart"]) }
                GalaxyButton { visible: root.major === "s2s" && !!(root.s2s.engine); primary: true; text: "󰓛 Stop S2S"; enabled: !actP.running; onClicked: root.sh(actP, ["s2sstop"]) }
            }
        }
        }
    }
}
