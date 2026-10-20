import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import Quickshell
import Quickshell.Io

ShellRoot {
    id: root
    property bool opened: true
    property var st: ({ running: false, pid: "", version: "" })
    property var voices: []
    property var results: []
    property string voice: ""
    property string mode: "balanced"
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
    function sh(p, args) { const h = Quickshell.env("HOME"), u = Quickshell.env("USER"); p.environment = ({ HOME: h, USER: u, PATH: h + "/.local/bin:" + h + "/.nix-profile/bin:/etc/profiles/per-user/" + u + "/bin:/run/current-system/sw/bin" }); p.command = ["/run/current-system/sw/bin/bash", Quickshell.shellDir + "/run.sh"].concat(args); p.running = true }
    function vname(v) { return v.name || v.display_name || v.id || String(v) }

    IpcHandler { target: "aivoice"; function show(): void { root.opened = true; root.refresh() } function open(): void { root.opened = true; root.refresh(); root.sh(doctorP, ["doctor"]) } function toggle(): void { root.opened = !root.opened } function tab(name: string): void { root.opened = true; root.tab = name; if (name === "doctor") root.sh(doctorP, ["doctor"]) } }
    Process { id: statusP; stdout: StdioCollector { onStreamFinished: { try { root.st = JSON.parse(text) } catch (e) {} } } }
    Process { id: voicesP; stdout: StdioCollector { onStreamFinished: { try { root.voices = JSON.parse(text); if (!root.voice && root.voices.length) root.voice = root.vname(root.voices[0]) } catch (e) { root.voices = [] } } } }
    Process { id: searchP; stdout: StdioCollector { onStreamFinished: { try { root.results = JSON.parse(text).results || []; root.message = root.results.length + " results" } catch (e) { root.results = []; root.message = "Search failed" } } } }
    Process { id: actP; stdout: StdioCollector { onStreamFinished: { root.message = text.trim().split("\n").slice(-3).join("  ·  "); root.refresh(); if (root.fixing) { root.fixing = false; root.sh(doctorP, ["doctor"]) } } } }
    Process { id: doctorP; stdout: StdioCollector { onStreamFinished: root.doctor = text.split("\n").filter(l => /[✓✗]/.test(l)) } }
    Process { id: inputsP; stdout: StdioCollector { onStreamFinished: { try { root.inputs = JSON.parse(text) } catch (e) { root.inputs = [] } } } }
    Process { id: levelP; stdout: StdioCollector { onStreamFinished: root.level = parseFloat(text) || 0 } }
    Timer { interval: 700; running: root.opened; repeat: true; onTriggered: if (!levelP.running) root.sh(levelP, ["level"]) }
    Process { id: rtP; stdout: StdioCollector { onStreamFinished: { try { const j = JSON.parse(text); if (j.engine) root.rt = j } catch (e) {} } } }
    function setRt(args) { sh(rtP, ["rt", root.voice].concat(args)) }
    Process { id: logP; stdout: StdioCollector { onStreamFinished: root.logText = text } }
    function refresh() { sh(statusP, ["status"]); sh(voicesP, ["voices"]) }
    Timer { interval: 3000; running: root.opened; repeat: true; triggeredOnStart: true; onTriggered: { root.sh(statusP, ["status"]); if (root.tab === "logs") root.sh(logP, ["log"]) } }
    Component.onCompleted: { sh(inputsP, ["inputs"]); sh(voicesP, ["voices"]); sh(doctorP, ["doctor"]) }

    component Card: Rectangle { radius: 14; color: "#120f22"; border.color: "#2a2344"; Layout.fillWidth: true }
    component Label2: Text { font.family: "Cellforge"; color: "#8a84a8"; font.pixelSize: 10; font.letterSpacing: 1.5 }

    FloatingWindow {
        id: window
        title: "Galaxy AIVoice"
        visible: root.opened
        implicitWidth: 600; implicitHeight: 760
        minimumSize: Qt.size(480, 520)
        color: "transparent"
        onClosed: root.opened = false
        Rectangle {
            anchors.fill: parent; radius: 22; border.color: "#3a2f5c"; border.width: 1
            color: "#0e0a1c"  // solid: rounded-rect vertex gradients interpolate per triangle -> visible seam at the diagonal
        }
        ColumnLayout {
            anchors.fill: parent; anchors.margins: 20; spacing: 12
            RowLayout {
                Layout.fillWidth: true; spacing: 10
                Text {
                    Layout.fillWidth: true; font.family: "Cellforge"; text: "✦  O R B I T  /  A I V O I C E"; font.pixelSize: 12; color: "#d4b8ff"
                    MouseArea { anchors.fill: parent; cursorShape: Qt.SizeAllCursor; onPressed: window.startSystemMove() }
                }
                Rectangle {
                    implicitWidth: lr.implicitWidth + 18; implicitHeight: 24; radius: 12
                    color: root.st.running ? "#1d3a2a" : "#1a1630"; border.color: root.st.running ? "#7dffa8" : "#3a2f5c"
                    RowLayout { id: lr; anchors.centerIn: parent; spacing: 6
                        Rectangle { width: 8; height: 8; radius: 4; color: root.st.running ? "#7dffa8" : "#5d5878"
                            SequentialAnimation on opacity { running: root.st.running; loops: Animation.Infinite; NumberAnimation { to: 0.25; duration: 700 } NumberAnimation { to: 1; duration: 700 } } }
                        Text { font.family: "Cellforge"; font.pixelSize: 10; color: "#f2f0ff"; text: root.st.running ? "LIVE" : "OFFLINE" } }
                }
                GalaxyButton { text: "×"; hint: "Hide · virtual mic keeps running"; onClicked: root.opened = false }
            }
            // hero
            Card {
                implicitHeight: 140; border.color: root.st.running ? "#5b3d99" : "#2a2344"
                RowLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 16
                    Text { font.family: "JetBrainsMono Nerd Font"; text: root.st.running ? "󰍬" : "󰍭"; font.pixelSize: 54; color: "#b46bff" }
                    ColumnLayout { Layout.fillWidth: true; spacing: 4
                        Label2 { text: "VOICE" }
                        Text { Layout.fillWidth: true; elide: Text.ElideRight; font.family: "Zerodot"; font.pixelSize: 18; color: "#f2f0ff"; text: root.voice || "no voice installed" }
                        Text { font.family: "NullType"; font.pixelSize: 11; color: "#9a96b3"
                            text: "mode " + root.mode + "   ·   v" + (root.st.version || "?") + (root.st.running ? "   ·   pid " + root.st.pid : "") }
                        RowLayout { spacing: 6; Layout.topMargin: 2
                            Repeater { model: [["rvc", "RVC v2"], ["meanvc2", "MeanVC2"]]
                                GalaxyButton { required property var modelData; implicitHeight: 26; text: modelData[1]
                                    selected: root.rt.engine === modelData[0]; enabled: !root.st.running && (modelData[0] !== "rvc" || !!root.rt.rvc_model)
                                    onClicked: root.setRt(["--engine", modelData[0]]) } } }
                    }
                }
            }
            RowLayout { Layout.fillWidth: true; spacing: 10; visible: root.rt.engine === "rvc"
                Label2 { text: "PITCH " + (root.rt.pitch > 0 ? "+" : "") + Math.round(root.rt.pitch) }
                Slider { id: pitchS; Layout.fillWidth: true; from: -24; to: 24; stepSize: 1; value: root.rt.pitch; enabled: !root.st.running
                    onPressedChanged: if (!pressed) root.setRt(["--pitch", String(value)]) }
                Label2 { text: "INDEX " + Number(root.rt.index_rate).toFixed(2) }
                Slider { id: idxS; Layout.preferredWidth: 140; from: 0; to: 1; stepSize: 0.05; value: root.rt.index_rate; enabled: !root.st.running && !!root.rt.rvc_index
                    onPressedChanged: if (!pressed) root.setRt(["--index-rate", String(value)]) }
            }
            RowLayout { Layout.fillWidth: true; spacing: 6
                Repeater { model: root.modes
                    GalaxyButton { required property string modelData; text: modelData; selected: root.mode === modelData; Layout.fillWidth: true; onClicked: root.mode = modelData } }
            }
            RowLayout { Layout.fillWidth: true; spacing: 8
                Text { font.family: "JetBrainsMono Nerd Font"; text: "󰍬"; color: "#b46bff"; font.pixelSize: 16 }
                ComboBox { id: inPick; Layout.fillWidth: true; implicitHeight: 34; model: root.inputs; textRole: "desc"; font.family: "Cellforge"; font.pixelSize: 11
                    currentIndex: root.inputs.findIndex(i => i.current)
                    onActivated: (i) => { root.sh(actP, ["setinput", root.inputs[i].name]); root.sh(inputsP, ["inputs"]) }
                    contentItem: Text { leftPadding: 10; text: inPick.displayText || "default input"; color: "#d4b8ff"; font: inPick.font; verticalAlignment: Text.AlignVCenter; elide: Text.ElideRight }
                    background: Rectangle { radius: 10; color: "#0b0914"; border.color: "#26223a" } }
                Rectangle { Layout.preferredWidth: 120; implicitHeight: 10; radius: 5; color: "#1a1630"; border.color: "#26223a"
                    Rectangle { height: parent.height; radius: 5; width: parent.width * Math.min(1, root.level); color: root.level > 0.8 ? "#ff8fb1" : "#7dffa8"
                        Behavior on width { NumberAnimation { duration: 150 } } } }
            }
            RowLayout { Layout.fillWidth: true; spacing: 6
                Repeater { model: [["voices", "󰗅 Voices"], ["search", "󰍉 Search"], ["doctor", "󰓙 Health"], ["logs", "󰦪 Logs"]]
                    GalaxyButton { required property var modelData; text: modelData[1]; selected: root.tab === modelData[0]; Layout.fillWidth: true
                        onClicked: { root.tab = modelData[0]; if (root.tab === "logs") root.sh(logP, ["log"]); if (root.tab === "doctor") root.sh(doctorP, ["doctor"]) } } }
            }
            Card {
                Layout.fillHeight: true; clip: true
                // voices
                ListView {
                    visible: root.tab === "voices"; anchors.fill: parent; anchors.margins: 10; spacing: 6; model: root.voices
                    delegate: Rectangle { required property var modelData; width: ListView.view.width; height: 40; radius: 10
                        color: root.voice === root.vname(modelData) ? "#1a1630" : "transparent"; border.color: root.voice === root.vname(modelData) ? "#b46bff" : "#26223a"
                        Text { anchors.verticalCenter: parent.verticalCenter; x: 12; width: parent.width - 24; elide: Text.ElideRight; font.family: "Cellforge"; font.pixelSize: 12; color: "#d4b8ff"; text: root.vname(modelData) }
                        MouseArea { anchors.fill: parent; onClicked: root.voice = root.vname(modelData) } }
                    Text { anchors.centerIn: parent; visible: !root.voices.length; font.family: "Cellforge"; color: "#9a96b3"; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter
                        text: "No voices installed yet\nUse 󰍉 Search to install one" }
                }
                // search
                ColumnLayout {
                    visible: root.tab === "search"; anchors.fill: parent; anchors.margins: 10; spacing: 8
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
                // doctor
                ListView { visible: root.tab === "doctor"; anchors.fill: parent; anchors.margins: 10; spacing: 5; clip: true; model: root.doctor
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
                // logs
                ScrollView { visible: root.tab === "logs"; anchors.fill: parent; anchors.margins: 12
                    TextArea { readOnly: true; wrapMode: Text.Wrap; font.family: "NullType"; font.pixelSize: 10; color: "#c9c3e6"; text: root.logText; background: null } }
            }
            Text { Layout.fillWidth: true; font.family: "Cellforge"; textFormat: Text.PlainText; wrapMode: Text.Wrap; maximumLineCount: 3; elide: Text.ElideRight; color: "#8a84a8"; font.pixelSize: 11
                text: root.message || "Disclosed synthetic media only · use voices you have rights to" }
            RowLayout { Layout.fillWidth: true; spacing: 8
                GalaxyButton { text: "󰑐"; hint: "Refresh"; onClicked: root.refresh() }
                GalaxyButton { text: "󰕾 Setup mic"; hint: "Create the aivoice null sink / virtual source only"; enabled: !actP.running; onClicked: root.sh(actP, ["setup"]) }
                GalaxyButton { text: "󰑓 Resample"; hint: "Regenerate the selected voice's reference sample (Piper TTS -> its RVC model)"; enabled: !!root.voice && !actP.running && !root.st.running; onClicked: { root.message = "Regenerating sample for " + root.voice + "…"; root.sh(actP, ["resynth", root.voice]) } }
                GalaxyButton { text: "󰇚 MeanVC2"; hint: "Download MeanVC2 runtime + models (multi-GB, needs PyTorch)"; enabled: !actP.running; onClicked: { root.message = "Downloading MeanVC2…"; root.sh(actP, ["models-install"]) } }
                Item { Layout.fillWidth: true }
                GalaxyButton { visible: !root.st.running; primary: true; text: "󰍬 Start"; enabled: !!root.voice && !actP.running; onClicked: root.sh(actP, ["start", root.voice, root.mode]) }
                GalaxyButton { visible: root.st.running; primary: true; text: "󰓛 Stop"; enabled: !actP.running; onClicked: root.sh(actP, ["stop"]) }
            }
        }
    }
}
