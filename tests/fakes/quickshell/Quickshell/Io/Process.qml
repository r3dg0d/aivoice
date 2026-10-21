import QtQuick
// Test stub: instead of running run.sh, answer with canned output for its subcommand
// (files written by the test from the real `aivoice` CLI), and log every command line.
QtObject {
    id: p
    property var command: []
    property var environment: ({})
    property bool running: false
    property var stdout: null
    onRunningChanged: if (running) Qt.callLater(p.finish)
    function finish() {
        const sub = p.command.length > 2 ? p.command[2] : ""
        console.log("STUBCMD " + JSON.stringify(p.command.slice(2)))
        const xhr = new XMLHttpRequest()
        xhr.open("GET", Qt.resolvedUrl("file://" + Qt.application.arguments[1] + "/" + sub + ".out"), false)
        let out = ""
        try { xhr.send(); out = xhr.responseText || "" } catch (e) { out = "" }
        if (p.stdout) { p.stdout.text = out; p.stdout.streamFinished() }
        p.running = false
    }
}
