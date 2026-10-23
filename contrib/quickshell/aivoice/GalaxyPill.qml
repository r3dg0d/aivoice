import QtQuick
import QtQuick.Layouts

// Status pill: state is "ok" | "warn" | "err" | "busy" | "idle". Dot pulses only for ok/busy.
Rectangle {
    id: pill
    property string state_: "idle"
    property string text: ""
    readonly property color fg: state_ === "ok" ? Galaxy.ok : state_ === "warn" ? Galaxy.warn : state_ === "err" ? Galaxy.err : state_ === "busy" ? Galaxy.purple : Galaxy.dim
    implicitWidth: row.implicitWidth + 18
    implicitHeight: 24
    radius: 12
    color: state_ === "ok" ? Galaxy.okBg : state_ === "warn" ? Galaxy.warnBg : state_ === "err" ? Galaxy.errBg : Galaxy.raised
    border.color: state_ === "idle" ? Galaxy.edge : fg
    RowLayout {
        id: row
        anchors.centerIn: parent
        spacing: 6
        Rectangle {
            width: 8; height: 8; radius: 4; color: pill.fg
            SequentialAnimation on opacity {
                running: pill.state_ === "ok" || pill.state_ === "busy"; loops: Animation.Infinite
                NumberAnimation { to: 0.3; duration: 800 } NumberAnimation { to: 1; duration: 800 }
            }
        }
        Text { font.family: Galaxy.font; font.pixelSize: 10; color: Galaxy.ink; text: pill.text }
    }
}
