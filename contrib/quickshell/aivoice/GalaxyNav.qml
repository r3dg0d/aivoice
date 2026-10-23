import QtQuick
import QtQuick.Layouts

// Sidebar entry: icon glyph + label, glowing left rail when current.
Rectangle {
    id: nav
    property string icon: ""
    property string text: ""
    property bool current: false
    property bool compact: false
    signal clicked()
    Layout.fillWidth: true
    implicitHeight: 38
    radius: Galaxy.radiusControl
    color: current ? Galaxy.raised : area.containsMouse ? Galaxy.surface : "transparent"
    border.color: current ? Galaxy.edge : "transparent"
    Behavior on color { ColorAnimation { duration: Galaxy.fast } }
    Rectangle { visible: nav.current; x: 0; y: 8; width: 3; height: parent.height - 16; radius: 2; color: Galaxy.purple }
    RowLayout {
        anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 10; spacing: 10
        Text { font.family: Galaxy.icons; text: nav.icon; color: nav.current ? Galaxy.purple : Galaxy.muted; font.pixelSize: 15 }
        Text { visible: !nav.compact; Layout.fillWidth: true; font.family: Galaxy.font; text: nav.text; color: nav.current ? Galaxy.ink : Galaxy.muted; font.pixelSize: 13; elide: Text.ElideRight }
    }
    MouseArea { id: area; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: nav.clicked() }
}
