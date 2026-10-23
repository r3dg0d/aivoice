import QtQuick
import QtQuick.Layouts

// Metric tile. value is shown as given; pass "—" when the backend did not report it.
Rectangle {
    property string label: ""
    property string value: "—"
    property string unit: ""
    property color tint: Galaxy.lavender
    Layout.fillWidth: true
    implicitWidth: 120
    implicitHeight: 62
    radius: Galaxy.radiusTile
    color: Galaxy.tile
    border.color: Galaxy.line
    Column {
        anchors.centerIn: parent
        width: parent.width - 16
        spacing: 6
        Text { width: parent.width; horizontalAlignment: Text.AlignHCenter; font.family: Galaxy.font; text: label.toUpperCase(); color: Galaxy.label; font.pixelSize: 9; font.letterSpacing: 1.4; elide: Text.ElideRight }
        Text { width: parent.width; horizontalAlignment: Text.AlignHCenter; font.family: Galaxy.mono; text: value + (value !== "—" && unit ? unit : ""); color: value === "—" ? Galaxy.dim : tint; font.pixelSize: 16; elide: Text.ElideRight }
    }
}
