import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TextField {
    Layout.fillWidth: true
    color: Galaxy.ink
    selectByMouse: true
    padding: 10
    font.family: Galaxy.font
    font.pixelSize: 13
    placeholderTextColor: Galaxy.dim
    selectionColor: Galaxy.violet
    background: Rectangle {
        radius: Galaxy.radiusControl
        color: Galaxy.base
        border.color: parent.activeFocus ? Galaxy.purple : Galaxy.line
        Behavior on border.color { ColorAnimation { duration: Galaxy.fast } }
    }
}
