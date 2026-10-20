import QtQuick
import QtQuick.Controls

Button {
    id: control
    property bool primary: false
    property bool selected: false
    property string hint: ""
    ToolTip.visible: hovered && hint.length > 0
    ToolTip.text: hint
    ToolTip.delay: 400
    implicitWidth: Math.max(40, label.implicitWidth + 28)
    implicitHeight: 36
    hoverEnabled: true
    padding: 8
    font.family: "Cellforge"
    font.pixelSize: 13
    contentItem: Text {
        id: label
        text: control.text
        font: control.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: !control.enabled ? "#5d5878" : control.primary ? "#07060d" : "#d4b8ff"
    }
    background: Rectangle {
        radius: control.primary ? 16 : 10
        color: !control.enabled ? "#0b0914" : control.down ? "#7c4dff" : control.primary ? (control.hovered ? "#d4b8ff" : "#b46bff") : (control.hovered || control.selected) ? "#1a1630" : "#0f0d1a"
        border.width: 1
        border.color: control.activeFocus ? "#d4b8ff" : (control.hovered || control.selected) ? "#b46bff" : "#26223a"
        Behavior on color { ColorAnimation { duration: 120 } }
        Behavior on border.color { ColorAnimation { duration: 120 } }
    }
}
