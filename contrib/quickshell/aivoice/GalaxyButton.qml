import QtQuick
import QtQuick.Controls

// Rice GalaxyButton (galaxy/GalaxyButton.qml) on Galaxy tokens. primary = filled lavender pill.
Button {
    id: control
    property bool primary: false
    property bool selected: false
    property bool danger: false
    property string hint: ""
    ToolTip.visible: hovered && hint.length > 0
    ToolTip.text: hint
    ToolTip.delay: 400
    implicitWidth: Math.max(40, label.implicitWidth + 28)
    implicitHeight: 36
    hoverEnabled: true
    padding: 8
    font.family: Galaxy.font
    font.pixelSize: 13
    contentItem: Text {
        id: label
        text: control.text
        font: control.font
        elide: Text.ElideRight
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: !control.enabled ? Galaxy.dim : control.primary ? Galaxy.void_ : control.danger ? Galaxy.err : Galaxy.lavender
    }
    background: Rectangle {
        radius: control.primary ? 16 : Galaxy.radiusControl
        color: !control.enabled ? Galaxy.base : control.down ? Galaxy.violet
             : control.primary ? (control.hovered ? Galaxy.lavender : Galaxy.purple)
             : (control.hovered || control.selected) ? Galaxy.raised : Galaxy.surface
        border.width: 1
        border.color: control.activeFocus ? Galaxy.lavender : control.danger && control.hovered ? Galaxy.err
                    : (control.hovered || control.selected) ? Galaxy.purple : Galaxy.line
        Behavior on color { ColorAnimation { duration: Galaxy.fast } }
        Behavior on border.color { ColorAnimation { duration: Galaxy.fast } }
    }
}
