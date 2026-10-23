import QtQuick
import QtQuick.Controls

Switch {
    id: sw
    font.family: Galaxy.font
    font.pixelSize: 13
    indicator: Rectangle {
        implicitWidth: 38; implicitHeight: 20; radius: 10
        x: sw.leftPadding; y: parent.height / 2 - height / 2
        color: sw.checked ? Galaxy.violet : Galaxy.base
        border.color: sw.checked ? Galaxy.lavender : Galaxy.line
        Rectangle {
            x: sw.checked ? parent.width - width - 3 : 3; y: 3; width: 14; height: 14; radius: 7
            color: sw.checked ? Galaxy.ink : Galaxy.dim
            Behavior on x { NumberAnimation { duration: Galaxy.fast } }
        }
    }
    contentItem: Text { leftPadding: sw.indicator.width + 10; text: sw.text; font: sw.font; color: sw.enabled ? Galaxy.ink : Galaxy.dim; verticalAlignment: Text.AlignVCenter }
}
