import QtQuick
import QtQuick.Layouts

// Download / job progress; indeterminate shows a drifting violet comet.
Rectangle {
    id: bar
    property real value: 0
    property bool indeterminate: false
    Layout.fillWidth: true
    implicitHeight: 6
    radius: 3
    color: Galaxy.line
    clip: true
    Rectangle {
        height: parent.height; radius: 3
        width: bar.indeterminate ? parent.width * 0.25 : parent.width * Math.max(0, Math.min(1, bar.value))
        gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0; color: Galaxy.violet } GradientStop { position: 1; color: Galaxy.lavender } }
        NumberAnimation on x { running: bar.indeterminate && bar.visible; from: -bar.width * 0.25; to: bar.width; duration: 1400; loops: Animation.Infinite }
        Behavior on width { enabled: !bar.indeterminate; NumberAnimation { duration: Galaxy.slow } }
    }
}
