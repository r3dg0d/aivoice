import QtQuick
import QtQuick.Controls

// Rice GalaxySlider (galaxy-hub/GalaxySlider.qml) on Galaxy tokens, for any range.
Slider {
    id: s
    property bool dim: false
    implicitHeight: 28
    background: Rectangle {
        x: s.leftPadding; y: s.topPadding + s.availableHeight / 2 - height / 2
        width: s.availableWidth; height: 8; radius: 4; color: Galaxy.line
        Rectangle {
            width: s.visualPosition * parent.width; height: parent.height; radius: 4
            opacity: s.dim || !s.enabled ? 0.35 : 1
            gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0; color: Galaxy.violet } GradientStop { position: 1; color: Galaxy.lavender } }
        }
    }
    handle: Rectangle {
        x: s.leftPadding + s.visualPosition * (s.availableWidth - width); y: s.topPadding + s.availableHeight / 2 - height / 2
        width: 18; height: 18; radius: 9
        color: !s.enabled ? Galaxy.dim : s.pressed ? Galaxy.lavender : Galaxy.ink; border.color: Galaxy.purple; border.width: 2
    }
    WheelHandler {
        enabled: s.enabled
        onWheel: ev => {
            const step = s.stepSize > 0 ? s.stepSize : (s.to - s.from) / 20
            s.value = Math.max(s.from, Math.min(s.to, s.value + (ev.angleDelta.y > 0 ? step : -step)))
            s.moved()
        }
    }
}
