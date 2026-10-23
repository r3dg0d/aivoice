import QtQuick

// Glass card: the rice's panel gradient with a thin luminous border. The gradient is painted by a Canvas
// clipped to the rounded rect (once per resize): a gradient on a rounded Rectangle is interpolated per
// triangle by the scene graph and shows a seam along the diagonal on the GPU renderer.
Item {
    id: panel
    property bool hot: false
    property real radius: Galaxy.radiusPanel
    property color stopTop: Galaxy.glassTop
    property color stopMid: Galaxy.glassMid
    property color stopBottom: Galaxy.glassBottom
    default property alias content: holder.data
    Canvas {
        id: bg
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            const c = getContext("2d"); c.reset()
            if (width < 2 || height < 2) return
            c.beginPath(); c.roundedRect(0.5, 0.5, width - 1, height - 1, panel.radius, panel.radius)
            const g = c.createLinearGradient(0, 0, 0, height)
            g.addColorStop(0, panel.stopTop); g.addColorStop(0.45, panel.stopMid); g.addColorStop(1, panel.stopBottom)
            c.fillStyle = g; c.fill()
        }
    }
    Rectangle {
        anchors.fill: parent; radius: panel.radius; color: "transparent"; border.width: 1
        border.color: panel.hot ? Galaxy.edgeHot : Galaxy.edge
        Behavior on border.color { ColorAnimation { duration: Galaxy.slow } }
    }
    Item { id: holder; anchors.fill: parent }
}
