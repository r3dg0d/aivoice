import QtQuick
import QtQuick.Layouts

// History sparkline with a soft violet fill. Repaints only when data changes.
Canvas {
    id: spark
    property var data: []
    property int capacity: 60
    property color stroke: Galaxy.purple
    property string label: ""
    property string current: ""
    Layout.fillWidth: true
    implicitHeight: 54
    renderStrategy: Canvas.Cooperative
    onDataChanged: requestPaint()
    onWidthChanged: requestPaint()
    onPaint: {
        const c = getContext("2d"); c.reset()
        const d = data || []; if (d.length < 2) return
        const max = Math.max.apply(null, d) * 1.15 || 1
        const top = 16
        c.beginPath()
        for (let i = 0; i < d.length; ++i) {
            const x = i / (capacity - 1) * width, y = height - 3 - d[i] / max * (height - top - 3)
            i ? c.lineTo(x, y) : c.moveTo(x, y)
        }
        c.strokeStyle = stroke; c.lineWidth = 2; c.stroke()
        c.lineTo((d.length - 1) / (capacity - 1) * width, height); c.lineTo(0, height); c.closePath()
        const g = c.createLinearGradient(0, top, 0, height)
        g.addColorStop(0, Qt.rgba(stroke.r, stroke.g, stroke.b, 0.32)); g.addColorStop(1, "transparent")
        c.fillStyle = g; c.fill()
    }
    Text { x: 0; y: 0; font.family: Galaxy.font; text: spark.label; color: Galaxy.label; font.pixelSize: 9; font.letterSpacing: 1.2 }
    Text { x: spark.width - width; y: 0; font.family: Galaxy.mono; text: spark.current; color: spark.stroke; font.pixelSize: 10 }
}
