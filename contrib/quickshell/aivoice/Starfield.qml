import QtQuick

// Static nebula + star backdrop, painted once per resize (no per-frame redraw, no shaders).
Canvas {
    id: sky
    property int seed: 7
    property real density: 1.0
    property real cornerRadius: 0
    property bool paintBase: false  // window background: vertical #120c24 -> void gradient (seam-free here)
    renderStrategy: Canvas.Cooperative
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onCornerRadiusChanged: requestPaint()
    onPaint: {
        const c = getContext("2d"); c.reset()
        if (cornerRadius > 0) { c.beginPath(); c.roundedRect(0, 0, width, height, cornerRadius, cornerRadius); c.clip() }
        if (paintBase) {
            const bg = c.createLinearGradient(0, 0, 0, height)
            bg.addColorStop(0, "#120c24"); bg.addColorStop(1, Galaxy.void_)
            c.fillStyle = bg; c.fillRect(0, 0, width, height)
        }
        let s = seed
        const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 }
        const neb = [[0.18, 0.12, 0.55, "rgba(124,77,255,0.10)"], [0.86, 0.78, 0.6, "rgba(180,107,255,0.08)"], [0.7, 0.2, 0.35, "rgba(255,107,213,0.045)"]]
        for (const [fx, fy, fr, col] of neb) {
            const r = Math.max(width, height) * fr
            const g = c.createRadialGradient(width * fx, height * fy, 0, width * fx, height * fy, r)
            g.addColorStop(0, col); g.addColorStop(1, "rgba(0,0,0,0)")
            c.fillStyle = g; c.fillRect(0, 0, width, height)
        }
        const n = Math.round(width * height / 9000 * density)
        for (let i = 0; i < n; ++i) {
            const x = rnd() * width, y = rnd() * height, big = rnd() > 0.93
            c.fillStyle = big ? "rgba(212,184,255,0.55)" : "rgba(242,240,255," + (0.12 + rnd() * 0.25).toFixed(2) + ")"
            c.fillRect(x, y, big ? 2 : 1, big ? 2 : 1)
        }
    }
}
