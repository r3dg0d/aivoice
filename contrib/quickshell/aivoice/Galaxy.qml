pragma Singleton
import QtQuick
import Quickshell

// Galaxy design tokens, taken from the installed rice (~/.config/quickshell/galaxy). The same file
// ships with aivoice's widget; see docs/galaxy-theme.md. Keep the two copies identical.
Singleton {
    // surfaces, darkest to raised
    readonly property color void_: "#07060d"
    readonly property color base: "#0b0914"
    readonly property color surface: "#0f0d1a"
    readonly property color raised: "#1a1630"
    readonly property color tile: "#120f22"
    // lines
    readonly property color line: "#26223a"
    readonly property color edge: "#3a2f5c"
    readonly property color edgeHot: "#5b3d99"
    // accents
    readonly property color violet: "#7c4dff"
    readonly property color purple: "#b46bff"
    readonly property color lavender: "#d4b8ff"
    readonly property color magenta: "#ff6bd5"
    // text
    readonly property color ink: "#f2f0ff"
    readonly property color muted: "#9a96b3"
    readonly property color label: "#8a84a8"
    readonly property color dim: "#5d5878"
    // status
    readonly property color ok: "#7dffa8"
    readonly property color okBg: "#1d3a2a"
    readonly property color warn: "#ffd479"
    readonly property color warnBg: "#3a2d12"
    readonly property color err: "#ff7a9a"
    readonly property color errBg: "#3a1424"
    // panel gradient (glass): top, middle, bottom
    readonly property color glassTop: "#fa1a1630"
    readonly property color glassMid: "#fa0f0d1a"
    readonly property color glassBottom: "#fc07060d"
    // type
    readonly property string font: "Cellforge"
    readonly property string mono: "Zerodot"
    readonly property string icons: "JetBrainsMono Nerd Font"
    // geometry
    readonly property int radiusWindow: 22
    readonly property int radiusPanel: 18
    readonly property int radiusTile: 14
    readonly property int radiusControl: 10
    readonly property int gap: 10
    readonly property int pad: 16
    // motion
    readonly property int fast: 120
    readonly property int slow: 260
}
