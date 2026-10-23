import QtQuick
import QtQuick.Layouts

// Spaced small-caps section heading, the rice's "✦  O R B I T" style.
Text {
    property string title: ""
    Layout.fillWidth: true
    text: "✦  " + title.toUpperCase().split("").join(" ")
    font.family: Galaxy.font
    font.pixelSize: 11
    color: Galaxy.lavender
    elide: Text.ElideRight
}
