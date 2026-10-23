import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ComboBox {
    id: box
    Layout.fillWidth: true
    implicitHeight: 36
    font.family: Galaxy.font
    font.pixelSize: 13
    contentItem: Text {
        leftPadding: 12; rightPadding: 28
        text: box.displayText; font: box.font; elide: Text.ElideRight
        color: box.enabled ? Galaxy.ink : Galaxy.dim; verticalAlignment: Text.AlignVCenter
    }
    indicator: Text { x: box.width - width - 12; anchors.verticalCenter: parent.verticalCenter; text: "▾"; color: Galaxy.purple; font.pixelSize: 12 }
    background: Rectangle {
        radius: Galaxy.radiusControl
        color: box.hovered ? Galaxy.raised : Galaxy.surface
        border.color: box.activeFocus || box.popup.visible ? Galaxy.purple : Galaxy.line
    }
    delegate: ItemDelegate {
        required property var modelData
        required property int index
        width: box.width
        highlighted: box.highlightedIndex === index
        contentItem: Text { text: modelData; font: box.font; color: highlighted ? Galaxy.void_ : Galaxy.lavender; verticalAlignment: Text.AlignVCenter }
        background: Rectangle { radius: 8; color: highlighted ? Galaxy.purple : "transparent" }
    }
    popup: Popup {
        y: box.height + 4; width: box.width; padding: 4
        implicitHeight: Math.min(contentItem.implicitHeight + 8, 320)
        contentItem: ListView { clip: true; implicitHeight: contentHeight; model: box.popup.visible ? box.delegateModel : null; currentIndex: box.highlightedIndex }
        background: Rectangle { radius: 12; color: Galaxy.surface; border.color: Galaxy.edge }
    }
}
