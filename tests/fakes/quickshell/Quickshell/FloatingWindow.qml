import QtQuick
import QtQuick.Window
Window {
    property int implicitWidth: 600
    property int implicitHeight: 800
    property size minimumSize
    signal closed()
    width: implicitWidth; height: implicitHeight
    function startSystemMove() {}
}
