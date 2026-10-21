pragma Singleton
import QtQuick
// Test stub of Quickshell's global: env() and shellDir only.
QtObject {
    property string shellDir: "/stub/shell"
    function env(name) { return name === "HOME" ? "/stub/home" : "stub" }
}
