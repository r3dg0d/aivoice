"""Load the QuickShell widget against stub Quickshell modules (offscreen) and report QML warnings.

argv: <canned-output dir> <shell.qml>; prints STUBCMD lines and WARN lines, exits 0.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QML_XHR_ALLOW_FILE_READ"] = "1"
from PySide6.QtCore import QTimer, QUrl, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402

msgs = []


def handler(mode, ctx, msg):
    msgs.append(msg)
    print(("STUBCMD " if msg.startswith("STUBCMD") else "WARN ") + msg.replace("STUBCMD ", ""), flush=True)


qInstallMessageHandler(handler)
app = QGuiApplication([sys.argv[0], sys.argv[1]])
eng = QQmlApplicationEngine()
eng.addImportPath(os.path.join(os.path.dirname(__file__), "fakes", "quickshell"))
eng.load(QUrl.fromLocalFile(sys.argv[2]))
if not eng.rootObjects():
    print("WARN load failed", flush=True)
    sys.exit(0)
root = eng.rootObjects()[0]
steps = [("major", "s2s"), ("major", "mod"), ("advanced", True), ("major", "s2s"), ("major", "vc"), ("showExperimental", True), ("vcEngine", "xvc")]


def step():
    if steps:
        k, v = steps.pop(0)
        root.setProperty(k, v)
        QTimer.singleShot(2300 if k == "major" and v == "s2s" else 300, step)
    else:
        print("PROPS " + str(root.property("major")) + " " + str(root.property("vcEngine")), flush=True)
        app.quit()


QTimer.singleShot(800, step)
QTimer.singleShot(8000, app.quit)
app.exec()
