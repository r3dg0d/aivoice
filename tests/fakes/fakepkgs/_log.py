import json
import os


def log(event, **kw):
    path = os.environ.get("AIVOICE_FAKE_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"event": event, **kw}, default=str) + "\n")
