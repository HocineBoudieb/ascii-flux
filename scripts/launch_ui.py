"""One-click local GPU interface. Reuse an existing server without resetting it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8766"


def ready():
    try:
        with urllib.request.urlopen(URL+"/config", timeout=1) as response:
            info = json.load(response)
        if info.get("application") != "ascii-flux-native":
            raise RuntimeError("Le port 8766 est occupé par une autre application.")
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Start without opening a browser")
    args = parser.parse_args()
    if not ready():
        checkpoint = ROOT/"data/runs/native-million-001/best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("Le modèle entraîné est introuvable : "+str(checkpoint))
        logs = ROOT/"outputs"
        logs.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "ascii_flux", "serve", "--checkpoint", str(checkpoint),
                   "--port", "8766", "--device", "cuda", "--debounce-ms", "0"]
        options = {"cwd": ROOT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        with (logs/"interface.log").open("a", encoding="utf-8") as output, (logs/"interface-errors.log").open("a", encoding="utf-8") as errors:
            process = subprocess.Popen(command, stdout=output, stderr=errors, **options)
        (logs/"interface-launch.json").write_text(json.dumps({"pid": process.pid, "url": URL, "checkpoint": str(checkpoint)}))
        deadline = time.monotonic()+120
        while not ready():
            if process.poll() is not None:
                raise RuntimeError("Le modèle n'a pas démarré. Voir outputs/interface-errors.log.")
            if time.monotonic() > deadline:
                raise TimeoutError("Le modèle met du temps à démarrer. Voir outputs/interface.log.")
            time.sleep(.5)
    if not args.no_open:
        webbrowser.open(URL)
    if sys.stdout is not None:
        print(URL, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if sys.stdout is None and os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(error), "ASCII Flux", 0x10)
        else:
            raise
