"""Preload the Naruto MP3 and play it without blocking video frames."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


SOUND_PATH = Path(__file__).resolve().parent / "assets" / "vanish.MP3"


class VanishSound:
    def __init__(self, path: Path = SOUND_PATH):
        if not path.is_file():
            raise RuntimeError(f"Naruto audio not found: {path}")
        self.path = path

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.process = None
        self.ffplay = None
        if platform.system() == "Darwin":
            # afplay ships with macOS; pygame would bring a second copy of SDL
            # that conflicts with OpenCV's.
            self.ffplay = shutil.which("afplay")
            if self.ffplay is None:
                raise RuntimeError("afplay not found: cannot play vanish.MP3.")
            return
        try:
            from pygame import mixer
        except ImportError:
            self.ffplay = shutil.which("ffplay")
            if self.ffplay is None:
                raise RuntimeError("Install pygame or ffplay to play vanish.MP3.")
            return

        self.mixer = mixer
        self.owns_mixer = mixer.get_init() is None
        try:
            if self.owns_mixer:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            self.sound = mixer.Sound(str(path))  # Decode once, before opening the webcam.
        except Exception as error:
            if self.owns_mixer:
                mixer.quit()
            raise RuntimeError(f"Could not open the Naruto audio: {error}") from error

    def play(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            try:
                command = [self.ffplay, str(self.path)]
                if Path(self.ffplay).name != "afplay":
                    command[1:1] = ["-nodisp", "-autoexit", "-loglevel", "quiet", "-i"]
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                raise RuntimeError(f"Could not start the Naruto audio: {error}") from error
            return
        self.sound.stop()
        if self.sound.play() is None:
            raise RuntimeError("Could not start the Naruto audio: no free channel.")

    def close(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if self.owns_mixer:
            self.mixer.quit()
            self.owns_mixer = False
