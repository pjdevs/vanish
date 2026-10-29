# VanishCam

Webcam vanishing effects you can use in Teams, Zoom or Meet through the OBS virtual camera.

This is a port of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). It runs on macOS, Linux and Windows, uses uv and up-to-date dependencies, fixes a memory leak that crashed the app after about half an hour, and adds a Goku teleport mode.

## Usage

You need [uv](https://docs.astral.sh/uv/) and [OBS Studio](https://obsproject.com/) 30 or later. Start the OBS virtual camera once so it gets installed (on macOS, approve it in System Settings), then stop it and quit OBS.

```sh
uv run vanish_cam.py goku    # or normal, naruto
```

Step out of the frame for the first 4 seconds while the background is captured, then pick OBS Virtual Camera in your call app. `uv run vanish_cam.py --help` lists the gestures, keys and options.

## License

MIT for the changes in this port. The original code belongs to Nicovic17 and has no license; see [LICENSE.md](LICENSE.md).
