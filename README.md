# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Getting started (macOS)

Requirements: [uv](https://docs.astral.sh/uv/) and [OBS Studio](https://obsproject.com/) 30 or later.

1. One-time OBS setup: open OBS, click **Start Virtual Camera**, approve **OBS Virtual Camera** in System Settings → General → Login Items & Extensions → Camera Extensions, then click **Stop Virtual Camera** and quit OBS.
2. Run `uv run vanish_cam.py <mode>`, with `normal`, `naruto` or `goku` (`uv run vanish_cam.py --help` describes them). The first run installs the dependencies and downloads the MediaPipe models. macOS asks for camera access for the app hosting your terminal (Terminal, iTerm, or Claude if you run it from there); grant it in System Settings → Privacy & Security → Camera if the prompt does not show.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.
4. In Teams (or Zoom, Meet, ...), pick **OBS Virtual Camera** as your camera. Restart Teams if it does not show up.

MediaPipe runs on the CPU (its macOS GPU path leaks memory until it crashes). On macOS the sound plays through `afplay`.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Goku Teleport:** Hold your index and middle fingers straight and together against your forehead (other fingers folded) for 1 second. Glowing vertical comic-style lines run across your silhouette, there is a flash and you are gone; the lines linger briefly. To reappear, lower your hand, then hold the pose again for 1 second: the effect plays in reverse. Breaking the pose resets the countdown, which the terminal shows. `V` and `R` are disabled in this mode. The sound is `assets/teleport.wav`; drop an `assets/teleport.MP3` in to use your own. The forehead comes from `face_landmarker.task` and the fingers from `hand_landmarker.task`.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

## Virtual camera and OBS

The app sends its video to **OBS Virtual Camera** automatically when it is available; the terminal shows the device name. Use `--virtual-camera` to require it or `--no-virtual-camera` for the local preview only. OBS's own virtual camera must be stopped while this app is producing its output. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear (not in Goku mode).
- `V`: manually start the selected mode's effect (not in Goku mode).
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

You can change the camera, resolution, and maximum virtual-output FPS:

```sh
uv run vanish_cam.py goku --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

