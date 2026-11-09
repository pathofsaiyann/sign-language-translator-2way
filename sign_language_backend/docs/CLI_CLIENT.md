# Reference webcam client

Run from the workspace root with the server already listening:

```sh
.venv-webcam/bin/python sign_language_backend/scripts/inference_webcam.py
.venv-webcam/bin/python sign_language_backend/scripts/inference_webcam.py --browser
```

The native mode uses OpenCV camera 0, unmirrored JPEG/Base64, at most 24 FPS and
one frame in flight. GUI remains on the main thread; async networking/reconnect
runs in a background thread. Only the latest camera image is retained. Press q
or close the window to stop; camera/socket/worker resources are released.

Defaults: --url ws://localhost:8000/ws/sign-language, --camera 0, --fps 24,
--jpeg-quality 80, --max-side 640, --response-timeout 30. --origin supplies an
Origin header. SIGN_WS_TOKEN supplies native Bearer auth. Root .env explicitly
loads with override=True; no key values are logged. The client consumes canonical
and legacy event fields and keeps waiting through SUBMITTING until completion.

The CLI has no local model or MediaPipe. Its overlay shows sentence, accumulated
glosses and gesture instruction. --browser opens the unified root UI; --ui-url can
select a different frontend. Browser microphone/TTS are provided by that UI.

A separate lightweight environment can use requirements-client.txt; do not install
opencv-python beside opencv-contrib-python in the server environment. Full setup
and the protocol are in the [README](../../README.md) and
[frontend handoff](../../FRONTEND_INTEGRATION.md).
