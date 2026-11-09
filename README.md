# AIML-SIGN

A two-way sign and speech conversation prototype. Sam signs into a webcam; Steve
speaks or types. Both review drafts before sending messages to the shared chat.
A deliberate hands-up gesture can confirm Sam's signs automatically.

## Architecture

```text
JPEG webcam frames → FastAPI WebSocket → MediaPipe Holistic
  → per-connection (64, 180) XYZ FIFO → hand-motion filter
  → shoulder normalization → PyTorch BiLSTM → debounced glosses
  → Groq model fallback → Gemini fallback → English sentence draft
  → button / hands-up confirmation → room-scoped shared chat

Browser microphone → Web Speech interim/final transcript → editable draft
  → Send to Chat → room broadcast → optional text-to-speech
```

The model has **19 classes**, input dimension 180, hidden size 128, two
bidirectional LSTM layers, dropout 0.4, and a 256→19 classifier. Its checkpoint,
label mapping and normalization configuration must be deployed together:

- `sign_language_backend/models/model_gislr_19class.pth`
- `sign_language_backend/models/labels_19class.json`
- `sign_language_backend/models/model_gislr_19class.config.json`

The selected GISLR corpus contains **7,327 samples**: 5,967 training / 1,360
validation, with 17 / 4 disjoint participants. Best recorded validation accuracy
is **71.10%**, at epoch 45 of the expanded 50-epoch run. This is an isolated-sign
research model; live-camera accuracy and continuous-language translation have
not been established. Only the 19 labels in the deployed mapping are recognized.

## Quickstart

Use **Python 3.11 or 3.12**. The legacy MediaPipe Holistic API is pinned to
0.10.21; Python 3.13 is not the server environment. From the workspace root:

```sh
python3.12 -m venv .venv-webcam
source .venv-webcam/bin/activate
python -m pip install -r sign_language_backend/requirements-server.txt
cp .env.example .env
```

Edit the **workspace-root** `.env` and replace the template values:

```dotenv
GROQ_API_KEY="your_groq_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
```

Keys stay on the server. Missing keys or provider failures produce a raw-gloss
sentence rather than disconnecting the client. Both the server and reference
CLI explicitly resolve root `.env`, independent of the working directory;
`override=True` replaces stale exported values.

```sh
python sign_language_backend/scripts/server_websocket.py
```

Open **http://localhost:8000** in Chrome. `/health` reports readiness after
checkpoint loading and the Holistic startup probe. Allow camera/microphone access
only when starting those controls. The three-column interface contains Sam's
sign controls, the shared chat, and Steve's voice controls. Speech recognition is
browser-dependent and may use the browser vendor's remote audio service.

Reference OpenCV client (server must already be running):

```sh
python sign_language_backend/scripts/inference_webcam.py
# Or open the unified browser UI:
python sign_language_backend/scripts/inference_webcam.py --browser
```

## Gesture controls and staging

1. Sign with visible shoulders and hands. Inference warms up over 64 received
   frames. Low motion or absent hands bypass classification.
2. A sign is accepted only when the same label has confidence **>0.70 for five
   consecutive inference frames**. Adjacent duplicate glosses are suppressed.
3. Translation previews remain in the editable sign draft. Idle time never sends
   a sentence. Steve's speech also remains in its draft until explicitly sent.
4. Click **Send to Chat**, or hold **both wrists above their corresponding
   shoulders for 15 consecutive processed frames** to translate and send Sam's
   signs. Hold duration depends on actual processing FPS (~0.625s at 24 FPS).
5. Lower at least one wrist below its shoulder to rearm the gesture. Holding hands
   up does not send repeatedly. Empty gloss lists do not invoke translation.

MediaPipe y increases downward. The condition is left wrist 15 y < shoulder 11 y
AND right wrist 16 y < shoulder 12 y, with valid pose visibility. Gesture frames
are excluded from classification. The gesture confirms server-accepted signs;
manual edits to a draft are submitted with the button.

## Frontend handoff

See **[FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md)** for the canonical
`ws://localhost:8000/ws/sign-language` schemas, one-frame-in-flight listener, and
`/ws/two-way` staging/broadcast extension used by the bundled UI.

HTTP CORS defaults to `allow_origins=["*"]`, **without credentials**, for frontend
development. WebSockets have a separate Origin check. Localhost ports 8000, 3000
and 5173 are allowed by default (127.0.0.1:8000 also). Configure exact origins for
other hosts; CORS does not authenticate a WebSocket:

```dotenv
SIGN_HTTP_ORIGINS=http://localhost:3000
SIGN_ALLOWED_ORIGINS=http://localhost:8000,http://localhost:3000
# Optional native-client Bearer handshake token:
# SIGN_WS_TOKEN=replace-with-a-long-random-value
```

Native clients may omit Origin. The bundled browser does not supply Bearer
headers: use an authenticated reverse proxy/session flow before a public browser
deployment. Sender names and conversation IDs are routing/UI roles, not identities.
Use one backend worker: room membership is process-local. Deploy TLS and enforce
connection/rate limits at the gateway. Current defaults bind **127.0.0.1**, not a
public interface. [Security audit and remaining limits](sign_language_backend/docs/SECURITY_AUDIT.md).

## Repository map

```text
AIML-SIGN/
├── README.md / FRONTEND_INTEGRATION.md
├── .env.example                    # .env is private and ignored
└── sign_language_backend/
    ├── scripts/
    │   ├── extract_gislr.py         # participant-isolated parquet extraction
    │   ├── train_gislr.py           # regularized BiLSTM training
    │   ├── server_websocket.py      # inference / translation / chat
    │   ├── inference_webcam.py      # reference native/browser launcher
    │   └── _shared/                # preprocessing + validated event contracts
    ├── frontend/                   # index.html, app.js, style.css
    ├── models/                     # three required inference assets
    ├── data/                       # cached tensors, provenance, run reports
    ├── metadata.json               # schema-v2 extraction mapping and splits
    ├── tests/                      # current GISLR/server/client regression suite
    ├── docs/                       # dataset, CLI, protocol and audit details
    └── context.md / progress.md     # canonical state and historical run log
```

Experimental pilot/placeholder scripts and their obsolete tests were removed.
The old `/webcam` URL redirects to the unified app; its duplicate UI was removed.

## Data and training

[Dataset/training guide](sign_language_backend/docs/DATASET.md).
Install `requirements-gislr.txt` in a data environment, or install the backend's
combined `requirements.txt` in a compatible Python 3.12 environment. From
`sign_language_backend`:

```sh
python scripts/extract_gislr.py --dataset-dir data/raw_shards/gislr
python scripts/train_gislr.py --epochs 50
```

For downloads use `--download --prompt-token`, with Kaggle competition access.
Credentials are prompted/read from environment, never stored in source. Output
arrays are raw finite float32 `(64,180)`: pose indices 0–17 (54 coordinates),
left hand 21×3, right hand 21×3; missing joints are zero-filled. Shoulder
normalization is shared by training and inference, not stored in the NPY files.

**Do not commit dataset caches.** `data/processed/` is about 343 MiB and
`data/raw_shards/` about 4 GiB. Both, run backup directories, virtual environments,
`.env`/`*.env`, Python caches and `.DS_Store` are ignored. They remain on disk.
Model weights, label/config JSON, metadata and small reports are preserved for
handoff; model/data distribution still depends on the source dataset's terms.

## Verification

From `sign_language_backend`, with runtime and data requirements installed:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
node --test tests/test_two_way_browser.cjs
```

In the existing workspace, extraction tests use `../.venv/bin/python`; runtime,
training and webcam tests use `../.venv-webcam/bin/python`. Real Holistic smoke
tests require a graphical session. Provider calls and hardware inputs are mocked;
passing tests does not verify live translation, microphone or signing accuracy.
