# Project context

## Architecture and continuity
- PyTorch 19-class GISLR BiLSTM with implemented FastAPI WebSocket inference backend.
- Read context.md and progress.md before work; update both after successful runs.
- Preserve data/, data/processed/, data/raw_shards/, models/, and scripts/.

## Pivot status
- SignVerse pilot retired. Clean slate completed before the current GISLR run;
  processed/raw directories now contain GISLR data (see latest metrics below).
- Removed data/pilot/, pilot weights, label mapping, training history and split,
  and scripts/stream_filter_signverse.py. Reset metadata.json to {}.
- Historical subtitle audits and Top 30 coverage report remain under data/.
- Other extraction/training scripts and tests remain legacy pilot code and may
  refer to deleted data or assets; adapt them once the new source is selected.

## Current inference infrastructure
- Unified app: frontend/index.html is served at http://localhost:8000; /ws/two-way
  carries private sign drafts and room-scoped confirmed Sam/Steve messages.
  Three-column DM layout: Sam controls / shared chat / Steve controls; only
  confirmed messages enter chat (Sam left/green, Steve right/purple).
- scripts/inference_webcam.py is now a lightweight OpenCV WebSocket client,
  default ws://localhost:8000/ws/sign-language. It captures JPEG frames and displays
  server translations/glosses/status; no local weights, model or MediaPipe.
- scripts/server_websocket.py owns native MediaPipe pose 0–17, left/right hand
  0–20, raw XYZ float32 (1,64,180) FIFO and GISLR BiLSTM inference. Shared shoulder
  normalization is applied at the model boundary; checkpoint config is required.
- Server: hand motion variance .015 gate, same top label >.70 for five windows,
  adjacent-duplicate-suppressed gloss accumulation, submission after 15 consecutive
  frames with both wrists above shoulders (no idle-timer boundary). Translation:
  AsyncGroq llama3-8b-8192 → llama-3.1-8b-instant → mixtral-8x7b-32768, then
  off-loop Gemini gemini-2.0-flash, then raw joined glosses only if all fail.
  Root .env refreshes with override=True at startup and per build_sentence call.
  Each Groq attempt has 3s SDK / 4s async limits; Gemini 12s HTTP / 12s async.
  Total await budgets 24s fit within the clients' 30s response limits.
  Live provider availability for the current models has not been verified.
- Full target dataset: 7,327 finite float32 (64,180) tensors; 5,967 train /
  1,360 validation; 17/4 disjoint signers. Latest 50-epoch best: 71.10% val at epoch 45.
- Metadata retains original 30 indices; model labels remap 19 present classes
  to contiguous indices 0–18. Weights: models/model_gislr_19class.pth;
  labels: models/labels_19class.json (server only).
- Webcam GUI stays on the main thread; async WebSocket I/O/reconnection uses a
  background thread and newest-frame-only buffering. Browser mode at /webcam
  retains legacy live subtitles; --browser now opens the unified root app with
  staged speech/sign drafts, send controls and optional TTS. Live hardware remains untested.

## 2026-11-07 — GISLR extractor implemented; live extraction blocked
- Added scripts/extract_gislr.py, requirements-gislr.txt, GISLR.md and parquet tests.
- Installed pandas, pyarrow and kagglehub in the project .venv.
- Two tests passed: explicit frame/landmark ordering, missing/NaN landmarks,
  padding/subsampling, real parquet parsing, 40 synthetic samples, and repeat runs.
  Synthetic outputs were isolated in temporary directories, not project data/.
- Live --download attempt failed before train.csv: Kaggle UnauthenticatedError.
  Real downloaded/extracted samples: 0; metadata.json remains {}. No training run.
- Dataset locations: data/raw_shards/gislr/ and data/processed/{class}/{id}.npy.
- New feature contract: MediaPipe pose 0–17, left hand 0–20, right hand 0–20,
  XYZ (not confidence), finite float32 (64,180), uniform subsampling/trailing zeros.
  Legacy webcam adapter must be updated before using a model trained on this data.
- Fixed 30-label indices in requested order; explicit thankyou/minemy aliases.
  GISLR lacks some requested words; missing classes are reported without substitutes.
- Next: authenticate Kaggle and accept asl-signs rules, or supply local train.csv
  and parquet files; run command in GISLR.md. Runtime writes actual counts,
  provenance, label mapping, metadata and tracking updates; exit 2 flags shortfalls.

## 2026-11-09 — GISLR authentication and exact vocabulary update
- Replaced vocabulary with the newly requested 30 exact strings, in requested
  order (hello=0, bye=1, thankyou=2, ..., man=29); removed all aliases.
- Added process-only token prompt/KAGGLE_API_TOKEN, legacy env credentials,
  and ~/.kaggle/kaggle.json (including KAGGLE_CONFIG_DIR) fallback handling.
  Credentials are not saved in source or dataset artifacts. Download failures
  exit 1 with sanitized error type/status and explicit authentication/CLI steps.
- Four tests passed, including mocked authentication fallbacks, sanitized errors,
  parquet ordering/zero filling/temporal formatting, repeatability and a complete
  synthetic 30-class x 35-sample extraction. Synthetic data was temporary only.
- Ran live extraction using the supplied token: Kaggle returned HTTP 403 for
  train.csv. No legacy fallback credentials were configured. No real parquet
  downloaded; no real tensors extracted; metadata.json remains {}.
- Final real sample counts for this blocked attempt: {"hello": 0, "bye": 0, "thankyou": 0, "please": 0, "yes": 0, "no": 0, "good": 0, "bad": 0, "sorry": 0, "help": 0, "water": 0, "eat": 0, "drink": 0, "food": 0, "need": 0, "want": 0, "wait": 0, "work": 0, "home": 0, "house": 0, "friend": 0, "see": 0, "look": 0, "go": 0, "today": 0, "tomorrow": 0, "where": 0, "who": 0, "how": 0, "man": 0}
- Updated data/gislr_labels.json and data/gislr_extraction_report.json with the
  exact mapping and blocked-attempt metrics. Source vocabulary coverage remains
  unverified because train.csv was inaccessible; 35–40 per class is not achieved.
- Next: verify token/account access and accept asl-signs competition rules, then
  rerun with --download --prompt-token, or supply the unzipped dataset locally.

## GISLR participant split extraction — 2026-11-09T09:30:01.447485+00:00
- Status: blocked; saved 538 samples.
- Counts by split/class: {"train": {"hello": 32, "bye": 32, "thankyou": 32, "please": 32, "yes": 32, "no": 32, "good": 0, "bad": 32, "sorry": 0, "help": 0, "water": 32, "eat": 0, "drink": 32, "food": 32, "need": 0, "want": 0, "wait": 32, "work": 0, "home": 32, "house": 0, "friend": 0, "see": 32, "look": 18, "go": 0, "today": 0, "tomorrow": 0, "where": 0, "who": 0, "how": 0, "man": 0}, "val": {"hello": 8, "bye": 8, "thankyou": 8, "please": 8, "yes": 8, "no": 8, "good": 0, "bad": 8, "sorry": 0, "help": 0, "water": 8, "eat": 0, "drink": 8, "food": 8, "need": 0, "want": 0, "wait": 8, "work": 0, "home": 8, "house": 0, "friend": 0, "see": 8, "look": 0, "go": 0, "today": 0, "tomorrow": 0, "where": 0, "who": 0, "how": 0, "man": 0}}
- Split totals: {"train": 434, "val": 104}; participant overlap: 0.
- Feature order: pose 0–17, left hand 0–20, right hand 0–20; XYZ; finite float32 (64,180), uniform subsampling/trailing zero padding.
- Metadata schema v2 contains class_to_index, path-to-index samples, sample_info, participants and split stats. Legacy consumers need adaptation.
- Report: data/gislr_extraction_report.json. Webcam source preserved.
- Blocker: Cannot download train_landmark_files/25571/1869843339.parquet (KaggleApiHTTPError, HTTP 429).

## GISLR participant split extraction — 2026-11-09T10:12:39.419571+00:00
- Status: incomplete; saved 760 samples.
- Counts by split/class: {"train": {"hello": 32, "bye": 32, "thankyou": 32, "please": 32, "yes": 32, "no": 32, "good": 0, "bad": 32, "sorry": 0, "help": 0, "water": 32, "eat": 0, "drink": 32, "food": 32, "need": 0, "want": 0, "wait": 32, "work": 0, "home": 32, "house": 0, "friend": 0, "see": 32, "look": 32, "go": 32, "today": 0, "tomorrow": 32, "where": 32, "who": 32, "how": 0, "man": 32}, "val": {"hello": 8, "bye": 8, "thankyou": 8, "please": 8, "yes": 8, "no": 8, "good": 0, "bad": 8, "sorry": 0, "help": 0, "water": 8, "eat": 0, "drink": 8, "food": 8, "need": 0, "want": 0, "wait": 8, "work": 0, "home": 8, "house": 0, "friend": 0, "see": 8, "look": 8, "go": 8, "today": 0, "tomorrow": 8, "where": 8, "who": 8, "how": 0, "man": 8}}
- Split totals: {"train": 608, "val": 152}; participant overlap: 0.
- Feature order: pose 0–17, left hand 0–20, right hand 0–20; XYZ; finite float32 (64,180), uniform subsampling/trailing zero padding.
- Metadata schema v2 contains class_to_index, path-to-index samples, sample_info, participants and split stats. Legacy consumers need adaptation.
- Report: data/gislr_extraction_report.json. Webcam source preserved.

## 2026-11-09 — Clean slate and participant-aware GISLR run verified
- Emptied data/processed/ and data/raw_shards/ before extraction and reset
  metadata.json to {}; preserved all webcam source bytes.
- Read train.csv and sign_to_prediction_index_map.json; both agree on all 250
  source labels. Exact target vocabulary has 19 present classes and 11 absent:
  good, sorry, help, eat, need, want, work, house, friend, today, how.
- Completed live --samples-per-class 40 run after recovering HTTP 429 limits.
  Each present class has 32 train + 8 val samples; each absent class has 0 + 0.
  Total: 760 real tensors (608 train / 152 val); zero read/validation failures.
- Deterministic participant split: 17 train signers, 4 val signers, zero overlap.
  All 760 arrays independently verified as finite float32 (64,180), with source
  CSV participant/path/sign and source label index matches; no unindexed NPY files.
- Metadata schema v2: class_to_index, samples (path -> output index), sample_info,
  participants and stats. Paths: data/processed/{train,val}/{class}/{id}.npy.
- Six unit/integration tests passed. Webcam SHA-256 unchanged; verification
  evidence saved in data/gislr_validation.json. No training was performed.
- Exit 2 / status incomplete accurately flags missing source vocabulary despite
  reaching the cap for all 19 available signs. No aliases or synthetic replacements.
- Script defaults to all matching sequences; --samples-per-class allows capped
  runs. --val-fraction defaults to .2, --seed to 42; downloads are cached and have
  bounded transient-error retries. See GISLR.md for schema and commands.
- Next: resolve the 11 absent signs or explicitly approve a 19-class vocabulary;
  adapt training to the saved split and inference to the documented XYZ features.

## GISLR 19-class training — 2026-11-09T10:18:49.959188+00:00
- Completed 40 epochs on cpu; 608 train / 152 val samples; batch size 32.
- Remapped the 19 present metadata classes to contiguous output indices 0–18.
- Participant-isolated extraction split retained; packed sequences exclude trailing zero padding.
- BiLSTM: input 180, hidden 128, 2 bidirectional layers, dropout .3, linear 256 -> 19.
- CrossEntropyLoss; AdamW lr .001 / weight decay .0001; ReduceLROnPlateau max/factor .5/patience 5.
- Best epoch 39: val accuracy 34.2105%, val loss 2.901256.
- Final: train accuracy 93.2566%, loss 0.275889; val accuracy 26.9737%, loss 3.138990.
- Best state_dict: models/model_gislr_19class.pth; sign-to-index mapping: models/labels_19class.json.
- History arrays/config: data/training_history.json. Best checkpoint reload/validation verified.
- Webcam source unchanged; its legacy architecture/features still require adaptation before inference.

## 2026-11-09 — FastAPI GISLR WebSocket inference server verified
- Added scripts/server_websocket.py, requirements-server.txt, WEBSOCKET.md and
  tests/test_server_websocket.py. Installed dependencies in .venv-webcam (Python 3.12).
- Route /ws/sign-language accepts JSON frame strings with Base64 JPEG data.
  Strict decoding, JPEG byte/dimension/message limits, bounded client count,
  idle timeout, optional Bearer handshake token and Origin allowlist are supported.
- Loads model_gislr_19class.pth and labels_19class.json at startup, strict state
  loading, eval/inference_mode; shared CPU model prediction lock. Reuses the exact
  training model including packed-sequence handling. Startup probes actual Holistic.
- Each connection owns a dedicated worker-thread Holistic tracker and float32
  (1,64,180) FIFO: pose 0–17, left hand 0–20, right hand 0–20, XYZ; missing/invalid
  landmarks zero-filled. Blocking computation is off the async event loop.
- Warmup requires 64 frames. Same top-1 label must have confidence strictly >.70
  on 5 consecutive inference frames (earliest response frame 68); otherwise returns
  detecting/prediction null. Invalid input, missing detections and gaps >3s reset
  temporal state. No frames/images are persisted or logged.
- Six tests passed: feature/decoder validation, strict threshold and label changes,
  independent client state, malformed/binary input, authentication/Origin/capacity,
  oversized messages, missing detections/gaps, cancellation-safe disconnect cleanup,
  real weights and a real Holistic JPEG WebSocket smoke test. Initial cleanup bug
  fixed with an AnyIO shielded cleanup scope; client slots and trackers are released.
- Real Holistic tests require macOS graphics-session access; they passed outside
  the restricted sandbox (sandbox prevented NSOpenGLPixelFormat creation).
- inference_webcam.py SHA-256 unchanged. No live signer/camera accuracy or external
  deployment/load test performed; existing best validation accuracy remains 34.21%.
- Launch from backend directory: ../.venv-webcam/bin/python scripts/server_websocket.py
  (localhost:8000). Deployment/protocol/environment settings are in WEBSOCKET.md.

## 2026-11-09 — Webcam CLI default paths updated
- Changed only the --weights and --labels defaults in scripts/inference_webcam.py
  to models/model_gislr_19class.pth and models/labels_19class.json, resolved relative
  to the backend ROOT. The architecture, feature adapter and streaming loop remain unchanged.
- Verified --help executes successfully in .venv-webcam (Python 3.12).
- Executed the script with no arguments: it reaches the new checkpoint path,
  then exits 1 at load_assets with KeyError: idx_to_class. The current labels are
  a flat 19-class sign-to-index dict; this loader expects the old nested five-class format.
- Webcam inference is therefore not yet operational with the new defaults.
  Its five-output model and XY/presence feature adapter also require GISLR-compatible
  adaptation (19 outputs, pose 0–17 and hand XYZ) before camera inference can work.
  No camera was opened. The existing WebSocket server already uses the matching model/adapter.

## 2026-11-09 — Webcam refactored for GISLR 19-class checkpoint
- Replaced the five-class model/label loader with dynamic class counts and support
  for flat name->int, numeric-string-index->name and nested idx_to_class mappings.
  Rejects duplicate/noncontiguous indices and empty/duplicate label names.
- Model matches train_gislr.py: 180 input, hidden 128, 2 bidirectional LSTM layers,
  dropout .3; linear 256->len(labels); packed-sequence padding behavior; strict
  weights_only state_dict load and eval mode. Defaults remain the GISLR assets.
- Features now match GISLR: native pose indices 0–17, left/right hand 0–20, XYZ,
  zero missing/invalid landmark triples. Removed the old DWPose/confidence adapter.
- FIFO forms float32 (1,64,180). Display uses a full five-window softmax mean,
  strictly above --threshold (.65 default); otherwise Detecting.... Missing
  detections and frame gaps reset state. Preserved camera/GUI cleanup and controls.
- Five tests passed in .venv-webcam, including real checkpoint loading through all
  label formats, logits matching the training model on padded sequences, feature
  order/zeros, smoothing boundary checks, and execution of main with real weights
  and a mocked camera/GUI (69 frames). Verified all inference inputs (1,64,180),
  successful asset loading and camera/window cleanup. --help also passed.
- Updated WEBCAM.md. Physical camera/GUI/live signer accuracy were not tested.
  This resolves the previous idx_to_class startup failure; historical unchanged-
  webcam hash guarantees apply only to the earlier pre-refactor versions.

## 2026-11-09 — WebSocket motion gate and gloss-to-sentence integration
- Current checkpoint remains models/model_gislr_19class.pth, labels remain
  models/labels_19class.json; model/feature contract unchanged: (1,64,180) XYZ.
- Added per-client hand temporal variance gate: mean of per-coordinate temporal
  variances over dimensions 54:180, threshold .015. Startup padding is excluded
  until 64 received frames; absent current hands force rest. Rest bypasses the
  classifier, sets state IDLE, returns status idle/prediction null and resets debounce.
- Per-client gloss_sequence preserves first-occurrence order and suppresses all
  repeated words in a sentence. Existing >.70 same-label five-frame debounce retained.
- After 36 consecutive rest frames with glosses, sends sentence_complete with
  glosses and sentence, then clears glosses/FIFO/debounce/pause state only after send.
  Invalid frames/gaps break idle counting but preserve accepted glosses.
- Added async build_sentence(list[str]) using off-loop groq.Groq, environment-only
  GROQ_API_KEY, llama-3.3-70b-versatile and the requested exact system prompt.
  SDK request timeout 10s, no retries, async deadline 12s. Missing key, SDK/API error,
  timeout or empty completion falls back to raw space-separated glosses. Only
  glosses are transmitted to Groq; no frames or keys are logged/persisted.
- Installed groq 1.7.0 in .venv-webcam and updated requirements-server.txt/WEBSOCKET.md.
- Eleven tests passed including static and missing-hand bypass, moving-to-static
  transitions, debounce, uniqueness, 36-frame completion/clearing/no repeat sends,
  connection isolation, mocked API success/failure, missing-key/timeout fallback,
  original protocol/access-limit/cleanup checks and real checkpoint/Holistic smoke.
- No live Groq call or live signer calibration performed. User will supply key
  in environment before launch; fallback already works without it.
- Pause is frame-count based (~1.5s at 24 FPS) after the rolling variance gate
  becomes idle; old motion in the 64-frame window and slower FPS add latency.
  No incoming frames do not advance the pause counter. The .015 threshold needs
  real-signer validation; heuristic gating does not guarantee removal of all bias.

## 2026-11-09 — Gemini translation and .env secrets management
- Replaced the server's Groq SDK integration with google-genai (installed 1.75.0)
  targeting gemini-2.5-flash via explicit AI Studio API-key client (vertexai=False).
  Retained models/model_gislr_19class.pth, labels_19class.json, the .015 hand-motion
  variance gate, (1,64,180) FIFO, >.70 five-frame debounce and 36-idle-frame completion.
- python-dotenv load_dotenv() runs at module startup against backend ROOT/.env,
  independent of launch working directory. override=False preserves exported env.
  build_sentence reads only GEMINI_API_KEY; missing/blank values return raw glosses.
- Added .env/.env.* exclusions to backend and workspace .gitignore, exempting only
  .env.example. Added a blank backend .env.example; no real Gemini key was written.
- Removed the old credential-looking literal from source. Any previously exposed
  Groq credential should be rotated; removing it from source does not revoke it.
- System instruction updated exactly as requested, including MY/YOUR and I/YOU,
  question/statement intent and no explanations/extra quotes. Only glosses are sent.
- Sync SDK call remains off-loop, with 10000 ms HTTP timeout, one attempt and 12s
  async deadline. Missing SDK, API errors/timeouts, blocked/empty output fall back
  to raw space-separated glosses. Client context closes connections. Exception
  text is never logged. Groq dependency removed from requirements-server.txt;
  google-genai and explicit python-dotenv dependencies added.
- Twelve tests passed: .env path/precedence, Gemini model/prompt/config/content,
  output handling, missing-key/error/timeout fallbacks, sentence state/isolation,
  motion/FIFO/debounce/security/cleanup regressions and real weights/Holistic smoke.
  Gemini calls were mocked; no live API request or new key validation performed.
- Updated WEBSOCKET.md. Copy backend .env.example to .env, set GEMINI_API_KEY,
  and launch/restart the server. Existing checkpoints and webcam code are unchanged.

## 2026-11-09 — Webcam converted to WebSocket streaming client
- Replaced local inference in scripts/inference_webcam.py with camera 0 capture,
  JPEG/Base64 JSON streaming to ws://localhost:8000/ws/sign-language. Removed
  local PyTorch/MediaPipe/model/label dependencies and former model CLI options.
- OpenCV GUI/capture remain on main thread; dedicated background asyncio worker
  uses websockets with ping/timeout handling and .5–5s reconnect backoff. Default
  target cap 24 FPS, JPEG quality 80, max side 640; configurable CLI controls.
- At most one frame request is in flight, with one freshest captured frame retained;
  frames older than .5s are dropped. Inference/LLM latency reduces actual send FPS
  rather than building stale queues. GUI stays responsive while network awaits.
- Displays complete server sentence prominently (wrap/paginate long text), recent
  glosses and detection/idle/sign/processing/error/connect status. Uses server
  gloss arrays when available; otherwise tracks confirmed prediction words locally.
- Current server emits only final completion, so processing status is inferred
  while waiting after 35 idle responses with pending words (next frame hits the
  server's 36-frame pause trigger). Also supports explicit processing messages.
- Reconnect resets incomplete local gloss state; retains last completed sentence.
  q/window close/Ctrl+C cancel networking and release camera/windows. Backend .env
  supports optional SIGN_WS_TOKEN; --origin supports server allowlists. No new
  Gemini calls or server inference changes. Model paths stay server-owned.
- Added requirements-client.txt for separate lightweight environments. Existing
  .venv-webcam already has dependencies; avoid mixing OpenCV distributions.
  Explicit compatible websockets range also added to requirements-server.txt.
- Seven tests passed: JPEG roundtrip/resize, overlay/gloss/sentence state, full-text
  wrapping/rendering, freshness/latest-only buffering, real loopback reconnect and
  JSON/sentence exchange, timeout/unavailability, cadence, mocked GUI execution and
  resource cleanup. --help passed. No physical camera/GUI or live signer/Gemini test.
- Updated WEBCAM.md and WEBSOCKET.md. Launch server first, then in another terminal:
  ../.venv-webcam/bin/python scripts/inference_webcam.py (from backend directory).

## GISLR 19-class training — 2026-11-09T12:04:06.336231+00:00
- Completed 70 epochs on cpu; 608 train / 152 val samples; batch size 32.
- Remapped the 19 present metadata classes to contiguous output indices 0–18.
- Participant-isolated extraction split retained; packed sequences exclude trailing zero padding.
- BiLSTM: input 180, hidden 128, 2 bidirectional layers, dropout .4, linear 256 -> 19.
- CrossEntropyLoss label smoothing .1; AdamW lr .001 / weight decay .001; ReduceLROnPlateau max/factor .5/patience 5.
- Best epoch 33: val accuracy 37.5000%, val loss 2.543471.
- Final: train accuracy 98.6842%, loss 0.683347; val accuracy 32.2368%, loss 2.578600.
- Best state_dict: models/model_gislr_19class.pth; sign-to-index mapping: models/labels_19class.json.
- History arrays/config: data/training_history.json. Best checkpoint reload/validation verified.
- Shared shoulder XYZ normalization; training-only noise/scaling/translation/frame masking.
- Server uses matching normalization via checkpoint config; restart after training. Live accuracy untested.

## 2026-11-09 — Regularized normalized GISLR training verified
- Added shared shoulder XYZ normalization (pose 11/12 midpoint and 3D shoulder
  distance >=.001). Missing joints remain zero; invalid-anchor frames are masked.
  Data on disk stays raw float32 (64,180). Added training-only stochastic Gaussian
  noise std .015, scaling .9–1.1, translation -.05–.05, and 7–9 masked positions
  out of 64 (10–15%, possibly including existing padding; preserve one active frame).
- Dataset __getitem__ augments fresh clones; cached data is not mutated. Validation
  normalization is deterministic. No split or label-index changes (608/152, 19 classes).
- Increased LSTM dropout to .4; label smoothing .1; AdamW weight decay .001,
  lr .001; retained ReduceLROnPlateau(max, factor .5, patience 5). Ran all 70 epochs.
- Best epoch 33: validation 37.50% (57/152), smoothed CE loss 2.543471.
  Previous best 34.21% (52/152): +3.29 percentage points on the same split.
  Final augmented training 98.6842%, loss .683347; val 32.2368%, loss 2.578600.
  Overfitting remains; live-webcam generalization improvement is not established.
- Exports: models/model_gislr_19class.pth, models/labels_19class.json,
  models/model_gislr_19class.config.json, data/training_history.json (70 metric entries).
  Best state reload reproduced recorded metrics. Prior artifacts backed up at
  data/training_runs/20261109T120309147450Z/. Baseline history remains available.
- Server now uses matching shared normalization, gated by the required checkpoint
  config; verifies label mapping/version. Motion variance still uses RAW coordinates.
  Restart server after replacing assets. Webcam stays a lightweight JPEG client.
- Four training tests passed (normalization invariance/invalid anchors/zeros,
  train-only augmentation/nonmutation/frame masking, split mapping, model gradients).
  Twelve server tests passed including actual checkpoint preprocessing/logit parity
  and real Holistic smoke, with Gemini mocked. No live camera/signer evaluation.
- Updated GISLR.md and WEBSOCKET.md with preprocessing, deployment and metric limits.

## 2026-11-09 — Deterministic raised-hands sentence submission
- Removed the 36-idle-frame automatic sentence boundary and the webcam client's
  idle-count processing inference. Rest still bypasses classification via the
  raw hand variance .015 gate, but cannot submit a sentence.
- Per-connection gloss_sequence now suppresses only adjacent duplicate predictions:
  A/A/B/A becomes A/B/A. Existing same-label confidence >.70 for 5 continuous
  inference frames remains required. Valid frame responses include live glosses.
- Submit checks raw MediaPipe pose: left wrist 15 y < left shoulder 11 y AND
  right wrist 16 y < right shoulder 12 y, for 15 consecutive processed frames.
  All four triples must be finite and pose visibility >=.5. Missing/unreliable
  pose, interruption, invalid input or a long frame gap breaks the count.
- Gesture runs before the motion gate/model, clears FIFO/debounce, and cannot add
  gesture frames as sign words. A nonempty list at frame 15 sets state
  SUBMITTING_SENTENCE; empty lists do not call Gemini. Trigger latches until an
  observed valid non-raised pose rearms it, preventing held-gesture repeats.
- Server emits submitting_sentence with state SUBMITTING_SENTENCE/live glosses,
  then sentence_complete with translated sentence and submitted glosses. Clears
  accumulator and temporal state only after final send. Gemini/environment/raw-
  gloss fallback and model/checkpoint normalization contract remain unchanged.
- Webcam displays live 'Accumulated Glosses: [hello, water, please]' and the
  instruction 'Raise both hands above shoulders to translate sentence'. Final
  translation remains prominent in green. Explicit submit payload drives the
  processing status; client waits for final completion before streaming again.
- Fourteen server tests + seven client tests passed (21 total), including gesture
  continuity/boundaries/visibility/empty list/latch/rearm, adjacent duplicate
  suppression and A/B/A retention, no idle submission, no gesture classification,
  isolation, intermediate/final payloads, GUI/transport, normalization parity,
  Gemini fallback and real Holistic smoke. API calls and camera/GUI were mocked
  where relevant; physical submit-gesture/signer behavior and live Gemini untested.
- Updated WEBSOCKET.md and WEBCAM.md. Restart both server and client to use the
  new protocol. Fifteen processed frames take ~.625s at 24 FPS (longer if slower).

## GISLR participant split extraction — 2026-11-09T13:40:44.930819+00:00
- Status: complete; saved 7327 samples.
- Counts by split/class: {"train": {"hello": 321, "bye": 327, "thankyou": 294, "please": 309, "yes": 312, "no": 303, "good": 0, "bad": 317, "sorry": 0, "help": 0, "water": 312, "eat": 0, "drink": 320, "food": 327, "need": 0, "want": 0, "wait": 272, "work": 0, "home": 321, "house": 0, "friend": 0, "see": 312, "look": 332, "go": 323, "today": 0, "tomorrow": 319, "where": 296, "who": 329, "how": 0, "man": 321}, "val": {"hello": 69, "bye": 75, "thankyou": 63, "please": 75, "yes": 74, "no": 67, "good": 0, "bad": 59, "sorry": 0, "help": 0, "water": 72, "eat": 0, "drink": 80, "food": 65, "need": 0, "want": 0, "wait": 74, "work": 0, "home": 74, "house": 0, "friend": 0, "see": 76, "look": 82, "go": 65, "today": 0, "tomorrow": 69, "where": 76, "who": 74, "how": 0, "man": 71}}
- Split totals: {"train": 5967, "val": 1360}; participant overlap: 0.
- Feature order: pose 0–17, left hand 0–20, right hand 0–20; XYZ; finite float32 (64,180), uniform subsampling/trailing zero padding.
- Metadata schema v2 contains class_to_index, path-to-index samples, sample_info, participants and split stats. Legacy consumers need adaptation.
- Report: data/gislr_extraction_report.json. Webcam source preserved.

## GISLR 19-class training — 2026-11-09T13:48:28.342967+00:00
- Completed 50 epochs on cpu; 5967 train / 1360 val samples; batch size 32.
- Remapped the 19 present metadata classes to contiguous output indices 0–18.
- Participant-isolated extraction split retained; packed sequences exclude trailing zero padding.
- BiLSTM: input 180, hidden 128, 2 bidirectional layers, dropout .4, linear 256 -> 19.
- CrossEntropyLoss label smoothing .1; AdamW lr .001 / weight decay .001; ReduceLROnPlateau max/factor .5/patience 5.
- Best epoch 45: val accuracy 71.1029%, val loss 1.526864.
- Final: train accuracy 98.3912%, loss 0.662513; val accuracy 67.1324%, loss 1.598593.
- Best state_dict: models/model_gislr_19class.pth; sign-to-index mapping: models/labels_19class.json.
- History arrays/config: data/training_history.json. Best checkpoint reload/validation verified.
- Shared shoulder XYZ normalization; training-only noise/scaling/translation/frame masking.
- Server uses matching normalization via checkpoint config; restart after training. Live accuracy untested.

## 2026-11-09 — Full target dataset extraction and 50-epoch retraining complete
- Removed the per-class cap/quota implementation and --samples-per-class option.
  Resumption validates and preserves prior indexed outputs, provenance and signer
  assignment. Full extraction coverage is separate from unavailable vocabulary.
- Added scripts/fetch_gislr_archive.py: official ZIP redirect, selective HTTP ranges,
  persistent worker connections, range/length/ETag and ZIP size/CRC validation,
  parquet magic checks and atomic writes. Did not download the full 40 GB archive
  or persist signed URLs/credentials. All 7,327 target parquet files are cached.
- Converted all 7,327 valid source sequences, with zero failures, to raw float32
  (64,180) tensors: 5,967 train / 1,360 val; exactly 19 present classes. Original
  seed-42/.2 participant assignment preserved: 17 train signers, 4 val, zero overlap.
- Updated metadata.json, data/gislr_extraction_report.json and gislr_labels.json.
  Source CSV path/participant/sign/index, full disk/metadata coverage, tensor shape,
  dtype/finiteness and normalization viability independently verified. No unindexed
  NPY files and no samples lacking valid shoulder normalization. Full validation
  evidence: data/gislr_validation.json. Per-split class counts: {"train": {"hello": 321, "bye": 327, "thankyou": 294, "please": 309, "yes": 312, "no": 303, "good": 0, "bad": 317, "sorry": 0, "help": 0, "water": 312, "eat": 0, "drink": 320, "food": 327, "need": 0, "want": 0, "wait": 272, "work": 0, "home": 321, "house": 0, "friend": 0, "see": 312, "look": 332, "go": 323, "today": 0, "tomorrow": 319, "where": 296, "who": 329, "how": 0, "man": 321}, "val": {"hello": 69, "bye": 75, "thankyou": 63, "please": 75, "yes": 74, "no": 67, "good": 0, "bad": 59, "sorry": 0, "help": 0, "water": 72, "eat": 0, "drink": 80, "food": 65, "need": 0, "want": 0, "wait": 74, "work": 0, "home": 74, "house": 0, "friend": 0, "see": 76, "look": 82, "go": 65, "today": 0, "tomorrow": 69, "where": 76, "who": 74, "how": 0, "man": 71}}
- Completed exactly 50 epochs on the expanded dataset with existing normalization,
  training-only augmentation, dropout .4, label smoothing .1, AdamW lr .001 /
  weight decay .001 and ReduceLROnPlateau. Fixed training summaries to use actual counts.
- Best epoch 45: val 71.1029% (967/1360), loss 1.526864.
  Final augmented train 98.3912%, loss 0.662513; val
  67.1324%, loss 1.598593. Generalization gap remains;
  live webcam accuracy is untested. Prior 152-example validation scores are not
  directly comparable with this expanded 1,360-example validation set.
- Exports: models/model_gislr_19class.pth, labels_19class.json, preprocessing config
  model_gislr_19class.config.json and data/training_history.json (50 entries).
  Best checkpoint reload/metrics verified; current metadata hash and class mapping
  verified against history/config. Prior assets: data/training_runs/20261109T134135506757Z/.
  Pre-expansion metadata audit: data/extraction_runs/20261109T125500Z/.
- Gemini SDK initialization verified offline using GEMINI_API_KEY configuration
  and explicit AI Studio client. Startup .env loading remains unchanged. This
  execution environment has no GEMINI_API_KEY, so no live translation was tested.
- Replaced the system instruction with the requested Sign Language Translator
  fluent-sentence prompt and man/water/please example. Missing-key, empty-output,
  API and timeout failures now print 'Gemini API Error: ...' with useful provider
  details and the configured key redacted; raw-gloss fallback remains robust.
  Optional GEMINI_MODEL override added; prior default gemini-2.5-flash retained.
  Non-2.5 models omit the 2.5-only thinking-budget setting. Updated .env.example.
- All 26 tests passed: 6 extraction, 2 archive, 4 training and 14 server, including
  full-dataset mapping, model normalization parity, real Holistic smoke, Gemini
  prompt/config/override/error logging/redaction tests with mocked API calls.
- Updated GISLR.md, WEBSOCKET.md and reader dependencies. Restart the server to
  load the new weights/config and translation code. Gesture submit/client behavior
  is unchanged. Set GEMINI_API_KEY in backend .env for live sentence generation.

## 2026-11-09 — Root .env loading and explicit Gemini invocation diagnostics
- Server now explicitly loads /Users/Arjun/AIML-sign/.env at module startup via
  load_dotenv(dotenv_path=ENV_ROOT / '.env', override=True). ROOT still points to
  the backend for model/data assets; ENV_ROOT is ROOT.parent. Root .env values
  override stale exported environment values. Existing API key was not printed.
- Updated the prompt to the requested expert Sign Language to English instruction,
  ordered hello/water/please example, complete sentence example and ALWAYS/NEVER rules.
- Every nonempty translation attempt prints '[LLM] Translating glosses: [...]'.
  Missing key, API failure, empty response or timeout prints
  '[LLM ERROR] Gemini API call failed: ...' before raw-gloss fallback. Provider
  error details remain visible; the configured API key is redacted if present.
- Live test with the root configuration reached Google: gemini-2.5-flash returned
  404 NOT_FOUND, explicitly unavailable to new users and recommending gemini-3.8-flash.
  Updated default/blank GEMINI_MODEL fallback and .env.example to gemini-3.8-flash;
  explicit model overrides remain supported. No model weights/labels changed.
- Further live replacement-model checks returned provider 504 DEADLINE_EXCEEDED
  and 503 UNAVAILABLE (high demand). These errors were printed as requested and
  fallback completed safely. Successful live sentence translation remains unverified.
- Final configuration sets LOW thinking for 3.8, a 15s HTTP timeout, one attempt,
  18s outer deadline and 1024 output-token cap; this remains within the unchanged
  webcam client's 20s response timeout. Explicit 2.5 overrides retain budget=0.
  Final LOW/timeout settings passed mocked tests; no further live retries were
  made after the provider reported high demand.
- All 15 server tests passed, including root-path/override precedence, invocation
  log/error log/redaction, exact system instruction/config, fallback paths, model
  overrides and real checkpoint/Holistic smoke. Model, motion gate, FIFO, debounce,
  raised-hands submit protocol and data/model metrics remain unchanged.
- Updated WEBSOCKET.md. Restart the server to load root .env and these changes.

## 2026-11-09 — Bounded Gemini 2.5 timeout and connection-safe fallback
- build_sentence now explicitly targets gemini-2.5-flash, overriding the prior
  3.8 default. GEMINI_MODEL environment settings cannot select another model.
  Updated the blank .env.example accordingly; did not modify the actual key.
- SDK invocation remains await asyncio.to_thread(request), wrapped in
  asyncio.wait_for with LLM_TIMEOUT_SECONDS=8.0. HTTP timeout is 6000 ms with one
  attempt; thinking_budget=0. Root .env load override=True and prompt are retained.
- Successful completion prints '[LLM SUCCESS] Sentence generated: {sentence}'.
  Local asyncio/native/HTTP timeouts and provider HTTP 504/DEADLINE_EXCEEDED print
  '[LLM TIMEOUT] Falling back to gloss joining.' and return joined glosses.
  Other failures retain explicit LLM ERROR logging with API-key redaction.
- Timeout fallback uses the normal sentence_complete path; it does not close the
  client connection. Late SDK-thread results are ignored; HTTP timeouts bound I/O,
  while the event-loop await returns after its 8-second deadline. The unchanged
  webcam response timeout remains 20 seconds.
- Executed all 17 server tests successfully: success/model/timeout configuration,
  SDK worker nonblocking heartbeat, prompt deadline fallback, provider 504 fallback
  through WebSocket, successful next-frame exchange after fallback, and existing
  gesture/model/normalization/security/cleanup/real Holistic smoke regressions.
- API requests were mocked for this change. No repeated live request was made:
  earlier live verification recorded a 404 for gemini-2.5-flash on this account.
  Restoring this explicitly requested model does not resolve provider access;
  ordinary 404 failures remain visible through LLM ERROR and safe fallback.
- Updated WEBSOCKET.md. Data, model weights, root .env policy, submit gestures and
  client protocol are unchanged. Restart the server to apply this configuration.

## 2026-11-09 — Groq primary and Gemini fallback translation
- Read context and retained the 19-class GISLR architecture and model paths.
  Server now tries AsyncGroq with clean model name llama-3.1-8b-instant, then
  google-genai gemini-2.5-flash on missing key, error, empty output or timeout.
- Workspace-root .env still loads at startup with override=True; both
  GROQ_API_KEY and GEMINI_API_KEY are read safely from environment variables.
  Updated blank .env.example and restored groq>=1,<2 in requirements-server.txt
  (installed SDK 1.7.0). Actual .env credentials were not changed or printed.
- Both providers use the requested expert gloss-to-English prompt. Groq uses
  its async client; synchronous Gemini runs in asyncio.to_thread. Each attempt
  has a 6s SDK timeout, no retries and an 8s async deadline (two attempts within
  the webcam's 20s response timeout). Empty input bypasses both providers.
- Success logs are [GROQ SUCCESS] and [GEMINI SUCCESS]; provider error/timeout
  diagnostics redact both keys. If neither service succeeds, logs
  '[LLM FALLBACK] Returning raw gloss sequence.' and returns space-joined glosses.
  Request cancellation propagates rather than starting another provider call.
- All 19 server tests passed, including Groq success/clean model/prompt, real
  async deadline failover, missing keys, empty provider output, diagnostics/key
  redaction, Gemini worker deadline/event-loop responsiveness, provider 504
  fallback with next-frame WebSocket exchange, gesture/client isolation,
  normalization, checkpoint and real MediaPipe smoke. API calls were mocked;
  successful live translation/provider availability was not verified.
- Motion variance, raw (1,64,180) FIFO, five-frame debounce, duplicate suppression,
  15-frame hands-up submit trigger, model assets and dataset metrics unchanged.
  Updated WEBSOCKET.md. Restart the server to load the new engine and root keys.

## 2026-11-09 — Gemini 12s deadline and Hearing-to-Deaf browser subtitles
- Read current architecture: server owns 19-class BiLSTM, raw (1,64,180) FIFO,
  normalization, motion variance, five-frame debounce and 15-frame hands-up submit.
  Data/model/gesture contracts and Groq-first translation order are preserved.
- Explicitly set Gemini model to gemini-2.5-flash (the file had been using
  gemini-3.1-flash-lite). Gemini HTTP timeout is 12000 ms and asyncio.wait_for
  uses GEMINI_TIMEOUT_SECONDS=12.0, retaining asyncio.to_thread and one attempt.
  Groq stays at 6s SDK / 8s async. Native and browser response deadlines are 30s,
  leaving headroom for both providers and inference. Existing error fallback stays.
- Added FastAPI /webcam and /webcam.js serving web/webcam.html and web/webcam.js.
  Browser camera sends unmirrored 640px JPEG/Base64 at up to 24 FPS with one frame
  in flight, intermediate-submit handling, reconnect backoff and 30s reply limit.
  Sign translation remains green at top, with live accumulated glosses/instruction.
- Added opt-in Steve microphone capture via SpeechRecognition /
  webkitSpeechRecognition, continuous=true, interimResults=true. Dedicated bottom
  Steve-to-Sam subtitle box displays interim revisions and final words independently
  of camera/network/LLM work. Supports language selection, stop/start, clear,
  recognition restart, visible unsupported/permission/network/audio errors and
  page-close resource cleanup. No extra speech SDK or API key is needed.
- Added inference_webcam.py --browser (optional --ui-url) to launch this UI without
  opening the native camera. Default OpenCV mode remains available; voice subtitles
  are implemented in browser mode rather than the native OpenCV display.
- Browser speech requires supported Chrome, localhost/HTTPS and microphone consent;
  its vendor may process audio remotely. Transcripts stay in browser memory; no
  microphone audio is sent to our server/Groq/Gemini. Browser sign mode currently
  requires SIGN_WS_TOKEN unset (native client still supports Bearer auth) and the
  UI origin must satisfy any configured origin allowlist.
- Passed 28 Python server/webcam tests, including real checkpoint/Holistic smoke,
  12000ms Gemini config/12s deadline, worker responsiveness, UI routes without .env
  access, launcher, original WS transport/gesture isolation; and 3 Node browser
  tests for partial/final revisions, restart/clear, errors, captions during submit,
  frame backpressure/reconnect and resource cleanup. CLI help and JS syntax passed.
- API/speech/camera inputs were mocked for automation. No live provider translation,
  physical microphone/camera capture or real-browser speech-service session was
  verified. Updated WEBCAM.md and WEBSOCKET.md. Restart the server, then open
  http://localhost:8000/webcam and click Start camera / Start Steve's microphone.

## 2026-11-09 — Unified two-way split-screen app
- Read context and verified the 19-class GISLR model: 2-layer bidirectional LSTM,
  input 180, hidden size 128; weights models/model_gislr_19class.pth and mapping
  models/labels_19class.json, companion normalization config unchanged. Raw FIFO
  is (1,64,180); motion .015, five-frame >.70 debounce and 15-frame hands-up retained.
- Gemini gemini-3.1-flash-lite is now primary, with explicit 12000ms HTTP timeout,
  12.0s async deadline, no retries and minimal thinking_level (replacing the old
  Gemini 2.5 thinking_budget option). AsyncGroq llama-3.1-8b-instant is fallback
  (6s SDK/8s deadline), then raw joined glosses. Root .env override/key-redaction
  behavior retained; actual credentials untouched. SDK minimum google-genai>=1.51.
- Created frontend/index.html, style.css and app.js. Mounted frontend/ at / last
  so HTTP APIs and /ws/sign-language, /webcam and /webcam.js remain compatible.
  Browser launcher default is now http://localhost:8000/.
- Split-screen Sam view: active camera canvas, live gloss JSON, editable translation
  draft, green Send to Steve, explicit preview control and large received subtitles.
  Steve view: received Sam messages, optional read-aloud TTS, microphone with
  interim/final Web Speech drafts, editable text and blue Send to Sam.
- Added /ws/two-way with frame/stage/submit/send/ping commands and typed staged
  responses. Gloss previews are scheduled after 900ms without a new gloss; this
  stages only, without auto-sending or replacing the hands-up sentence boundary.
  Hands-up stages and sends once, reusing a matching cached translation. Confirmed
  Sam sends clear the originating connection's gloss/FIFO state; previews do not.
- Translation runs as a cancellable background task; confirmed hearing messages
  dispatch while generation runs. Per-socket send locks serialize outputs. FIFO,
  tracker and draft remain connection-local; only confirmed text broadcasts to
  active room members. URL-fragment conversation IDs select room query parameters.
- Both directions require explicit button confirmation except Sam's deliberate
  hands-up trigger. Drafts survive reconnect in the browser; uncertain sends are
  not replayed automatically. Message IDs suppress repeated display, and only
  matching request IDs clear a sending draft. TTS stops microphone capture to
  avoid transcribing synthetic speech; microphone can be restarted manually.
- Room delivery/history is in-memory and process-local; use one server worker.
  Room IDs are routing identifiers, not authentication. Existing origin, Bearer,
  connection/message/image limits retained. Browser mode requires SIGN_WS_TOKEN
  unset; native CLI continues Bearer support. UI assets never include API keys.
- Passed 31 Python server/webcam tests and 6 Node browser tests: Gemini-first
  failover, stage privacy, dispatch during translation, room isolation, validation,
  preview-cache reuse, one-shot hands-up send, original model/Holistic behavior,
  draft confirmation/acks, speech revisions, TTS/capture interaction and reconnect.
- Browser-verified both directions using typed sample drafts on a temporary fake
  runtime server; delivered messages appeared in the correct received boxes.
  Screenshot: data/reports/two_way_preview.png. Temporary server stopped afterward.
  No physical camera/mic, live TTS audio or live Gemini/Groq response was verified.
- Updated WEBSOCKET.md, WEBCAM.md and blank .env.example. Restart the normal server,
  open http://localhost:8000 in Chrome and enable camera/mic as needed. Share the
  complete page URL to join the same conversation from another browser tab/device.

## 2026-11-09 — Three-column DM chat and supported model targets
- Read current context; preserved the 19-class GISLR checkpoint/label/config,
  normalization, (1,64,180) FIFO, motion gate, debounce and hands-up trigger.
- Fixed the current Groq openai/gpt-4o-mini target to llama-3.3-70b-versatile.
  Gemini now explicitly uses gemini-2.5-flash with 12000ms HTTP / 12.0s async
  deadline and thinking_budget=0; removed the Gemini 3 thinking_level setting.
  Gemini remains primary, Groq fallback (6s SDK/8s async), then raw joined glosses.
- Error logs now include exception type/message, available exact provider response
  body and HTTP status. Both configured keys are redacted throughout. Unread
  response bodies do not interrupt fallback. Existing success/fallback logs retained.
- Refactored frontend/index.html and style.css into three columns: Sam signer
  controls on left, shared Instagram-style DM stream center, Steve voice controls
  on right. Side panels retain editable drafts, camera/glosses, mic/language and
  optional TTS controls; both send buttons now read Send to Chat.
- app.js appends only confirmed server message events to the center chat, tagging
  Sam (Sign) left/green and Steve (Voice) right/purple. Live glosses, partial speech
  and LLM draft events never append messages. Hands-up confirmation still stages
  and sends via the existing server protocol. Message-ID dedup and matching-request
  acknowledgements retained; messages render as text, never HTML.
- Added smooth auto-scroll to the shared bounded chat stream, with instant scroll
  for reduced-motion preferences. Existing room isolation, reconnect/draft retention,
  staged previews, frame backpressure and mic/TTS separation remain intact.
- All 32 Python server/webcam tests and 6 Node browser tests passed, including
  provider model/timeout settings, full response diagnostics/key redaction,
  raw-gloss fallback, real checkpoint/Holistic smoke, room/stage/gesture protocol,
  staged-only center chat, sender labels/alignment, smooth scroll and speech/TTS.
- Browser verification on a temporary fake-inference server showed zero messages
  after typing both drafts, then correctly aligned/tagged messages after confirming
  both sends. Saved data/reports/dm_chat_preview.png. Physical camera/mic, live
  provider translation and actual TTS audio remain unverified. Preview server stopped.
- Updated WEBSOCKET.md and blank .env.example; actual API keys and model/data assets
  were untouched. Restart server and open http://localhost:8000 to use the new UI.

## 2026-11-09 — Ordered Groq multi-model fallback with Gemini last
- Read context and preserved the 19-class GISLR inference assets, FIFO/motion/
  debounce/gesture logic, three-column DM UI and staged two-way protocol.
- build_sentence now calls the explicit root load_environment helper per request;
  it uses load_dotenv(dotenv_path=ENV_ROOT / '.env', override=True). Startup loading
  remains. Configured keys are read after refresh; actual .env was not edited.
- Tries the exact Groq list in order: llama3-8b-8192, llama-3.1-8b-instant,
  mixtral-8x7b-32768. Each AsyncGroq attempt advances on 404/error/timeout or empty
  output; missing Groq key skips the list. Stops immediately on valid sentence.
- Only after Groq exhaustion, tries google-genai gemini-2.0-flash with the sync
  SDK in asyncio.to_thread, 12000ms HTTP timeout and 12s async deadline. Removed
  Gemini 2.5 thinking_budget configuration for the requested older fallback model.
- Set Groq SDK timeout 3s, async attempt deadline 4s and retries=0. Three attempts
  plus Gemini's 12s deadline total 24s of await budgets, within existing 30s client
  limits. Each client closes per attempt. Cancellation propagates without retries.
- Uses the exact requested Sign Language Gloss-to-English prompt, including
  hello/water/please example and ONLY/no-markdown/no-quotes instruction. Success
  logs '[LLM SUCCESS] Model: {used_model} | Output: {sentence}'. Failure diagnostics
  include attempted model, exact available provider response/status and redacted
  keys. Raw space-joined glosses are returned only after all available APIs fail.
- All 35 Python server/webcam tests passed: first/second/third Groq successes,
  ordered 404 fallthrough, all-Groq failure then Gemini success, real async attempt
  timeout advancement, missing keys, empty output, all-API raw fallback, refresh
  invocation, exact prompt/model/config/logging, cancellation and existing real
  model/Holistic, WebSocket cleanup, room/stage/gesture/client regressions.
- Provider calls and per-request credential loading were mocked for engine tests;
  dedicated dotenv tests verify real temporary-file override behavior. No live
  provider request was made or model availability verified. Updated WEBSOCKET.md
  and blank .env.example. Restart the server to apply the fallback chain.
