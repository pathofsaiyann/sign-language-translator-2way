# AIML-SIGN current context

## Architecture and state
- Root project docs: README.md and FRONTEND_INTEGRATION.md (one level above backend).
  Canonical current state lives here; chronological history is docs/CONTEXT_HISTORY.md
  and progress.md. Read context/progress before work and update them after runs.
- Four supported entrypoints in scripts/: extract_gislr.py, train_gislr.py,
  server_websocket.py, inference_webcam.py. Shared preprocessing/event contracts
  are internal modules in scripts/_shared/. Pilot/placeholder/archive tooling and
  associated tests were removed; there are no legacy executable paths to maintain.
- Keep data/processed/, data/raw_shards/, models/, metadata.json and history intact.
  These existing assets were not regenerated or retrained during cleanup.

## Data and model
- GISLR tensors: 7,327 raw finite float32 (64,180), train 5,967 / val 1,360;
  17 / 4 participant-isolated signers, zero overlap. Pose 0–17 (54 XYZ), left
  hand 0–20 (63 XYZ), right hand 0–20 (63 XYZ); missing joints zero-filled.
- Metadata schema v2 keeps original 30 requested indices; trained labels remap
  the 19 present signs to contiguous indices 0–18. Missing source signs were
  never substituted or synthesized. Data guide: docs/DATASET.md.
- BiLSTM: 180 input, 128 hidden, 2 bidirectional layers, dropout .4, linear 256→19.
  Shoulder midpoint/3D shoulder-distance normalization is shared between training
  and inference; invalid anchors mask frames. Training-only augmentation retained.
- Active assets: models/model_gislr_19class.pth, models/labels_19class.json,
  models/model_gislr_19class.config.json (shoulder_xyz_v1); training history is
  data/training_history.json. Best expanded 50-epoch val accuracy 71.1029% at
  epoch 45; final train 98.3912%, val 67.1324%. Live accuracy is unverified.
- Labels in index order: hello, bye, thankyou, please, yes, no, bad, water, drink, food, wait, home, see, look, go, tomorrow, where, who, man.

## Inference and gestures
- FastAPI loads strict weights_only state_dict, eval/inference_mode, shared CPU
  model lock. Per connection: dedicated worker-thread Holistic tracker, raw
  (1,64,180) FIFO. Static/missing hands bypass classification at variance .015.
- Warmup 64 frames; same label confidence >.70 for five continuous inference
  frames; adjacent duplicates suppressed; maximum 100 accepted glosses.
- Both wrists above their shoulders for 15 processed frames triggers submission.
  Raised-hand frames bypass classification, gesture latches until observed hands
  down, and empty gloss lists do not translate. No idle-timer sentence boundary.
- /ws/sign-language emits validated type=gloss_update, status (IDLE/SIGNING/
  SUBMITTING), sentence_complete or error, plus compatibility fields. One frame
  in flight; SUBMITTING is intermediate until completion/error. Ping/pong supported.
- /ws/two-way preserves private stage/submit drafts and room-scoped confirmed
  messages for the three-column frontend. Room IDs/sender strings are routing,
  not authenticated roles. Single worker; no persistence/replay or external pubsub.
- frontend/ mounts at http://localhost:8000; /health reports readiness. /webcam
  redirects to /. Sam controls left, confirmed chat center, Steve voice right.
  Web Speech drafts and sign previews never enter chat until confirmed. Optional
  TTS stops mic capture. Reference CLI remains a JPEG client with no local model.

## Environment, providers and security
- Root .env resolves to ENV_ROOT=backend ROOT.parent, explicitly loaded with
  override=True at server startup and each translation; reference CLI also uses
  root .env. Template is root .env.example; real keys were not changed or exposed.
- Translation: AsyncGroq llama3-8b-8192 → llama-3.1-8b-instant, then Gemini
  gemini-1.5-flash via asyncio.to_thread; only after failure of every available
  attempt returns raw joined glosses. Groq 3s SDK / 4s async per model, Gemini
  12s HTTP / 12s async, retries=0; combined 20s budgets within 30s client limits.
- Request/model errors and missing credentials do not break the socket. Default
  logs omit transcript/provider body content; LLM_DEBUG=1 enables redacted details.
  Requested legacy provider IDs have not been verified available live.
- HTTP CORS allow_origins=['*'], allow_credentials=False by default; override with
  SIGN_HTTP_ORIGINS. WebSockets independently check SIGN_ALLOWED_ORIGINS, default
  localhost:8000/3000/5173 and 127.0.0.1:8000. Native clients may omit Origin.
  Optional SIGN_WS_TOKEN native Bearer auth remains; browser session/role auth
  and gateway rate limiting are required before public deployment.
- JPEG-only decode, 2MiB JPEG / 3MiB JSON / 2048px side / 4MP limits, 8 active
  clients, 30s idle timeout; frontend-only static serving, CSP/nosniff/no-referrer.
- Root .env mode 0600. No hardcoded credentials found in scanned project text.
  Ignores cover .env/*.env, Python caches, both virtualenvs, .DS_Store and heavy
  processed/raw/run caches. No Git repository exists here; ignore rules were
  verified with 11 cases in a disposable repository, not via existing history.

## 2026-11-09 cleanup and audit completion
- Removed obsolete pilot/placeholder/validator/archive scripts, three obsolete
  Python tests, duplicate web/ UI, legacy browser test and retired COLAB.md.
  Moved current docs under docs/ and preprocessing under scripts/_shared/.
- Processed cache ~343MiB; raw cache ~4GiB. All 7,327 NPY files remain. SHA-256
  hashes of weights/label/config, metadata and training history unchanged.
- Pillow upgraded from 12.3.0 to patched 13.0.0; requirements require >=13,<14.
  Active runtime pip-audit now reports one unique remaining protobuf advisory
  (two duplicate records): PYSEC-2026-1805 / CVE-2026-0994. MediaPipe 0.10.21
  requires protobuf<5; reported fixed versions 5.29.6 / 6.33.5 conflict. A compatible
  MediaPipe/protobuf migration remains a production blocker, not a clean audit.
- pip check also reports the existing MediaPipe platform metadata incompatibility,
  despite passing real Holistic smoke. Audit scope/limitations in docs/SECURITY_AUDIT.md.
- Passed 41 runtime/training/client Python tests + 6 extraction Python tests + 3
  frontend Node tests (50 total); targeted provider-log regression also passed.
  CLI help for extract/train/webcam verified. Full single-env discovery requires
  both data and runtime dependencies; existing workspace uses split environments.
- No live camera, microphone, TTS audio, model-provider request, public deployment,
  load/penetration test or data-environment vulnerability scan performed.
- Evidence: data/reports/cleanup_audit.json, runtime_versions.json,
  dependency_audit_before.json and dependency_audit.json. Restart normal server
  to load typed events, environment, provider and security changes.
