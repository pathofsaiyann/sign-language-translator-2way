# Backend protocol

The maintained API contract is [FRONTEND_INTEGRATION.md](../../FRONTEND_INTEGRATION.md).
Setup, architecture and gesture controls are in [README.md](../../README.md).

## Runtime details

- Python 3.11/3.12; use requirements-server.txt from the backend directory.
- Loads root .env with override=True at startup and before each translation.
- Each connection owns a dedicated worker-thread Holistic tracker and raw XYZ
  (1,64,180) float32 FIFO. Shared PyTorch predictions are serialized under a lock.
- Pose 0–17, left/right hand 0–20; shoulder normalization occurs at model boundary.
- Hand variance threshold .015, >.70 same-label confidence for five frames,
  adjacent duplicate suppression, 15-frame raised-wrists submit with hands-down rearm.
- Canonical typed inference events: gloss_update, status, sentence_complete, error.
  /ws/two-way preserves its typed room/staging extension for the three-column UI.
- Translation: AsyncGroq llama3-8b-8192 then llama-3.1-8b-instant; Gemini
  gemini-1.5-flash via asyncio.to_thread last; raw glosses if all fail. Requested
  model availability is provider-dependent and has not been verified live.
- Groq 3s SDK/4s async per attempt; Gemini 12s HTTP/12s async; no SDK retries.
  Combined await budgets 20s, within the client 30s response deadline.
- Default logs omit glosses/sentences and provider response bodies. LLM_DEBUG=1
  enables diagnostic content with configured keys redacted. Never enable it for
  real conversations without considering transcript retention/access.
- HTTP CORS wildcard without credentials is the development default. Production
  origins can be set through SIGN_HTTP_ORIGINS. WebSocket origins are independently
  controlled by SIGN_ALLOWED_ORIGINS; absent Origin is accepted for native clients.
- Root-only static frontend mount; datasets, metadata, model assets and .env are
  not served. /webcam redirects to /; its duplicate legacy assets were removed.

[Security audit](SECURITY_AUDIT.md) describes dependency and deployment limitations.
