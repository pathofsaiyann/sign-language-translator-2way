# Cleanup and security audit — 2026-11-09

## Scope and result

Reviewed the current GISLR extraction/training/inference code, reference webcam
client, frontend transport/rendering, secrets handling, static serving, WebSocket
contracts, local caches and the active Python 3.12 runtime dependencies. This is
a source/dependency audit with automated regression tests, not a penetration test
or a certification for public production deployment.

**The handoff has a remaining production dependency blocker:** the MediaPipe
Holistic pin constrains protobuf below the advisory's fixed versions. Retain the
local prototype while planning a compatible MediaPipe/runtime migration; do not
claim a clean vulnerability scan.

## Changes completed

| Area | Finding | Remediation / evidence |
| --- | --- | --- |
| Legacy execution paths | Pilot extractor/trainer/validator, placeholder trainer and experimental archive helper remained | Removed five obsolete scripts, their three tests and retired Colab instructions; four core entrypoints remain |
| Shared preprocessing | Training/inference require identical shoulder normalization | Retained in scripts/_shared/preprocessing.py; imports and parity tests updated |
| Duplicate browser UI | A second /webcam page diverged from the maintained frontend | Removed web/ and its obsolete JS test; /webcam redirects to / |
| Secrets | Root .env is private; scanner found no provider credential literals elsewhere | Root .env.example is a placeholder-only template; .gitignore covers .env, *.env and credential variants |
| Environment path | Reference CLI looked for backend .env | Both server/CLI resolve workspace-root .env explicitly with override=True |
| Transcript logging | Glosses, sentences and provider response bodies were logged by default | Diagnostic text requires LLM_DEBUG=1; default errors expose exception class/status, with configured keys redacted in debug mode |
| Public event contract | Inference events lacked a consistent type discriminator | Pydantic status/gloss_update/sentence_complete/error schemas, validated before emission; compatibility fields retained |
| Development CORS | HTTP middleware was absent | CORSMiddleware wildcard without credentials by default; configurable production HTTP origins |
| WebSocket Origin | Any browser Origin was accepted by default | Independent local-development allowlist; rejects other browser origins, with SIGN_ALLOWED_ORIGINS override |
| Message/LLM growth | A connection could accumulate indefinitely many glosses | Maximum 100 accepted glosses; maximum 2000-character model sentence; existing JPEG/JSON/connections/time limits retained |
| Image plugins | Image.open initially tried plugins before format validation | Decoder now permits JPEG format only before parsing; 2MiB/2048px/4MP checks retained |
| Pillow advisories | Runtime 12.3.0 had advisory findings | Upgraded active runtime to 13.0.0; requirements-server.txt now requires >=13,<14; post-update regression and advisory scan recorded |
| Browser/static exposure | Frontend was the only static mount, but no explicit response headers | Retained frontend-only static mount; added nosniff, no-referrer and CSP/frame-ancestor restrictions; traversal/secret/model paths tested |
| Dataset/version control | 343MiB tensors + 4GiB raw cache lacked full ignore rules | Ignored processed/raw/run backups without deleting them; metadata, weights/config and small audit artifacts preserved |

Pillow release evidence: [13.0.0 security fixes](https://pillow.readthedocs.io/en/latest/releasenotes/13.0.0.html).
HTTP/WS boundaries follow the [FastAPI CORS documentation](https://fastapi.tiangolo.com/tutorial/cors/)
and [OWASP WebSocket guidance](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html).

## Dependency evidence and unresolved findings

`data/reports/dependency_audit_before.json` and `dependency_audit.json` contain
pip-audit results for `.venv-webcam` site-packages. The initial scan reported
41 advisory records in Pillow 12.3.0 and protobuf 4.25.9. Advisory records can
overlap through aliases or duplicates; this is not 41 independently exploitable
application vulnerabilities. Pillow was upgraded to 13.0.0. The post-remediation scan reports only protobuf:
two records for one unique advisory; no Pillow findings remain.

Protobuf finding: **PYSEC-2026-1805 / CVE-2026-0994 / GHSA-7gcm-g887-7qv7**,
reported twice by the initial service response. The advisory concerns
json_format.ParseDict recursion limits for nested Any messages. Fixed versions
reported by the scanner are 5.29.6 / 6.33.5; installed MediaPipe 0.10.21 requires
protobuf <5,>=4.25.3. Forcing a conflicting protobuf version would leave an
unsupported inference dependency graph and was not done. This application accepts
JPEG/ordinary JSON and does not expose a protobuf JSON/Any ingestion API; that
reduces direct exposure but does not establish dependency-wide non-exploitability.

The existing environment also reports `mediapipe 0.10.21 is not supported on this
platform` from `pip check`, despite successful import and real Holistic smoke
execution. Recreate and validate the pinned Python 3.11/3.12 environment on the
actual deployment host; this metadata/platform issue remains recorded.

Dependency auditing covered the active runtime, not the separate Python 3.13 data
environment, OS/graphics drivers, browser engines or system libraries. Re-run the
scanner during release builds and lock the final supported deployment environment.
Provider model availability was not checked live; requested legacy model IDs may
fail, and bounded fallbacks remain necessary.

## Deployment work remaining

- Implement authenticated browser sessions and role/room authorization before
  public multi-user use. The current sender values and room IDs are UI routing,
  not authenticated identities. Optional native Bearer auth remains supported.
- Put TLS, Origin policy, per-user/IP rate limits and connection limits at a
  trusted gateway. Wildcard HTTP CORS is a requested development default;
  restrict SIGN_HTTP_ORIGINS for public hosting and keep credentials disabled
  unless an explicit credentialed Origin policy is implemented.
- Use one server worker for current in-memory rooms, or add an authenticated
  external pub/sub/session backend. No persistent message replay/history exists.
- Keep model assets trusted. weights_only=True and allow_pickle=False are used;
  do not accept user-uploaded checkpoints. Revalidate live signing/microphone/TTS
  and language intent before an accessibility-critical deployment.
- CPU inference and synchronous image work run off the ASGI loop, but native
  SDK/model work cannot be force-killed by coroutine cancellation. Existing
  request limits/timeouts do not replace process isolation or load testing.

## Audit artifacts

- `data/reports/cleanup_audit.json`: removed/moved files, preserved asset hashes,
  final secret/cache checks and test summaries.
- `data/reports/dependency_audit_before.json`: initial runtime advisory records.
- `data/reports/dependency_audit.json`: runtime re-scan after Pillow remediation.
- `data/reports/runtime_versions.json`: installed runtime versions and constraints.

Private root .env and virtual environments are intentionally retained. Source
cache cleanup excludes installed virtual-environment packages. No dataset tensors,
parquet files, model weights, label/config assets or training history were deleted.
