# Frontend integration — AIML-SIGN

## Start and connect

Start the backend using the [root quickstart](README.md#quickstart).
`GET http://localhost:8000/health` returns:

```json
{"status":"ready","classes":19,"sequence_shape":[64,180]}
```

The canonical inference endpoint is **ws://localhost:8000/ws/sign-language**.
Use `wss://` when the page is served over HTTPS. The unified app at `/` instead
uses the staging/broadcast extension **/ws/two-way?room=<id>** below.

HTTP CORS permits all origins by default with credentials disabled. The WebSocket
handshake independently checks `SIGN_ALLOWED_ORIGINS`. Defaults include
http://localhost:8000, http://127.0.0.1:8000, http://localhost:3000 and
http://localhost:5173. Set exact additional development/production origins in root
`.env`, then restart. Native clients may omit Origin. If `SIGN_WS_TOKEN` is set,
requests require `Authorization: Bearer <token>`; browsers cannot set this header
with `new WebSocket()`. The bundled browser requires that optional token unset;
public browser authentication belongs at the gateway/session layer. Never ship
Groq/Gemini keys to the frontend.

## Frame ingestion

Capture **unmirrored raw video**, draw it on an offscreen canvas, encode JPEG
(quality ~0.8), and send text JSON:

```json
{"frame":"<base64 JPEG bytes>"}
```

`data:image/jpeg;base64,...` is also accepted. Do not include overlays in submitted
pixels. Resize proportionally to a maximum side of 640px; the server accepts up
to 2048px per side / 4 million pixels. Limits: 2 MiB JPEG, 3 MiB JSON message,
8 active connections per process, 30s without incoming messages before close.

Target at most 24 FPS with **one frame in flight**. Wait for the frame's response
before sending another. Actual processing FPS may be lower. Keep only the latest
camera image; do not buffer or replay old frames after reconnect. Allow at least
30 seconds for sentence generation. A frame gap over 3s resets temporal inference
but preserves accumulated glosses. Camera warmup requires 64 frames; confidence
must then stay >0.70 for five consecutive inference frames.

## Validated inference responses

`type` is the event discriminator. The server validates these canonical events
using Pydantic in `scripts/_shared/events.py`. Legacy `status`, `prediction`,
`confidence`, `gesture_state` and `glosses` fields may also appear; ignore unknown
fields and branch on `type`.

**Accepted sign:**

```json
{"type":"gloss_update","glosses":["hello","water"],"current_sign":"water","confidence":0.91}
```

Update the live gloss display. This event acknowledges the frame; do not insert
it into the confirmed chat. Adjacent repeated signs are suppressed; nonconsecutive
repetitions are allowed. At 100 glosses, the user must submit before adding more.

**State:**

```json
{"type":"status","state":"IDLE"}
```

| State | Meaning | Send next frame? |
| --- | --- | --- |
| `IDLE` | Missing/static hands, classifier bypassed | Yes |
| `SIGNING` | Warmup, low confidence, or counting/waiting on submit gesture | Yes |
| `SUBMITTING` | Translating the accepted glosses after hands-up | Wait for completion/error |

Only the translating `SUBMITTING` event is intermediate. It is followed by:

```json
{"type":"sentence_complete","sentence":"Hello, please give me water.","glosses":["hello","water","please"]}
```

Display the sentence. It may be joined raw glosses if all providers failed. This
acknowledges the submitted frame; the server clears glosses/FIFO only after
successful emission. Holding the gesture does not resubmit. Lower at least one
wrist below its shoulder to rearm. There is no automatic idle-timer submission.

**Recoverable input error:**

```json
{"type":"error","error":"Invalid Base64 JPEG frame"}
```

Show the error and resume with a fresh frame. Invalid input resets FIFO/debounce,
not the accumulated gloss list. `{"type":"ping"}` receives `{"type":"pong"}` on
either endpoint; heartbeats are independent of frame acknowledgements.

## Minimal event listener

```js
const socket = new WebSocket('ws://localhost:8000/ws/sign-language');
let inFlight = false;
let replyDeadline;
function acknowledge() {
  inFlight = false;
  clearTimeout(replyDeadline);
}
function sendLatestCanvas(canvas) { // call at <=24 FPS; redraw from video first
  if (socket.readyState !== WebSocket.OPEN || inFlight) return;
  inFlight = true;
  socket.send(JSON.stringify({frame: canvas.toDataURL('image/jpeg', 0.8)}));
  replyDeadline = setTimeout(() => socket.close(), 30000);
}
socket.onmessage = ({data}) => {
  let event;
  try { event = JSON.parse(data); }
  catch { socket.close(); return; }
  switch (event.type) {
    case 'gloss_update':
      glossOutput.textContent = JSON.stringify(event.glosses);
      acknowledge();
      break;
    case 'status':
      statusOutput.textContent = event.state;
      if (event.state !== 'SUBMITTING') acknowledge();
      break;
    case 'sentence_complete':
      sentenceOutput.textContent = event.sentence;
      glossOutput.textContent = '[]';
      acknowledge();
      break;
    case 'error':
      statusOutput.textContent = event.error;
      acknowledge();
      break;
    case 'pong': break; // Never release an in-flight frame on heartbeat response.
    default: break;
  }
};
socket.onclose = () => { acknowledge(); /* reconnect with 0.5–5s backoff */ };
// glossOutput/statusOutput/sentenceOutput are your DOM element references.
```

Render text with `textContent`, not `innerHTML`. Stop media tracks on unmount.
Keep the camera GUI responsive while translating or reconnecting. Never retry an
uncertain confirmed send automatically; the backend has no persistent replay log.

## Two-way staging and chat extension

Connect `/ws/two-way?room=<1–64 ASCII letters/digits/_/->`. The bundled UI generates
a random room in the URL fragment and passes it in the WS query. Sharing the full
page URL joins the same room. Rooms and message history are not persistent and
are process-local; use **one worker**. Room IDs and sender names are not identity
or authorization mechanisms. Drafts/FIFO remain private to each connection.

| Client payload | Server result |
| --- | --- |
| `{"type":"frame","frame":"..."}` | `frame_result` with status/prediction/glosses; wait for it before next frame |
| `{"type":"stage"}` | `processing`, then private `draft` (`sentence`, `glosses`, `auto_send:false`) |
| `{"type":"submit"}` | `processing`, then `draft` with `auto_send:true`, confirmed `message`, then `sent` |
| `{"type":"send","sender":"sam","text":"Edited draft","request_id":"UUID"}` | Room broadcast `message`, then `sent` to the sender; clears its glosses |
| `{"type":"send","sender":"steve","text":"Confirmed speech","request_id":"UUID"}` | Room broadcast `message`; echoed request ID acknowledges the send |
| `{"type":"ping"}` | `pong` |

`ready` reports the joined room. `error` reports invalid messages, empty glosses,
or a busy translation. Text must contain 1–2000 characters. Request IDs are
optional strings up to 80 characters. Message example:

```json
{"type":"message","status":"message_sent","message_id":"server-generated-id","sender":"sam","recipient":"steve","text":"Hello, Steve.","request_id":"client-generated-id"}
```

Broadcasts include the origin connection. Deduplicate by `message_id`, and clear
only the sending draft whose `request_id` matches. Sam messages align left/green;
Steve messages right/purple. Only `message` events enter the center chat. Glosses,
interim speech and `draft` events stay in their side panels until confirmed.

Hands-up for 15 processed frames triggers submit. A matching preview translation
is reused, otherwise one is generated. The gesture sends server-accepted signs;
use the green button for manual draft edits. Preview requests never clear glosses
or broadcast text. Background translation permits Steve's confirmed sends while
Sam's translation runs; pause frame streaming during `processing` until `draft`.
The two-way `frame_result` envelope remains for the existing UI; it does not use
the canonical endpoint's `gloss_update` event.

Speech recognition uses `SpeechRecognition || webkitSpeechRecognition`, with
continuous/interim results. Stop the microphone before manual edits. TTS is opt-in
and pauses microphone capture to avoid transcribing synthesized output. Vendor
speech availability/latency and hardware permissions require manual testing.

## Failure handling and verification

Close codes: 1008 invalid Origin/auth/room, 1013 capacity reached, 1009 oversized
message, 1001 idle timeout, 1011 unexpected inference failure. JSON/decoding errors
normally return typed errors while retaining the socket. Groq network/model errors
advance to the next model; Gemini runs off the event loop with a 12s deadline;
exhaustion returns raw glosses through normal completion. Cancellation propagates.

Start with typed frames and mocked media. Verify warmup, rest bypass, five-frame
confidence, private previews, duplicate suppression, hands-up latch/rearm, two-way
broadcast isolation and reconnect cleanup. The test suite includes these cases
and a real checkpoint/Holistic smoke test. It does not establish live signing,
audio or provider-model availability. Full audit: [SECURITY_AUDIT.md](sign_language_backend/docs/SECURITY_AUDIT.md).
