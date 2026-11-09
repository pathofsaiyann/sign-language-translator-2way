"""GISLR WebSocket inference. Run: python scripts/server_websocket.py (see workspace-root FRONTEND_INTEGRATION.md)."""
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time

import anyio
import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

try:
    from .train_gislr import GISLRBiLSTM
    from ._shared.preprocessing import normalize_features, NORMALIZATION
    from ._shared.events import public_event
except ImportError:
    from train_gislr import GISLRBiLSTM
    from _shared.preprocessing import normalize_features, NORMALIZATION
    from _shared.events import public_event

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT.parent


def load_environment(directory=ENV_ROOT):
    # Explicit workspace-root path; root .env values override stale exported values.
    load_dotenv(dotenv_path=directory / '.env', override=True)


load_environment()
LOG = logging.getLogger('gislr.websocket')
DETECTING = {'status': 'detecting', 'prediction': None}
IDLE = {'status': 'idle', 'state': 'IDLE', 'prediction': None}
LLM_TIMEOUT_SECONDS = 4.0
GEMINI_TIMEOUT_SECONDS = 12.0
GROQ_MODELS = ['llama3-8b-8192', 'llama-3.1-8b-instant']
GEMINI_MODEL = 'gemini-1.5-flash'

GLOSS_PROMPT = 'You are a Sign Language Gloss-to-English translator. Convert input glosses (e.g. ["hello", "water", "please"]) into a fluent English sentence (e.g. "Hello, please give me water."). Output ONLY the sentence without markdown or quotes.'


def sentence_text(text, provider):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f'{provider} returned no sentence text (empty or blocked output)')
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    if not text or len(text) > 2000:
        raise ValueError(f'{provider} returned an empty or oversized sentence')
    return text


def log_llm_error(provider, exc, keys):
    # Provider errors can include request details; never emit configured secrets.
    debug = os.getenv('LLM_DEBUG', '0') == '1'
    detail = f'{type(exc).__name__}: {str(exc)}' if debug else type(exc).__name__
    code = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
    response = getattr(exc, 'response', None)
    code = code or getattr(response, 'status_code', None)
    if debug and response is not None:
        code = code or getattr(response, 'status_code', None)
        try:
            body = response.text
        except Exception:
            body = None  # Unread/streaming responses must not break fallback.
        if isinstance(body, str) and body:
            detail += f' | response={body}'
    if code is not None:
        detail += f' | status={code}'
    for key in keys:
        if key:
            detail = detail.replace(key, '[REDACTED]')
    timed_out = (isinstance(exc, (TimeoutError, httpx.TimeoutException)) or
                 str(code) == '504' or getattr(response, 'status_code', None) == 504 or
                 getattr(exc, 'status', None) == 'DEADLINE_EXCEEDED' or
                 'DEADLINE_EXCEEDED' in str(exc))
    kind = 'TIMEOUT' if timed_out else 'ERROR'
    print(f'[{provider} {kind}] {detail}', flush=True)


async def build_sentence(gloss_sequence: list[str]) -> str:
    try:
        load_environment()  # Refresh explicit root .env with override=True.
    except (OSError, UnicodeError):
        LOG.warning('Could not refresh root .env; using existing environment')
    glosses = list(gloss_sequence)
    if not glosses:
        return ''
    groq_key = os.getenv('GROQ_API_KEY', '').strip()
    gemini_key = os.getenv('GEMINI_API_KEY', '').strip()
    keys = (groq_key, gemini_key)
    if os.getenv('LLM_DEBUG', '0') == '1':
        print(f'[LLM] Translating glosses: {glosses}', flush=True)

    async def request_groq(model):
        from groq import AsyncGroq
        async with AsyncGroq(api_key=groq_key, timeout=3.0, max_retries=0) as client:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': GLOSS_PROMPT},
                          {'role': 'user', 'content': json.dumps(glosses)}],
                temperature=0, max_tokens=1024)
            return sentence_text(completion.choices[0].message.content, 'Groq')

    def request_gemini():
        from google import genai
        from google.genai import types
        with genai.Client(api_key=gemini_key, vertexai=False,
                          http_options=types.HttpOptions(timeout=12_000,
                              retry_options=types.HttpRetryOptions(attempts=1))) as client:
            completion = client.models.generate_content(
                model=GEMINI_MODEL, contents=json.dumps(glosses),
                config=types.GenerateContentConfig(
                    system_instruction=GLOSS_PROMPT, temperature=0,
                    max_output_tokens=1024))
            return sentence_text(completion.text, 'Gemini')

    if groq_key:
        for model in GROQ_MODELS:
            try:
                sentence = await asyncio.wait_for(request_groq(model), timeout=LLM_TIMEOUT_SECONDS)
                print(f'[LLM SUCCESS] Model: {model}' + (f' | Output: {sentence}' if os.getenv('LLM_DEBUG', '0') == '1' else ''), flush=True)
                return sentence
            except Exception as exc:
                log_llm_error(f'GROQ {model}', exc, keys)
    else:
        log_llm_error('GROQ', ValueError('GROQ_API_KEY is not configured'), keys)

    try:
        if not gemini_key:
            raise ValueError('GEMINI_API_KEY is not configured')
        sentence = await asyncio.wait_for(asyncio.to_thread(request_gemini), timeout=GEMINI_TIMEOUT_SECONDS)
        print(f'[LLM SUCCESS] Model: {GEMINI_MODEL}' + (f' | Output: {sentence}' if os.getenv('LLM_DEBUG', '0') == '1' else ''), flush=True)
        return sentence
    except Exception as exc:
        log_llm_error(f'GEMINI {GEMINI_MODEL}', exc, keys)

    print('[LLM FALLBACK] Returning raw gloss sequence.', flush=True)
    return ' '.join(glosses)


MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_MESSAGE_BYTES = 3 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    weights: Path = ROOT / 'models/model_gislr_19class.pth'
    labels: Path = ROOT / 'models/labels_19class.json'
    max_clients: int = 8
    idle_timeout: float = 30.0
    max_frame_gap: float = 3.0
    motion_threshold: float = .015
    submit_frames: int = 15
    origins: tuple = ('http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:3000', 'http://localhost:5173')
    http_origins: tuple = ('*',)
    token: str = ''

    @classmethod
    def from_env(cls):
        return cls(origins=tuple(origin.strip() for origin in os.getenv('SIGN_ALLOWED_ORIGINS', ','.join(cls.origins)).split(',') if origin.strip()),
                   http_origins=tuple(origin.strip() for origin in os.getenv('SIGN_HTTP_ORIGINS', '*').split(',') if origin.strip()),
                   token=os.getenv('SIGN_WS_TOKEN', ''))


def extract_features(results):
    """Exactly match GISLR training: pose 0..17, left/right hand 0..20, XYZ."""
    output = np.zeros((60, 3), dtype=np.float32)
    for name, count, offset in [('pose_landmarks', 18, 0),
                                ('left_hand_landmarks', 21, 18),
                                ('right_hand_landmarks', 21, 39)]:
        detected = getattr(results, name, None)
        if detected is None:
            continue
        for index, point in enumerate(detected.landmark[:count]):
            xyz = (point.x, point.y, point.z)
            if np.isfinite(xyz).all():
                output[offset + index] = xyz
    return output.reshape(180)


def decode_frame(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('frame'), str):
        raise ValueError('Expected a JSON object with a Base64 JPEG frame string')
    encoded = payload['frame']
    prefix = 'data:image/jpeg;base64,'
    if encoded.startswith(prefix):
        encoded = encoded[len(prefix):]
    if not encoded or len(encoded) > 4 * ((MAX_JPEG_BYTES + 2) // 3):
        raise ValueError('Frame is empty or exceeds the 2 MiB JPEG limit')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_JPEG_BYTES:
            raise ValueError('JPEG exceeds 2 MiB')
        with Image.open(BytesIO(raw), formats=['JPEG']) as image:
            if image.format != 'JPEG':
                raise ValueError('Only JPEG frames are accepted')
            if max(image.size) > 2048 or image.width * image.height > 4_000_000:
                raise ValueError('JPEG dimensions exceed 2048 per side or 4 million pixels')
            return np.array(image.convert('RGB'), dtype=np.uint8)
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid Base64 JPEG frame') from exc


class Debouncer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.index, self.count = None, 0

    def update(self, index, confidence):
        if not np.isfinite(confidence) or confidence <= .70:
            self.reset()
            return False
        self.count = min(5, self.count + 1) if index == self.index else 1
        self.index = index
        return self.count >= 5


def holistic_factory():
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Install requirements-server.txt with Python 3.11/3.12 for legacy Holistic')
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        enable_segmentation=False, refine_face_landmarks=False,
        min_detection_confidence=.5, min_tracking_confidence=.5)


class Runtime:
    def __init__(self, settings):
        mapping = json.loads(settings.labels.read_text())
        if (not isinstance(mapping, dict) or len(mapping) != 19 or
                any(not isinstance(k, str) or not k or type(v) is not int for k, v in mapping.items()) or
                set(mapping.values()) != set(range(19))):
            raise ValueError('Expected 19 unique sign labels mapped to indices 0..18')
        self.labels = [name for name, index in sorted(mapping.items(), key=lambda item: item[1])]
        config_path = settings.weights.with_suffix('.config.json')
        if not config_path.is_file():
            raise FileNotFoundError(f'Missing preprocessing contract: {config_path}')
        config = json.loads(config_path.read_text())
        if config.get('preprocessing') not in (None, NORMALIZATION):
            raise ValueError('Unsupported checkpoint preprocessing contract')
        self.normalize = config.get('preprocessing') == NORMALIZATION
        if self.normalize and config.get('class_to_index') != mapping:
            raise ValueError('Checkpoint config and label mapping disagree')
        self.model = GISLRBiLSTM()
        self.model.load_state_dict(torch.load(settings.weights, map_location='cpu', weights_only=True), strict=True)
        self.model.eval()
        self.lock = threading.Lock()
        with torch.inference_mode():
            self.model(torch.zeros(1, 64, 180))

    def predict(self, buffer):
        with self.lock, torch.inference_mode():
            probabilities = self.model(torch.from_numpy(normalize_features(buffer) if self.normalize else buffer)).softmax(dim=1)[0]
            confidence, index = probabilities.max(dim=0)
        return int(index), float(confidence)


def raised_hands(results):
    """True/False for a reliable pose; None for absent/invalid joint observations."""
    pose = getattr(results, 'pose_landmarks', None)
    if pose is None or len(pose.landmark) <= 16:
        return None
    points = [pose.landmark[index] for index in (11, 12, 15, 16)]
    for point in points:
        if (not np.isfinite([point.x, point.y, point.z]).all() or
                not np.isfinite(getattr(point, 'visibility', 1.)) or
                getattr(point, 'visibility', 1.) < .5):
            return None
    left_shoulder, right_shoulder, left_wrist, right_wrist = points
    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y


class ClientSession:
    """Created, used and closed on a single dedicated worker thread per connection."""
    def __init__(self, runtime, factory, settings):
        self.runtime, self.settings = runtime, settings
        self.holistic = factory()
        self.buffer = np.zeros((1, 64, 180), dtype=np.float32)
        self.debouncer = Debouncer()
        self.frames = 0
        self.last_frame = None
        self.gloss_sequence = []
        self.submit_armed = True
        self.submit_count = 0
        self.state = "WAITING"

    def reset(self):
        self.buffer.fill(0)
        self.frames = 0
        self.debouncer.reset()
        self.submit_count = 0
        self.state = "WAITING"

    def complete_sentence(self):
        self.gloss_sequence.clear()
        self.reset()

    def process(self, payload):
        response = self._process(payload)
        response['glosses'] = list(self.gloss_sequence)
        return response

    def _process(self, payload):
        now = time.monotonic()
        if self.last_frame is not None and now - self.last_frame > self.settings.max_frame_gap:
            self.reset()
        self.last_frame = now
        try:
            image = decode_frame(payload)
        except ValueError:
            self.reset()
            raise
        results = self.holistic.process(image)
        raised = raised_hands(results)
        if raised is True:
            self.submit_count = min(self.settings.submit_frames, self.submit_count + 1)
            self.buffer.fill(0)
            self.frames = 0
            self.debouncer.reset()
            if self.submit_count >= self.settings.submit_frames and self.submit_armed:
                self.submit_armed = False
                if self.gloss_sequence:
                    self.state = 'SUBMITTING_SENTENCE'
                    return {'status': 'sentence_pending', 'state': self.state}
            self.state = 'SUBMIT_GESTURE' if self.submit_armed else 'WAITING_FOR_HANDS_DOWN'
            return {'status': 'detecting', 'prediction': None, 'state': self.state,
                    'submit_frames': self.submit_count}
        self.submit_count = 0
        if raised is False:
            self.submit_armed = True  # require an observed hands-down pose to rearm
        features = extract_features(results)
        self.buffer[0, :-1] = self.buffer[0, 1:].copy()
        self.buffer[0, -1] = features
        self.frames = min(64, self.frames + 1)
        # Temporal variance for each coordinate, averaged over the 126 hand
        # dimensions. Spatial variance between different joints is not motion.
        hands = self.buffer[0, -self.frames:, 54:180]
        variance = float(np.var(hands, axis=0).mean())
        resting = not features[54:180].any() or variance < self.settings.motion_threshold
        if resting:
            self.state = 'IDLE'
            self.debouncer.reset()
            return dict(IDLE)
        self.submit_count = 0
        if self.frames < 64:
            self.state = 'WAITING'
            self.debouncer.reset()
            return dict(DETECTING)
        self.state = 'DETECTING'
        index, confidence = self.runtime.predict(self.buffer)
        if not self.debouncer.update(index, confidence):
            return dict(DETECTING)
        word = self.runtime.labels[index]
        if not self.gloss_sequence or word != self.gloss_sequence[-1]:
            if len(self.gloss_sequence) >= 100:
                return {'status': 'error', 'prediction': None, 'error': '100-gloss limit reached; submit your sentence'}
            self.gloss_sequence.append(word)
        self.state = 'SIGNING'
        return {'status': 'prediction', 'prediction': word, 'confidence': confidence}

    def close(self):
        self.holistic.close()


def create_app(settings=None, runtime_factory=Runtime, detector_factory=holistic_factory):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        # Fail startup on invalid checkpoint/labels or unavailable Holistic assets.
        app.state.runtime = await asyncio.to_thread(runtime_factory, settings)
        def probe():
            detector = detector_factory()
            detector.close()
        await asyncio.to_thread(probe)
        app.state.clients = 0
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.http_origins),
                       allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

    @app.middleware('http')
    async def response_security_headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'"
        return response

    rooms = {}  # Active sockets only; draft/landmark state remains connection-local.

    @app.get('/health')
    async def health():
        return {'status': 'ready', 'classes': 19, 'sequence_shape': [64, 180]}

    @app.get('/webcam', include_in_schema=False)
    async def webcam_page():
        return RedirectResponse('/')

    @app.websocket('/ws/two-way')
    @app.websocket('/ws/sign-language')
    async def sign_language(websocket: WebSocket):
        origin = websocket.headers.get('origin')
        if origin is not None and '*' not in settings.origins and origin not in settings.origins:
            await websocket.close(code=1008)
            return
        if settings.token and not secrets.compare_digest(websocket.headers.get('authorization', '').encode(), f'Bearer {settings.token}'.encode()):
            await websocket.close(code=1008)
            return
        if app.state.clients >= settings.max_clients:
            await websocket.close(code=1013)
            return
        two_way = websocket.url.path == '/ws/two-way'
        room = websocket.query_params.get('room', 'main')
        if two_way and (not 1 <= len(room) <= 64 or
                        any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in room)):
            await websocket.close(code=1008)
            return
        app.state.clients += 1
        send_lock = asyncio.Lock()
        stage_task = None
        cached_draft = {}

        async def emit(payload):
            async with send_lock:
                await websocket.send_json(public_event(payload))

        async def dispatch(sender, text, request_id=None):
            message = {'type': 'message', 'status': 'message_sent',
                       'message_id': secrets.token_hex(12), 'sender': sender,
                       'recipient': 'steve' if sender == 'sam' else 'sam',
                       'text': text, 'request_id': request_id}
            async def deliver(send):
                try:
                    await asyncio.wait_for(send(message), timeout=3)
                except (TimeoutError, RuntimeError, WebSocketDisconnect, OSError):
                    pass  # A failed peer must not break the sender's connection.
            await asyncio.gather(*(deliver(send) for send in list(rooms.get(room, {}).values())))

        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gislr-client')
        loop = asyncio.get_running_loop()
        holder = {}
        def initialize():
            holder['session'] = ClientSession(app.state.runtime, detector_factory, settings)
            return holder['session']
        def cleanup():
            if 'session' in holder:
                try:
                    holder['session'].close()
                except Exception:
                    LOG.exception('Holistic cleanup failed')
        try:
            await websocket.accept()
            session = await loop.run_in_executor(worker, initialize)
            if two_way:
                rooms.setdefault(room, {})[websocket] = emit
                await emit({'type': 'ready', 'room': room})

            async def translate(glosses, auto_send=False):
                try:
                    sentence = (cached_draft['text'] if cached_draft.get('glosses') == glosses
                                else await build_sentence(glosses))
                    cached_draft.update(text=sentence, glosses=list(glosses))
                    await emit({'type': 'draft', 'status': 'draft_ready', 'sentence': sentence,
                                'glosses': glosses, 'auto_send': auto_send})
                    if auto_send:
                        await dispatch('sam', sentence)
                        await loop.run_in_executor(worker, session.complete_sentence)
                        cached_draft.clear()
                        await emit({'type': 'sent', 'sender': 'sam', 'glosses': []})
                except Exception:
                    LOG.exception('Two-way translation failed')
                    await emit({'type': 'error', 'error': 'Translation unavailable; draft preserved'})

            async def begin_translation(glosses, auto_send=False):
                nonlocal stage_task
                if stage_task and not stage_task.done():
                    await emit({'type': 'error', 'error': 'Translation already in progress'})
                elif not glosses:
                    await emit({'type': 'error', 'error': 'No glosses to translate'})
                else:
                    await emit({'type': 'processing', 'status': 'submitting_sentence', 'glosses': glosses})
                    stage_task = asyncio.create_task(translate(list(glosses), auto_send))
            while True:
                message = await asyncio.wait_for(websocket.receive(), timeout=settings.idle_timeout)
                if message['type'] == 'websocket.disconnect':
                    break
                text = message.get('text')
                if text is None:
                    await loop.run_in_executor(worker, session.reset)
                    await emit({'status': 'error', 'prediction': None, 'error': 'Text JSON messages required'})
                    continue
                if len(text.encode('utf-8')) > MAX_MESSAGE_BYTES:
                    await websocket.close(code=1009, reason='Message too large')
                    break
                try:
                    payload = json.loads(text)
                except (ValueError, RecursionError):
                    await loop.run_in_executor(worker, session.reset)
                    await emit({'status': 'error', 'prediction': None, 'error': 'Invalid JSON'})
                    continue
                if isinstance(payload, dict) and payload.get('type') == 'ping':
                    await emit({'type': 'pong'})
                    continue
                if two_way:
                    if not isinstance(payload, dict):
                        await emit({'type': 'error', 'error': 'Expected a JSON object'})
                        continue
                    kind = payload.get('type', 'frame' if 'frame' in payload else '')
                    if kind == 'ping':
                        await emit({'type': 'pong'})
                        continue
                    if kind == 'send':
                        sender, text = payload.get('sender'), payload.get('text')
                        request_id = payload.get('request_id')
                        if (sender not in ('sam', 'steve') or not isinstance(text, str) or
                                not text.strip() or len(text) > 2000 or
                                (request_id is not None and (not isinstance(request_id, str) or len(request_id) > 80))):
                            await emit({'type': 'error', 'error': 'Send requires sender sam/steve and 1–2000 text characters'})
                        elif sender == 'sam' and stage_task and not stage_task.done():
                            await emit({'type': 'error', 'error': 'Wait for the sign draft to finish'})
                        else:
                            await dispatch(sender, text.strip(), request_id)
                            if sender == 'sam':
                                await loop.run_in_executor(worker, session.complete_sentence)
                                cached_draft.clear()
                                await emit({'type': 'sent', 'sender': 'sam', 'glosses': []})
                        continue
                    if kind in ('stage', 'submit'):
                        glosses = await loop.run_in_executor(worker, lambda: list(session.gloss_sequence))
                        await begin_translation(glosses, auto_send=kind == 'submit')
                        continue
                    if kind != 'frame':
                        await emit({'type': 'error', 'error': 'Unknown message type'})
                        continue
                    if stage_task and not stage_task.done():
                        await emit({'type': 'frame_result', 'status': 'processing', 'glosses': cached_draft.get('glosses', [])})
                        continue
                try:
                    response = await loop.run_in_executor(worker, session.process, payload)
                except ValueError as exc:
                    response = {'status': 'error', 'prediction': None, 'error': str(exc)}
                if two_way:
                    await emit(dict(response, type='frame_result'))
                    if response['status'] == 'sentence_pending':
                        await begin_translation(response['glosses'], auto_send=True)
                    continue
                if response['status'] == 'sentence_pending':
                    glosses = response['glosses']
                    await emit({'status': 'submitting_sentence',
                                               'state': 'SUBMITTING_SENTENCE',
                                               'prediction': None, 'glosses': glosses})
                    sentence = await build_sentence(glosses)
                    await emit({'status': 'sentence_complete',
                                               'glosses': glosses, 'sentence': sentence})
                    # Clear only after the response was successfully emitted.
                    await loop.run_in_executor(worker, session.complete_sentence)
                else:
                    await emit(response)
        except (WebSocketDisconnect, OSError):
            pass
        except asyncio.TimeoutError:
            try:
                await websocket.close(code=1001, reason='Idle timeout')
            except (RuntimeError, WebSocketDisconnect, OSError):
                pass
        except Exception:
            LOG.exception('WebSocket inference failed')
            try:
                await websocket.close(code=1011, reason='Inference unavailable')
            except (RuntimeError, WebSocketDisconnect, OSError):
                pass
        finally:
            # Starlette/AnyIO cancellation is level-triggered: shield the entire
            # cleanup scope so disconnect cannot interrupt tracker/slot release.
            with anyio.CancelScope(shield=True):
                if two_way:
                    peers = rooms.get(room, {})
                    peers.pop(websocket, None)
                    if not peers:
                        rooms.pop(room, None)
                if stage_task:
                    stage_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await stage_task
                try:
                    await loop.run_in_executor(worker, cleanup)
                finally:
                    try:
                        await asyncio.to_thread(worker.shutdown, wait=True, cancel_futures=False)
                    finally:
                        app.state.clients -= 1


    # Mount last so explicit API/legacy routes take precedence. Only UI assets are public.
    app.mount('/', StaticFiles(directory=ROOT / 'frontend', html=True), name='frontend')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    uvicorn.run(app, host='127.0.0.1', port=8000, ws_max_size=MAX_MESSAGE_BYTES,
                ws_max_queue=1, ws_ping_interval=20, ws_ping_timeout=20)
