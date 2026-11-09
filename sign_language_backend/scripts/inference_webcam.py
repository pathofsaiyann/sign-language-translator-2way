"""Webcam client: native OpenCV or --browser for live microphone subtitles."""
import argparse
import asyncio
import base64
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('webcam')
WINDOW = 'GISLR WebSocket webcam'
SUBMIT_INSTRUCTION = 'Raise both hands above shoulders to translate sentence'


@dataclass
class OverlayState:
    sentence: str = ''
    glosses: list = field(default_factory=list)
    pending_words: list = field(default_factory=list)
    status: str = 'Connecting...'

    def connected(self):
        self.pending_words.clear()
        self.glosses.clear()
        self.status = 'Detecting...'

    def apply(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Server response must be a JSON object')
        glosses = payload.get('glosses', payload.get('gloss_sequence'))
        if isinstance(glosses, list) and all(isinstance(word, str) for word in glosses):
            self.glosses = glosses[:100]
        if isinstance(payload.get('sentence'), str):
            self.sentence = payload['sentence'][:4000]
            self.pending_words.clear()
            self.glosses.clear()
            self.status = 'Idle'
            return
        status = str(payload.get('status', payload.get('type', ''))).lower()
        if status == 'idle' or payload.get('state') in ('IDLE', 'WAITING'):
            self.status = 'Idle'
            return
        if status in ('processing', 'sentence_pending', 'sentence_processing', 'submitting_sentence') or payload.get('state') == 'SUBMITTING':
            self.status = 'Processing sentence...'
        elif status == 'error':
            self.status = 'Server error: ' + str(payload.get('error', 'Unknown error'))[:100]
        else:
            self.status = 'Detecting...'
            word = payload.get('current_sign', payload.get('prediction'))
            if isinstance(word, str) and word:
                if not self.pending_words or word != self.pending_words[-1]:
                    self.pending_words.append(word)
                if glosses is None:
                    self.glosses = list(self.pending_words)
                self.status = 'Sign: ' + word[:100]



class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.overlay = OverlayState()
        self.latest = None
        self.serial = 0

    def publish(self, encoded):
        with self.lock:
            self.serial += 1
            self.latest = (self.serial, time.monotonic(), encoded)

    def frame_after(self, previous):
        with self.lock:
            if self.latest and self.latest[0] != previous and time.monotonic() - self.latest[1] <= .5:
                return self.latest
        return None

    def update(self, action, *args):
        with self.lock:
            getattr(self.overlay, action)(*args)

    def status(self, text):
        with self.lock:
            self.overlay.status = text

    def snapshot(self):
        with self.lock:
            return self.overlay.sentence, list(self.overlay.glosses), self.overlay.status


async def stream_connection(socket, shared, fps, response_timeout):
    previous, deadline = 0, 0.
    while True:
        await asyncio.sleep(max(0, deadline - time.monotonic()))
        latest = shared.frame_after(previous)
        if latest is None:
            await asyncio.sleep(.01)
            continue
        previous, _, encoded = latest
        deadline = time.monotonic() + 1 / fps
        await socket.send(json.dumps({'frame': encoded}))
        # One request in flight matches the server's frame/completion protocol.
        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=response_timeout)
            try:
                payload = json.loads(raw)
                shared.update('apply', payload)
            except (ValueError, TypeError, RecursionError):
                shared.status('Invalid server response')
                break
            if not (payload.get('status') in ('processing', 'sentence_pending', 'sentence_processing', 'submitting_sentence') or payload.get('state') == 'SUBMITTING'):
                break



async def connection_loop(shared, url, fps=24, response_timeout=30, token='', origin=None):
    delay = .5
    while True:
        try:
            shared.status('Connecting...')
            headers = {'Authorization': f'Bearer {token}'} if token else None
            async with connect(url, additional_headers=headers, origin=origin,
                               open_timeout=3, close_timeout=1, ping_interval=20,
                               ping_timeout=20, max_size=65536, max_queue=1,
                               compression=None, proxy=None) as socket:
                shared.update('connected')
                delay = .5
                await stream_connection(socket, shared, fps, response_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Avoid logging URLs, authorization headers or received sentences.
            LOG.warning('WebSocket unavailable (%s); reconnecting', type(exc).__name__)
            shared.status('Server unavailable - reconnecting...')
            await asyncio.sleep(delay)
            delay = min(5., delay * 2)


class NetworkWorker:
    def __init__(self, shared, **options):
        self.shared, self.options = shared, options
        self.loop = None
        self.task = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name='webcam-websocket', daemon=True)

    def _run(self):
        async def run():
            self.loop = asyncio.get_running_loop()
            self.task = asyncio.current_task()
            if not self.stopped.is_set():
                await connection_loop(self.shared, **self.options)
        try:
            asyncio.run(run())
        except asyncio.CancelledError:
            pass
        except Exception:
            LOG.exception('WebSocket worker stopped')
            self.shared.status('Network worker stopped')

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.loop is not None and self.task is not None:
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:
                pass  # loop already closed
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            LOG.warning('Network shutdown exceeded five seconds')


def encode_frame(cv2, image, quality=80, max_side=640):
    height, width = image.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        image = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))))
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return base64.b64encode(jpeg.tobytes()).decode('ascii')


def wrap_text(cv2, text, width, scale, max_lines=4):
    # Character wrapping also handles long tokens and caps overlay work.
    lines, line = [], ''
    for char in ' '.join(text.split())[:4000]:
        if cv2.getTextSize(line + char, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > width and line:
            lines.append(line)
            line = ''
            if max_lines is not None and len(lines) == max_lines:
                lines[-1] = lines[-1][:-3] + '...'
                return lines
        line += char
    if line:
        lines.append(line)
    return lines


def draw_overlay(cv2, image, snapshot):
    sentence, glosses, status = snapshot
    width = image.shape[1]
    scale = max(.4, min(.9, width / 800))
    lines = wrap_text(cv2, sentence or 'Translation: waiting for a sentence...', max(30,width-30), scale, max_lines=None)
    # Paginate long translations rather than permanently truncating their text.
    if len(lines) > 4:
        page = int(time.monotonic() / 4) % math.ceil(len(lines) / 4)
        lines = lines[page*4:(page+1)*4]
    spacing = max(20, round(32*scale))
    panel_height = min(image.shape[0], spacing * (len(lines)+5) + 15)
    cv2.rectangle(image,(0,0),(width,panel_height),(20,20,20),-1)
    y = spacing
    for line in lines:
        cv2.putText(image,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(80,255,120),2,cv2.LINE_AA)
        y += spacing
    # Sentence is prominent; glosses/status occupy smaller secondary lines.
    gloss_line = wrap_text(cv2,'Accumulated Glosses: ['+', '.join(glosses)+']',max(30,width-30),scale*.7,1)
    cv2.putText(image,gloss_line[0] if gloss_line else 'Accumulated Glosses: []',(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale*.7,(240,240,240),1,cv2.LINE_AA)
    instructions = wrap_text(cv2,SUBMIT_INSTRUCTION,max(30,width-30),scale*.7,2)
    for row, instruction in enumerate(instructions):
        cv2.putText(image,instruction,(12,y+(row+2)*spacing),cv2.FONT_HERSHEY_SIMPLEX,scale*.7,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,status+' | q: quit',(12,y+spacing),cv2.FONT_HERSHEY_SIMPLEX,scale*.7,(0,210,255),1,cv2.LINE_AA)


def main():
    load_dotenv(ROOT.parent / '.env', override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', action='store_true', help='Open browser webcam with live microphone subtitles')
    parser.add_argument('--ui-url', default='http://localhost:8000/')
    parser.add_argument('--url', default='ws://localhost:8000/ws/sign-language')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--response-timeout', type=float, default=30)
    parser.add_argument('--origin', default=None, help='Origin header if the server has an allowlist')
    args = parser.parse_args()
    if (not math.isfinite(args.fps) or not 0 < args.fps <= 60 or
            not math.isfinite(args.response_timeout) or args.response_timeout <= 0 or
            not 1 <= args.jpeg_quality <= 100 or not 32 <= args.max_side <= 1920):
        parser.error('Invalid cadence, response timeout or JPEG settings')
    if urlparse(args.url).scheme not in ('ws','wss') or not urlparse(args.url).hostname:
        parser.error('--url must be a ws:// or wss:// endpoint')
    if args.browser:
        if urlparse(args.ui_url).scheme not in ('http', 'https') or not urlparse(args.ui_url).hostname:
            parser.error('--ui-url must be an http:// or https:// page')
        import webbrowser
        if not webbrowser.open(args.ui_url):
            print(f'Open {args.ui_url} in Chrome for webcam and microphone subtitles.')
        return
    import cv2
    logging.basicConfig(level=logging.INFO)
    shared = SharedState()
    worker = NetworkWorker(shared, url=args.url, fps=args.fps,
                           response_timeout=args.response_timeout,
                           token=os.getenv('SIGN_WS_TOKEN',''), origin=args.origin)
    camera = None
    started_worker = False
    try:
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError(f'Cannot open camera {args.camera}; check camera permissions')
        camera.set(cv2.CAP_PROP_FPS,args.fps)
        worker.start()
        started_worker = True
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        deadline = 0.
        while True:
            ok, image = camera.read()
            if not ok or image is None:
                LOG.error('Camera read failed')
                break
            now = time.monotonic()
            if now >= deadline:
                try:
                    # Encode before drawing overlays; transmit unmirrored raw camera pixels.
                    shared.publish(encode_frame(cv2,image,args.jpeg_quality,args.max_side))
                except ValueError:
                    shared.status('JPEG encoding failed')
                deadline = now + 1 / args.fps
            draw_overlay(cv2,image,shared.snapshot())
            cv2.imshow(WINDOW,image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if started_worker:
                worker.close()
        finally:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
