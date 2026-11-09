import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import time

import cv2
import numpy as np
from websockets.asyncio.server import serve

spec=importlib.util.spec_from_file_location('webcam',Path(__file__).resolve().parents[1]/'scripts/inference_webcam.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


class ClientTests(unittest.TestCase):
    def test_overlay_ingestion_and_sentence_boundary(self):
        state=m.OverlayState();state.connected()
        for word in ('hello','water','water','please'):
            state.apply({'status':'prediction','prediction':word})
        self.assertEqual(state.glosses,['hello','water','please'])
        state.apply({'status':'prediction','prediction':'hello'})
        self.assertEqual(state.glosses,['hello','water','please','hello'])
        for _ in range(100):state.apply({'status':'idle','prediction':None})
        self.assertEqual(state.status,'Idle')
        state.apply({'status':'submitting_sentence','state':'SUBMITTING_SENTENCE','glosses':['hello','water','please','hello']})
        self.assertEqual(state.status,'Processing sentence...')
        state.apply({'status':'sentence_complete','sentence':'Hello, water please.','glosses':['hello','water','please']})
        self.assertEqual(state.sentence,'Hello, water please.');self.assertFalse(state.pending_words)
        self.assertEqual(state.status,'Idle');self.assertFalse(state.glosses)
        state.apply({'status':'prediction','prediction':'hello'})
        self.assertEqual(state.glosses,['hello'])
        state.connected();self.assertEqual(state.sentence,'Hello, water please.');self.assertFalse(state.glosses)
        with self.assertRaises(ValueError):state.apply([])

    def test_jpeg_resize_and_latest_frame(self):
        encoded=m.encode_frame(cv2,np.zeros((720,1280,3),dtype=np.uint8))
        image=cv2.imdecode(np.frombuffer(base64.b64decode(encoded),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(image.shape,(360,640,3))
        shared=m.SharedState();shared.publish('old');shared.publish(encoded)
        serial,stamp,value=shared.frame_after(0)
        self.assertEqual(value,encoded);self.assertIsNone(shared.frame_after(serial))
        with shared.lock:shared.latest=(serial,time.monotonic()-1,value)
        self.assertIsNone(shared.frame_after(0))

    def test_full_sentence_wrap_and_render_without_gui(self):
        text='A long translated sentence with many words. '*30
        lines=m.wrap_text(cv2,text,300,.7,max_lines=None)
        self.assertEqual(''.join(lines),' '.join(text.split()))
        image=np.zeros((480,640,3),dtype=np.uint8)
        m.draw_overlay(cv2,image,(text,['hello','water'],'Idle'))
        self.assertTrue(image.any())

    def test_browser_launcher_does_not_open_native_camera(self):
        with patch('webbrowser.open',return_value=True) as launch,patch.object(sys,'argv',['inference_webcam.py','--browser']),patch('cv2.VideoCapture') as camera:
            m.main()
            launch.assert_called_once_with('http://localhost:8000/')
            camera.assert_not_called()

    def test_cli_ui_responsive_without_server_and_cleanup(self):
        camera=MagicMock();camera.isOpened.return_value=True
        camera.read.return_value=(True,np.zeros((480,640,3),dtype=np.uint8))
        fake=MagicMock(wraps=cv2);fake.VideoCapture.return_value=camera
        for name in ('IMWRITE_JPEG_QUALITY','CAP_PROP_FPS','WINDOW_NORMAL','WND_PROP_VISIBLE','FONT_HERSHEY_SIMPLEX','LINE_AA'):
            setattr(fake,name,getattr(cv2,name))
        fake.namedWindow=MagicMock();fake.imshow=MagicMock();fake.destroyAllWindows=MagicMock()
        fake.waitKey.side_effect=[-1,-1,ord('q')];fake.getWindowProperty.return_value=1
        with patch.dict(sys.modules,{'cv2':fake}),patch.object(sys,'argv',['inference_webcam.py','--url','ws://127.0.0.1:1/ws/sign-language']):
            m.main()
        camera.release.assert_called_once();fake.destroyAllWindows.assert_called_once()
        self.assertEqual(fake.imshow.call_count,3)
        self.assertFalse(any(t.name=='webcam-websocket' and t.is_alive() for t in m.threading.enumerate()))


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_transport_reconnect_and_sentence(self):
        shared=m.SharedState();connected=0;complete=asyncio.Event()
        encoded=m.encode_frame(cv2,np.zeros((32,32,3),dtype=np.uint8))
        async def handler(socket):
            nonlocal connected
            connected+=1
            self.assertEqual(socket.request.path,'/ws/sign-language')
            payload=json.loads(await socket.recv());self.assertEqual(payload,{'frame':encoded})
            if connected==1:
                await socket.close();return
            await socket.send(json.dumps({'type':'status','state':'SUBMITTING','glosses':['hello']}))
            await socket.send(json.dumps({'status':'sentence_complete','sentence':'Hello!','glosses':['hello']}))
            complete.set()
            await socket.wait_closed()
        async with serve(handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            async def producer():
                while True:shared.publish(encoded);await asyncio.sleep(.02)
            produce=asyncio.create_task(producer())
            network=asyncio.create_task(m.connection_loop(shared,f'ws://127.0.0.1:{port}/ws/sign-language'))
            try:
                await asyncio.wait_for(complete.wait(),3)
                for _ in range(50):
                    if shared.snapshot()[0]=='Hello!':break
                    await asyncio.sleep(.01)
                self.assertEqual(shared.snapshot()[:2],('Hello!',[]))
                self.assertGreaterEqual(connected,2)
            finally:
                network.cancel();produce.cancel()
                await asyncio.gather(network,produce,return_exceptions=True)

    async def test_target_cadence_without_bursts(self):
        from unittest.mock import AsyncMock
        shared=m.SharedState();shared.publish('frame')
        times=[]
        async def send(payload):times.append(time.monotonic())
        async def receive():
            shared.publish('next-frame')
            if len(times)==3:raise asyncio.CancelledError()
            return '{"status":"detecting","prediction":null}'
        socket=MagicMock();socket.send=send;socket.recv=receive
        with self.assertRaises(asyncio.CancelledError):
            await m.stream_connection(socket,shared,24,1)
        self.assertEqual(len(times),3)
        self.assertTrue(all(b-a>=.035 for a,b in zip(times,times[1:])))

    async def test_unavailable_timeout_and_latest_only(self):
        shared=m.SharedState();shared.publish('first');shared.publish('newest')
        socket=MagicMock()
        from unittest.mock import AsyncMock
        socket.send=AsyncMock()
        async def stalled():await asyncio.sleep(1)
        socket.recv=stalled
        with self.assertRaises(asyncio.TimeoutError):
            await m.stream_connection(socket,shared,24,.02)
        self.assertEqual(json.loads(socket.send.call_args.args[0]),{'frame':'newest'})
        network=asyncio.create_task(m.connection_loop(shared,'ws://127.0.0.1:1/ws/sign-language'))
        await asyncio.sleep(.1)
        self.assertIn('reconnecting',shared.snapshot()[2])
        network.cancel();await asyncio.gather(network,return_exceptions=True)


if __name__=='__main__':unittest.main()
