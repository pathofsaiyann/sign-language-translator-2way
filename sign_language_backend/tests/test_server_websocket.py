import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import tempfile
from types import SimpleNamespace as S
import unittest
import asyncio
import os
import httpx
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import server_websocket as server


def frame():
    stream = BytesIO()
    Image.new('RGB', (32,32)).save(stream, format='JPEG')
    return {'frame':base64.b64encode(stream.getvalue()).decode()}


def core(payload):
    return {key:value for key,value in payload.items() if key not in ('glosses','type','gesture_state','current_sign')}


def pose_result(raised=True):
    points=[S(x=.3,y=.6,z=0.,visibility=.99) for _ in range(33)]
    points[15].y=.2 if raised else .8
    points[16].y=.2 if raised else .8
    return S(pose_landmarks=S(landmark=points))


class Detector:
    instances = []
    def __init__(self):
        self.thread = threading.get_ident()
        self.closed = False
        self.instances.append(self)
    def process(self, image):
        assert threading.get_ident() == self.thread
        return S(pose_landmarks=S(landmark=[S(x=.1,y=.2,z=-.3)]*33))
    def close(self):
        assert threading.get_ident() == self.thread
        self.closed = True


class MovingDetector(Detector):
    def __init__(self):
        super().__init__()
        self.frame = 0
    def process(self, image):
        result = super().process(image)
        self.frame += 1
        result.left_hand_landmarks = S(landmark=[S(x=.1 if self.frame % 2 else .9,y=.2,z=-.3)]*21)
        return result


class FakeRuntime:
    def __init__(self, settings):
        self.labels = [f'sign{i}' for i in range(19)]
    def predict(self, buffer):
        assert buffer.shape == (1,64,180) and buffer.dtype == np.float32
        return 2,.9


class ServerTests(unittest.TestCase):
    def setUp(self):
        # Tests must never reload real root credentials over mocked environment keys.
        self.debug_environment = patch.dict(os.environ, {'LLM_DEBUG':'1'})
        self.debug_environment.start(); self.addCleanup(self.debug_environment.stop)
        self.environment_reload = patch.object(server, 'load_environment')
        self.reload = self.environment_reload.start()
        self.addCleanup(self.environment_reload.stop)

    def test_features_and_decoder(self):
        features = server.extract_features(Detector().process(None))
        np.testing.assert_allclose(features[:3], [.1,.2,-.3])
        self.assertFalse(features[54:].any())
        self.assertEqual(server.decode_frame(frame()).shape,(32,32,3))
        for payload in ({}, [], {'frame':'???'}, {'frame':base64.b64encode(b'bad').decode()}):
            with self.assertRaises(ValueError): server.decode_frame(payload)
        stream = BytesIO(); Image.new('RGB',(8,8)).save(stream,format='PNG')
        with self.assertRaises(ValueError): server.decode_frame({'frame':base64.b64encode(stream.getvalue()).decode()})

    def test_debounce_threshold_switch_reset(self):
        d = server.Debouncer()
        for _ in range(4): self.assertFalse(d.update(0,.71))
        self.assertTrue(d.update(0,.71))
        self.assertFalse(d.update(0,.70))
        self.assertFalse(d.update(1,.99))
        self.assertFalse(d.update(0,.99))
        self.assertFalse(d.update(0,float('nan')))

    def test_websocket_isolation_errors_and_cleanup(self):
        Detector.instances = []
        app = server.create_app(runtime_factory=FakeRuntime,detector_factory=MovingDetector)
        with TestClient(app) as client:
            with client.websocket_connect('/ws/sign-language') as first, client.websocket_connect('/ws/sign-language') as second:
                for _ in range(67):
                    first.send_json(frame()); self.assertIn(first.receive_json()['status'], ('detecting','idle'))
                first.send_json(frame()); self.assertEqual(first.receive_json()['prediction'],'sign2')
                second.send_json(frame()); self.assertEqual(core(second.receive_json()),server.IDLE)
                first.send_text('{'); self.assertEqual(first.receive_json()['status'],'error')
                first.send_json(frame()); self.assertEqual(core(first.receive_json()),server.IDLE)
                first.send_bytes(b'bad'); self.assertEqual(first.receive_json()['status'],'error')
                first.send_json({'frame':'invalid'}); self.assertEqual(first.receive_json()['status'],'error')
        self.assertEqual(app.state.clients,0)
        self.assertTrue(all(d.closed for d in Detector.instances))

    def test_access_capacity_and_message_limit(self):
        from starlette.websockets import WebSocketDisconnect
        settings = server.Settings(max_clients=1,token='test',origins=('https://example.test',))
        app = server.create_app(settings,FakeRuntime,Detector)
        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect('/ws/sign-language'): pass
            headers = {'authorization':'Bearer test','origin':'https://example.test'}
            with client.websocket_connect('/ws/sign-language',headers=headers) as websocket:
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect('/ws/sign-language',headers=headers): pass
                websocket.send_text('x'*(server.MAX_MESSAGE_BYTES+1))
                with self.assertRaises(WebSocketDisconnect) as caught: websocket.receive_json()
                self.assertEqual(caught.exception.code,1009)
        self.assertEqual(app.state.clients,0)

    def test_missing_landmarks_and_gap_reset(self):
        import time
        session = server.ClientSession(FakeRuntime(None),Detector,server.Settings())
        session.frames = 64
        session.last_frame = time.monotonic()-4
        self.assertEqual(core(session.process(frame())),server.IDLE)
        self.assertEqual(session.frames,1)
        session.holistic.process = lambda image: S()
        self.assertEqual(core(session.process(frame())),server.IDLE)
        self.assertEqual(session.frames,2)
        session.close()

    def test_motion_gate_no_idle_submit_and_consecutive_duplicates(self):
        runtime=FakeRuntime(None);runtime.predict=MagicMock(return_value=(2,.9))
        session=server.ClientSession(runtime,Detector,server.Settings())
        for _ in range(80):self.assertEqual(core(session.process(frame())),server.IDLE)
        runtime.predict.assert_not_called();session.close()
        session=server.ClientSession(runtime,MovingDetector,server.Settings())
        for _ in range(75):session.process(frame())
        self.assertEqual(session.gloss_sequence,['sign2'])
        runtime.predict.return_value=(3,.9)
        for _ in range(5):session.process(frame())
        runtime.predict.return_value=(2,.9)
        for _ in range(10):session.process(frame())
        self.assertEqual(session.gloss_sequence,['sign2','sign3','sign2'])
        session.holistic.process=lambda image:S()
        for _ in range(150):self.assertEqual(core(session.process(frame())),server.IDLE)
        self.assertEqual(session.gloss_sequence,['sign2','sign3','sign2'])
        session.close()

    def test_gesture_continuity_latch_and_no_classifier_pollution(self):
        runtime=FakeRuntime(None);runtime.predict=MagicMock(return_value=(2,.9))
        session=server.ClientSession(runtime,Detector,server.Settings())
        session.gloss_sequence=['hello']
        session.holistic.process=lambda image:pose_result()
        for _ in range(14):self.assertNotEqual(session.process(frame())['status'],'sentence_pending')
        session.holistic.process=lambda image:S()
        session.process(frame())
        session.holistic.process=lambda image:pose_result()
        for _ in range(14):self.assertNotEqual(session.process(frame())['status'],'sentence_pending')
        result=session.process(frame())
        self.assertEqual(result,{'status':'sentence_pending','state':'SUBMITTING_SENTENCE','glosses':['hello']})
        runtime.predict.assert_not_called()
        session.complete_sentence()
        session.gloss_sequence=['next']
        for _ in range(30):self.assertNotEqual(session.process(frame())['status'],'sentence_pending')
        session.holistic.process=lambda image:pose_result(False)
        session.process(frame())
        session.holistic.process=lambda image:pose_result()
        for _ in range(14):session.process(frame())
        self.assertEqual(session.process(frame())['status'],'sentence_pending')
        session.close()

    def test_gesture_invalid_visibility_one_hand_and_empty_glosses(self):
        self.assertTrue(server.raised_hands(pose_result()))
        self.assertFalse(server.raised_hands(pose_result(False)))
        one=pose_result();one.pose_landmarks.landmark[16].y=.8
        self.assertFalse(server.raised_hands(one))
        invalid=pose_result();invalid.pose_landmarks.landmark[15].visibility=.1
        self.assertIsNone(server.raised_hands(invalid))
        invalid.pose_landmarks.landmark[15].y=float('nan')
        self.assertIsNone(server.raised_hands(invalid))
        session=server.ClientSession(FakeRuntime(None),Detector,server.Settings())
        session.holistic.process=lambda image:pose_result()
        for _ in range(40):self.assertNotEqual(session.process(frame())['status'],'sentence_pending')
        session.close()

    def test_detected_static_hands_roll_into_idle(self):
        runtime=FakeRuntime(None)
        runtime.predict=MagicMock(return_value=(2,.9))
        session=server.ClientSession(runtime,MovingDetector,server.Settings())
        for _ in range(68):session.process(frame())
        static=S(left_hand_landmarks=S(landmark=[S(x=.4,y=.2,z=-.3)]*21))
        session.holistic.process=lambda image:static
        session.gloss_sequence.clear()  # isolate gate behavior from completion
        for _ in range(64):session.process(frame())
        session.gloss_sequence.clear()  # only measure the gate, not sentence completion
        before=runtime.predict.call_count
        for _ in range(10):self.assertEqual(core(session.process(frame())),server.IDLE)
        self.assertEqual(runtime.predict.call_count,before)
        session.close()

    def test_ordered_groq_models_then_gemini_fallback(self):
        groq=MagicMock();groq.__aenter__=AsyncMock(return_value=groq)
        groq.__aexit__=AsyncMock(return_value=False)
        answer=S(choices=[S(message=S(content='Groq sentence.'))])
        gemini=MagicMock();gemini.__enter__.return_value=gemini
        gemini.models.generate_content.return_value=S(text='Gemini sentence.')
        models=['llama3-8b-8192','llama-3.1-8b-instant']
        self.assertEqual(server.GROQ_MODELS,models)
        with patch.dict(os.environ,{'GROQ_API_KEY':'groq-test','GEMINI_API_KEY':'gemini-test'}),patch('groq.AsyncGroq',return_value=groq) as provider,patch('google.genai.Client',return_value=gemini) as secondary,patch('builtins.print') as output:
            for failed in range(2):
                response=httpx.Response(404,text='{"error":"model not found"}')
                error=httpx.HTTPStatusError('not found',request=httpx.Request('POST','https://example.test'),response=response)
                groq.chat.completions.create=AsyncMock(side_effect=[error]*failed+[answer])
                self.assertEqual(asyncio.run(server.build_sentence(['water','please'])),'Groq sentence.')
                attempts=groq.chat.completions.create.call_args_list
                self.assertEqual([call.kwargs['model'] for call in attempts],models[:failed+1])
                self.assertEqual(attempts[-1].kwargs['messages'][0]['content'],server.GLOSS_PROMPT)
                self.assertEqual(json.loads(attempts[-1].kwargs['messages'][1]['content']),['water','please'])
                secondary.assert_not_called()
                output.assert_any_call(f'[LLM SUCCESS] Model: {models[failed]} | Output: Groq sentence.',flush=True)
            provider.assert_called_with(api_key='groq-test',timeout=3.0,max_retries=0)
            groq.chat.completions.create=AsyncMock(side_effect=[RuntimeError('groq-test invalid'),TimeoutError('deadline')])
            self.assertEqual(asyncio.run(server.build_sentence(['hello'])),'Gemini sentence.')
            self.assertEqual(groq.chat.completions.create.await_count,2)
            output.assert_any_call('[LLM SUCCESS] Model: gemini-1.5-flash | Output: Gemini sentence.',flush=True)
            for text in (None,'','  ', '""'):
                groq.chat.completions.create=AsyncMock(return_value=S(choices=[S(message=S(content=text))]))
                gemini.models.generate_content.side_effect=RuntimeError('gemini-test rejected')
                self.assertEqual(asyncio.run(server.build_sentence(['water','please'])),'water please')
                self.assertEqual(groq.chat.completions.create.await_count,2)
            diagnostic=' '.join(str(c.args[0]) for c in output.call_args_list)
            self.assertNotIn('groq-test',diagnostic);self.assertNotIn('gemini-test',diagnostic)
            output.assert_any_call('[LLM FALLBACK] Returning raw gloss sequence.',flush=True)

    def test_groq_deadlines_advance_to_gemini(self):
        groq=MagicMock();groq.__aenter__=AsyncMock(return_value=groq)
        groq.__aexit__=AsyncMock(return_value=False)
        async def slow(**kwargs):await asyncio.sleep(1)
        groq.chat.completions.create=AsyncMock(side_effect=slow)
        gemini=MagicMock();gemini.__enter__.return_value=gemini
        gemini.models.generate_content.return_value=S(text='Gemini sentence.')
        with patch.dict(os.environ,{'GROQ_API_KEY':'groq-test','GEMINI_API_KEY':'gemini-test'}),patch('groq.AsyncGroq',return_value=groq),patch('google.genai.Client',return_value=gemini),patch.object(server,'LLM_TIMEOUT_SECONDS',.02):
            self.assertEqual(asyncio.run(server.build_sentence(['hello'])),'Gemini sentence.')
            self.assertEqual(groq.chat.completions.create.await_count,2)
            self.assertEqual(groq.__aexit__.await_count,2)

    def test_translation_cancellation_does_not_try_other_models(self):
        groq=MagicMock();groq.__aenter__=AsyncMock(return_value=groq)
        groq.__aexit__=AsyncMock(return_value=False)
        groq.chat.completions.create=AsyncMock(side_effect=asyncio.CancelledError())
        with patch.dict(os.environ,{'GROQ_API_KEY':'groq-test','GEMINI_API_KEY':'gemini-test'}),patch('groq.AsyncGroq',return_value=groq),patch('google.genai.Client') as gemini:
            with self.assertRaises(asyncio.CancelledError):asyncio.run(server.build_sentence(['hello']))
            self.assertEqual(groq.chat.completions.create.await_count,1)
            gemini.assert_not_called()

    def test_translation_refreshes_root_environment_and_empty_input(self):
        self.assertEqual(asyncio.run(server.build_sentence([])),'')
        self.reload.assert_called_once_with()

    def test_provider_error_logs_exact_response_with_secrets_redacted(self):
        response=httpx.Response(400,text='{"error":{"message":"invalid model", "details":"gemini-secret groq-secret"}}')
        error=httpx.HTTPStatusError('API failed',request=httpx.Request('POST','https://example.test'),response=response)
        with patch('builtins.print') as output:
            server.log_llm_error('GEMINI',error,('gemini-secret','groq-secret'))
        message=output.call_args.args[0]
        self.assertIn('status=400',message)
        self.assertIn('response={"error":{"message":"invalid model", "details":"[REDACTED] [REDACTED]"}}',message)
        self.assertIn('HTTPStatusError: API failed',message)
        self.assertNotIn('gemini-secret',message)
        with patch.dict(os.environ,{'LLM_DEBUG':'0'}),patch('builtins.print') as output:
            server.log_llm_error('GEMINI',error,())
        self.assertIn('status=400',output.call_args.args[0])
        self.assertNotIn('response=',output.call_args.args[0])

    def test_gemini_missing_key_uses_groq(self):
        groq=MagicMock();groq.__aenter__=AsyncMock(return_value=groq)
        groq.__aexit__=AsyncMock(return_value=False)
        groq.chat.completions.create=AsyncMock(return_value=S(choices=[S(message=S(content='Hello.'))]))
        with patch.dict(os.environ,{'GROQ_API_KEY':'groq-test','GEMINI_API_KEY':''}),patch('groq.AsyncGroq',return_value=groq),patch('google.genai.Client') as gemini:
            self.assertEqual(asyncio.run(server.build_sentence(['hello'])),'Hello.')
            gemini.assert_not_called()

    def test_sentence_timeout_fallback(self):
        async def expired(awaitable,timeout):
            awaitable.close()
            raise asyncio.TimeoutError()
        with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':'test-only'}),patch.object(server.asyncio,'wait_for',side_effect=expired):
            self.assertEqual(asyncio.run(server.build_sentence(['hello','please'])),'hello please')

    def test_websocket_sentence_and_client_isolation(self):
        class SignThenRaise(MovingDetector):
            def process(self,image):
                result=super().process(image)
                return result if self.frame<=72 else pose_result()
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=SignThenRaise)
        with patch.object(server,'build_sentence',new_callable=AsyncMock,return_value='A sentence.') as build:
            with TestClient(app) as client:
                with client.websocket_connect('/ws/sign-language') as first, client.websocket_connect('/ws/sign-language') as second:
                    for _ in range(72+14):first.send_json(frame());first.receive_json()
                    first.send_json(frame())
                    pending=first.receive_json()
                    self.assertEqual(pending['state'],'SUBMITTING')
                    self.assertEqual(pending['glosses'],['sign2'])
                    self.assertEqual(first.receive_json(),{'type':'sentence_complete','status':'sentence_complete','glosses':['sign2'],'sentence':'A sentence.'})
                    build.assert_awaited_once_with(['sign2'])
                    for _ in range(40):
                        first.send_json(frame());self.assertNotEqual(first.receive_json()['status'],'sentence_complete')
                        second.send_json(frame());self.assertNotEqual(second.receive_json()['status'],'sentence_complete')
                    self.assertEqual(build.await_count,1)
        self.assertEqual(app.state.clients,0)

    def test_sentence_fallback_and_mocked_gemini(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':''}):
            self.assertEqual(asyncio.run(server.build_sentence(['hello','water'])),'hello water')
        client=MagicMock()
        client.__enter__.return_value=client
        client.models.generate_content.return_value=S(text=' "Hello, water please." ')
        with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':'test-only','GEMINI_MODEL':'gemini-2.5-flash'}),patch('google.genai.Client',return_value=client) as gemini:
            self.assertEqual(asyncio.run(server.build_sentence(['hello','water','please'])),'Hello, water please.')
            kwargs=client.models.generate_content.call_args.kwargs
            self.assertEqual(kwargs['model'],'gemini-1.5-flash')
            self.assertEqual(kwargs['config'].system_instruction,server.GLOSS_PROMPT)
            self.assertEqual(json.loads(kwargs['contents']),['hello','water','please'])
            options=gemini.call_args.kwargs
            self.assertEqual(options['api_key'],'test-only')
            self.assertFalse(options['vertexai'])
            self.assertEqual(options['http_options'].timeout,12000)
            self.assertEqual(options['http_options'].retry_options.attempts,1)
            for text in (None,'','  '):
                client.models.generate_content.return_value=S(text=text)
                self.assertEqual(asyncio.run(server.build_sentence(['hello','water'])),'hello water')
            client.models.generate_content.side_effect=RuntimeError('provider rejected test-only: quota exhausted')
            with patch('builtins.print') as output:
                self.assertEqual(asyncio.run(server.build_sentence(['hello','water'])),'hello water')
                output.assert_any_call("[LLM] Translating glosses: ['hello', 'water']",flush=True)
                diagnostic=' '.join(str(call.args[0]) for call in output.call_args_list)
                self.assertIn('[GEMINI gemini-1.5-flash ERROR]',diagnostic)
                self.assertIn('quota exhausted',diagnostic)
                self.assertIn('[REDACTED]',diagnostic)
                self.assertNotIn('test-only',diagnostic)
            client.models.generate_content.side_effect=None
            client.models.generate_content.return_value=S(text='Please give the man some water.')
            with patch.dict(os.environ,{'GEMINI_MODEL':'gemini-3.8-flash'}),patch('builtins.print') as output:
                self.assertEqual(asyncio.run(server.build_sentence(['man','water','please'])),'Please give the man some water.')
                options=client.models.generate_content.call_args.kwargs
                self.assertEqual(options['model'],'gemini-1.5-flash')
                self.assertIsNone(options['config'].thinking_config)
                output.assert_any_call('[LLM SUCCESS] Model: gemini-1.5-flash | Output: Please give the man some water.',flush=True)

    def test_deadline_returns_promptly_without_blocking_event_loop(self):
        import time
        release=threading.Event()
        client=MagicMock();client.__enter__.return_value=client
        def slow_request(**kwargs):
            release.wait(.5)
            return S(text='Hello, please give me water.')
        client.models.generate_content.side_effect=slow_request
        async def run():
            start=time.monotonic()
            task=asyncio.create_task(server.build_sentence(['hello','water']))
            await asyncio.sleep(.01)  # Must run while the SDK worker is blocked.
            self.assertFalse(task.done())
            result=await task
            self.assertLess(time.monotonic()-start,.2)
            self.assertEqual(result,'hello water')
        with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':'test-only'}),patch('google.genai.Client',return_value=client),patch.object(server,'GEMINI_TIMEOUT_SECONDS',.04),patch('builtins.print') as output:
            try:asyncio.run(run())
            finally:release.set()
            output.assert_any_call('[LLM FALLBACK] Returning raw gloss sequence.',flush=True)
        self.assertEqual(server.LLM_TIMEOUT_SECONDS,4.)
        self.assertEqual(server.GEMINI_TIMEOUT_SECONDS,12.)

    def test_provider_504_preserves_websocket_connection(self):
        class DeadlineError(Exception):
            code=504
            status='DEADLINE_EXCEEDED'
        class SignThenRaise(MovingDetector):
            def process(self,image):
                result=super().process(image)
                return result if self.frame<=72 else pose_result()
        sdk=MagicMock();sdk.__enter__.return_value=sdk
        sdk.models.generate_content.side_effect=DeadlineError('deadline')
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=SignThenRaise)
        with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':'test-only'}),patch('google.genai.Client',return_value=sdk),patch('builtins.print') as output:
            with TestClient(app) as client:
                with client.websocket_connect('/ws/sign-language') as socket:
                    for _ in range(86):socket.send_json(frame());socket.receive_json()
                    socket.send_json(frame())
                    self.assertEqual(socket.receive_json()['status'],'submitting_sentence')
                    self.assertEqual(socket.receive_json(),{'type':'sentence_complete','status':'sentence_complete','sentence':'sign2','glosses':['sign2']})
                    socket.send_json(frame())
                    self.assertEqual(socket.receive_json()['status'],'detecting')
            output.assert_any_call('[LLM FALLBACK] Returning raw gloss sequence.',flush=True)
        self.assertEqual(app.state.clients,0)

    def test_two_way_broadcast_validation_and_room_isolation(self):
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=Detector)
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertIn('Send to Chat',client.get('/index.html').text)
            self.assertIn('id="chat-stream"',client.get('/index.html').text)
            self.assertEqual(client.get('/app.js').status_code,200)
            with client.websocket_connect('/ws/two-way?room=pair') as sam,client.websocket_connect('/ws/two-way?room=pair') as steve,client.websocket_connect('/ws/two-way?room=other') as outsider:
                for sock in (sam,steve,outsider):self.assertEqual(sock.receive_json()['type'],'ready')
                sam.send_json({'type':'send','sender':'sam','text':' Hello Steve. ','request_id':'draft-1'})
                first=sam.receive_json();self.assertEqual(first,steve.receive_json())
                self.assertEqual(first['recipient'],'steve');self.assertEqual(first['text'],'Hello Steve.')
                self.assertEqual(sam.receive_json()['type'],'sent')
                outsider.send_json({'type':'ping'});self.assertEqual(outsider.receive_json(),{'type':'pong'})
                steve.send_json({'type':'send','sender':'steve','text':'Hello Sam.'})
                self.assertEqual(steve.receive_json()['recipient'],'sam')
                self.assertEqual(sam.receive_json()['text'],'Hello Sam.')
                for payload in ({'type':'send','sender':'other','text':'x'},{'type':'send','sender':'sam','text':' '},{'type':'send','sender':'steve','text':'x'*2001},[],{'type':'unknown'}):
                    sam.send_json(payload);self.assertEqual(sam.receive_json()['type'],'error')
                steve.send_json({'type':'ping'});self.assertEqual(steve.receive_json(),{'type':'pong'})
        self.assertEqual(app.state.clients,0)

    def test_two_way_private_stage_and_dispatch_during_translation(self):
        release=threading.Event()
        async def build(glosses):
            while not release.is_set():await asyncio.sleep(.005)
            return 'A staged sentence.'
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=MovingDetector)
        with patch.object(server,'build_sentence',side_effect=build):
            with TestClient(app) as client:
                with client.websocket_connect('/ws/two-way') as sam,client.websocket_connect('/ws/two-way') as steve:
                    sam.receive_json();steve.receive_json()
                    for _ in range(72):sam.send_json(frame());sam.receive_json()
                    sam.send_json({'type':'stage'})
                    self.assertEqual(sam.receive_json()['type'],'processing')
                    sam.send_json({'type':'send','sender':'steve','text':'I can speak during translation.'})
                    self.assertEqual(sam.receive_json()['type'],'message')
                    self.assertEqual(steve.receive_json()['text'],'I can speak during translation.')
                    release.set()
                    draft=sam.receive_json();self.assertEqual(draft['type'],'draft')
                    self.assertFalse(draft['auto_send'])
                    steve.send_json({'type':'ping'});self.assertEqual(steve.receive_json(),{'type':'pong'})
                    sam.send_json({'type':'send','sender':'sam','text':'Edited sentence.'})
                    self.assertEqual(sam.receive_json()['text'],'Edited sentence.')
                    self.assertEqual(steve.receive_json()['text'],'Edited sentence.')
                    self.assertEqual(sam.receive_json()['glosses'],[])
        self.assertEqual(app.state.clients,0)

    def test_two_way_hands_up_stages_and_sends_once(self):
        class SignThenRaise(MovingDetector):
            def process(self,image):
                result=super().process(image)
                return result if self.frame<=72 else pose_result()
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=SignThenRaise)
        with patch.object(server,'build_sentence',new_callable=AsyncMock,return_value='Hello, Steve.') as build:
            with TestClient(app) as client:
                with client.websocket_connect('/ws/two-way') as sock:
                    sock.receive_json()
                    for _ in range(72):sock.send_json(frame());sock.receive_json()
                    sock.send_json({'type':'stage'})
                    self.assertEqual(sock.receive_json()['type'],'processing')
                    self.assertFalse(sock.receive_json()['auto_send'])
                    for _ in range(15):sock.send_json(frame());sock.receive_json()
                    self.assertEqual(sock.receive_json()['type'],'processing')
                    self.assertTrue(sock.receive_json()['auto_send'])
                    self.assertEqual(sock.receive_json()['text'],'Hello, Steve.')
                    self.assertEqual(sock.receive_json()['type'],'sent')
                    for _ in range(20):
                        sock.send_json(frame());self.assertEqual(sock.receive_json()['type'],'frame_result')
                    build.assert_awaited_once()
        self.assertEqual(app.state.clients,0)

    def test_frontend_assets_cors_and_no_secret_exposure(self):
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=Detector)
        with TestClient(app) as client:
            page=client.get('/webcam')
            self.assertEqual(page.status_code,200)
            self.assertIn('text/html',page.headers['content-type'])
            self.assertIn('id="chat-stream"',page.text)
            self.assertEqual(page.headers['x-content-type-options'],'nosniff')
            self.assertIn("frame-ancestors 'none'",page.headers['content-security-policy'])
            script=client.get('/app.js');self.assertEqual(script.status_code,200)
            self.assertIn('webkitSpeechRecognition',script.text)
            for path in ('/.env','/../.env','/%2e%2e/.env','/models/model_gislr_19class.pth','/metadata.json','/context.md'):
                self.assertEqual(client.get(path).status_code,404)
            options=client.options('/',headers={'origin':'https://frontend.test','access-control-request-method':'GET'})
            self.assertEqual(options.status_code,200)
            self.assertEqual(options.headers['access-control-allow-origin'],'*')
            self.assertNotIn('access-control-allow-credentials',options.headers)
            from starlette.websockets import WebSocketDisconnect
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect('/ws/sign-language',headers={'origin':'https://untrusted.test'}):pass

    def test_canonical_typed_events(self):
        app=server.create_app(runtime_factory=FakeRuntime,detector_factory=MovingDetector)
        with TestClient(app) as client:
            with client.websocket_connect('/ws/sign-language') as sock:
                sock.send_json(frame());event=sock.receive_json()
                self.assertEqual((event['type'],event['state']),('status','IDLE'))
                for _ in range(67):sock.send_json(frame());event=sock.receive_json()
                self.assertEqual(event['type'],'gloss_update')
                self.assertEqual(event['current_sign'],'sign2')
                self.assertEqual(event['glosses'],['sign2'])
                sock.send_json({'frame':'bad'});self.assertEqual(sock.receive_json()['type'],'error')
        with self.assertRaises(ValueError):
            server.public_event({'status':'prediction','glosses':[],'prediction':42,'confidence':.8})

    def test_bounded_glosses_and_no_translation_text_logs_by_default(self):
        session=server.ClientSession(FakeRuntime(None),MovingDetector,server.Settings())
        session.gloss_sequence=['existing']*100
        for _ in range(68):response=session.process(frame())
        self.assertEqual(response['status'],'error')
        self.assertEqual(len(session.gloss_sequence),100)
        session.close()
        with patch.dict(os.environ,{'LLM_DEBUG':'0'}),patch('builtins.print') as output:
            server.log_llm_error('GROQ',RuntimeError('private sentence'),())
            self.assertNotIn('private sentence',output.call_args.args[0])

    def test_dotenv_path_and_environment_precedence(self):
        self.environment_reload.stop()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'.env').write_text('GEMINI_API_KEY=dotenv-test-only\n')
            with patch.dict(os.environ,{},clear=True):
                server.load_environment(path)
                self.assertEqual(os.getenv('GEMINI_API_KEY'),'dotenv-test-only')
            with patch.dict(os.environ,{'GROQ_API_KEY':'','GEMINI_API_KEY':'environment-test-only'},clear=True):
                server.load_environment(path)
                self.assertEqual(os.getenv('GEMINI_API_KEY'),'dotenv-test-only')

    def test_dotenv_default_is_workspace_root_with_override(self):
        self.environment_reload.stop()
        with patch.object(server,'load_dotenv') as load:
            server.load_environment()
            load.assert_called_once_with(dotenv_path=server.ROOT.parent/'.env',override=True)

    def test_real_assets_and_holistic_smoke(self):
        runtime = server.Runtime(server.Settings())
        path = next((server.ROOT / 'data/processed/val').rglob('*.npy'))
        raw=np.load(path)[None]
        self.assertTrue(runtime.normalize)
        index, probability = runtime.predict(raw)
        import torch
        with torch.inference_mode():
            expected=runtime.model(torch.from_numpy(server.normalize_features(raw))).softmax(1)[0]
        self.assertEqual(index,int(expected.argmax()))
        self.assertAlmostEqual(probability,float(expected.max()),places=6)
        self.assertTrue(0 <= index < 19 and 0 <= probability <= 1)
        app = server.create_app()
        with TestClient(app) as client:
            with client.websocket_connect('/ws/sign-language') as websocket:
                websocket.send_json(frame())
                self.assertEqual(core(websocket.receive_json()),server.IDLE)
        self.assertEqual(app.state.clients,0)


if __name__ == '__main__': unittest.main()
