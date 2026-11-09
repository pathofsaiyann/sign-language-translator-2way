'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let room = location.hash.slice(1);
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(room)) {
    room = crypto.randomUUID(); history.replaceState(null, '', `#${room}`);
  }
  $('room-label').textContent = `Conversation ${room.slice(0, 8)} · Share this page's full URL to join.`;
  let ws = null, stopped = false, retry = 500, retryTimer, frameTimer, previewTimer;
  let framePending = false, busy = false, glosses = [], previewed = '', planned = '', camera = null, cameraStarting = false;
  let samRequest = null, steveRequest = null, pendingSteveText = '';
  const seen = new Set(), video = $('video'), canvas = $('camera-canvas'), ctx = canvas.getContext('2d');
  function controls() {
    const connected = ws && ws.readyState === WebSocket.OPEN;
    $('send-sam').disabled = !connected || busy || !!samRequest || !$('sam-draft').value.trim();
    $('send-steve').disabled = !connected || !!steveRequest || !$('steve-draft').value.trim();
    $('preview').disabled = !connected || busy || !glosses.length;
  }
  function send(data) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    try { ws.send(JSON.stringify(data)); return true; }
    catch (_) { ws.close(); return false; }
  }
  function showGlosses(words) {
    if (!Array.isArray(words)) return;
    glosses = words.filter(x => typeof x === 'string').slice(0, 100);
    $('glosses').textContent = `Glosses: ${JSON.stringify(glosses)}`;
  }
  function doneFrame() { framePending = false; clearTimeout(frameTimer); }
  function stage() {
    if (busy || !glosses.length) return;
    if (send({type: 'stage'})) { busy = true; controls(); }
  }
  function addMessage(data) {
    if (seen.has(data.message_id)) return;
    seen.add(data.message_id); if (seen.size > 500) seen.delete(seen.values().next().value);
    if (!['sam', 'steve'].includes(data.sender) || typeof data.text !== 'string') return;
    const sam = data.sender === 'sam';
    const target = $('chat-stream');
    const item = document.createElement('article');
    item.className = 'chat-message ' + (sam ? 'from-sam' : 'from-steve');
    const tag = document.createElement('span'); tag.className = 'message-tag';
    tag.textContent = sam ? 'Sam (Sign)' : 'Steve (Voice)';
    const bubble = document.createElement('p'); bubble.className = 'bubble'; bubble.textContent = data.text;
    item.appendChild(tag); item.appendChild(bubble);
    target.appendChild(item); if (target.children.length > 100) target.firstElementChild.remove();
    $('chat-empty').hidden = true;
    target.scrollTo({top: target.scrollHeight,
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
    if (sam && $('tts').checked && window.speechSynthesis) {
      // Avoid recognizing TTS output as Steve's microphone input.
      stopMic(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(data.text));
    }
    if (data.sender === 'steve' && data.request_id === steveRequest) {
      if ($('steve-draft').value === pendingSteveText) $('steve-draft').value = '';
      steveRequest = null; pendingSteveText = '';
    }
    if (data.sender === 'sam' && data.request_id === samRequest) {
      samRequest = null; $('sam-draft').value = '';
    }
  }
  function connect() {
    if (stopped) return;
    const current = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/two-way?room=${encodeURIComponent(room)}`);
    ws = current; $('connection').textContent = 'Connecting…'; controls();
    current.onopen = () => { retry = 500; $('connection').textContent = 'Connected'; controls(); };
    current.onmessage = event => {
      if (ws !== current) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'frame_result') {
          doneFrame(); showGlosses(data.glosses);
          $('sign-status').textContent = data.status === 'sentence_pending' || data.status === 'processing' ? 'Translating…' : data.status === 'error' ? data.error : data.status === 'idle' ? 'Idle — ready to sign' : data.prediction ? `Sign: ${data.prediction}` : 'Detecting…';
          if (data.status === 'sentence_pending') busy = true;
          const signature = JSON.stringify(glosses);
          if (glosses.length && signature !== previewed && signature !== planned && !busy) {
            clearTimeout(previewTimer); planned = signature; previewTimer = setTimeout(() => { planned = ''; previewed = signature; stage(); }, 900);
          }
        } else if (data.type === 'processing') {
          busy = true; clearTimeout(previewTimer); $('sign-status').textContent = 'Translating your signs…';
        } else if (data.type === 'draft') {
          busy = false; $('sam-draft').value = data.sentence; showGlosses(data.glosses);
          previewed = JSON.stringify(glosses); $('sign-status').textContent = data.auto_send ? 'Sending to chat…' : 'Draft ready — review, then send to chat';
        } else if (data.type === 'sent') {
          $('sam-draft').value = ''; showGlosses([]); previewed = ''; planned = ''; busy = false; $('sign-status').textContent = 'Sent to chat';
        } else if (data.type === 'message') addMessage(data);
        else if (data.type === 'error' || data.status === 'error') {
          doneFrame(); busy = false; samRequest = null; steveRequest = null;
          $('sign-status').textContent = data.error || 'Request failed. Your draft is preserved.';
        }
        controls();
      } catch (_) { $('connection').textContent = 'Invalid server response'; current.close(); }
    };
    current.onerror = () => { $('connection').textContent = 'Connection unavailable'; };
    current.onclose = event => {
      if (ws !== current) return;
      ws = null; doneFrame(); busy = false; clearTimeout(previewTimer); showGlosses([]); previewed = ''; planned = '';
      samRequest = null; steveRequest = null; controls();
      if (stopped) return;
      if (event.code === 1008) { $('connection').textContent = 'Connection denied — check server origin/auth settings'; return; }
      $('connection').textContent = 'Reconnecting… Drafts kept; unconfirmed sends are not retried.';
      retryTimer = setTimeout(connect, retry); retry = Math.min(5000, retry * 2);
    };
  }
  $('preview').onclick = stage;
  $('sam-draft').oninput = controls; $('steve-draft').oninput = controls;
  $('send-sam').onclick = () => {
    if ($('send-sam').disabled) return;
    samRequest = crypto.randomUUID();
    if (!send({type: 'send', sender: 'sam', text: $('sam-draft').value, request_id: samRequest})) samRequest = null;
    controls();
  };
  $('send-steve').onclick = () => {
    if ($('send-steve').disabled) return;
    stopMic(); pendingSteveText = $('steve-draft').value; steveRequest = crypto.randomUUID();
    if (!send({type: 'send', sender: 'steve', text: pendingSteveText, request_id: steveRequest})) steveRequest = null;
    controls();
  };
  function stopCamera() {
    if (camera) camera.getTracks().forEach(track => track.stop());
    camera = null; video.srcObject = null; ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('camera-toggle').textContent = 'Start camera'; $('camera-toggle').setAttribute('aria-pressed', 'false');
    $('camera-placeholder').hidden = false; $('sign-status').textContent = 'Camera is off.';
  }
  $('camera-toggle').onclick = async () => {
    if (camera) { stopCamera(); return; }
    if (cameraStarting) return;
    cameraStarting = true; $('camera-toggle').disabled = true;
    try {
      const captured = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}}, audio: false});
      if (stopped) { captured.getTracks().forEach(track => track.stop()); return; }
      camera = captured; video.srcObject = captured; await video.play();
      $('camera-placeholder').hidden = true; $('camera-toggle').textContent = 'Stop camera';
      $('camera-toggle').setAttribute('aria-pressed', 'true');
    } catch (error) { stopCamera(); $('sign-status').textContent = `Camera unavailable: ${error.name}. Check permissions.`; }
    finally { cameraStarting = false; $('camera-toggle').disabled = false; }
  };
  let lastFrame = 0;
  function draw(now) {
    if (stopped) return;
    requestAnimationFrame(draw);
    if (!camera || video.readyState < 2) return;
    const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale)); canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (busy || samRequest || framePending || now - lastFrame < 1000 / 24) return;
    if (send({type: 'frame', frame: canvas.toDataURL('image/jpeg', .8).split(',')[1]})) {
      framePending = true; lastFrame = now;
      frameTimer = setTimeout(() => { if (ws) ws.close(); }, 30000);
    }
  }
  requestAnimationFrame(draw);

  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null, micWanted = false, micRestart, base = '';
  function micButton() { $('mic-toggle').textContent = micWanted ? '◼ Stop microphone' : '◉ Start microphone'; $('mic-toggle').setAttribute('aria-pressed', String(micWanted)); }
  function startMic() {
    if (!micWanted || !Recognition) return;
    base = $('steve-draft').value.trim();
    const current = new Recognition(); recognition = current;
    current.lang = $('language').value; current.continuous = true; current.interimResults = true;
    // Lock only while recording so interim revisions cannot overwrite manual edits.
    $('steve-draft').readOnly = true;
    current.onstart = () => { if (recognition === current) $('mic-status').textContent = 'Listening to Steve…'; };
    current.onresult = event => {
      if (recognition !== current || !micWanted) return;
      const finals = [], partials = [];
      for (let i = 0; i < event.results.length; i++) (event.results[i].isFinal ? finals : partials).push(event.results[i][0].transcript.trim());
      const words = [...finals, ...partials].join(' ');
      $('steve-draft').value = [base, words].filter(Boolean).join(' ').slice(0, 2000);
      $('speech-interim').textContent = partials.length ? 'Listening: ' + partials.join(' ') : '';
      controls();
    };
    current.onerror = event => {
      if (recognition !== current) return;
      $('mic-status').textContent = `Microphone: ${event.error}. Check permissions or type your reply.`;
      if (event.error !== 'no-speech') { micWanted = false; micButton(); $('steve-draft').readOnly = false; }
    };
    current.onend = () => {
      if (recognition !== current) return;
      recognition = null; $('steve-draft').readOnly = false; $('speech-interim').textContent = '';
      if (micWanted) micRestart = setTimeout(startMic, 500);
    };
    try { current.start(); }
    catch (error) { recognition = null; micWanted = false; micButton(); $('steve-draft').readOnly = false; $('mic-status').textContent = `Cannot start speech recognition: ${error.name}`; }
  }
  function stopMic() {
    micWanted = false; clearTimeout(micRestart); const old = recognition; recognition = null;
    if (old) old.abort();
    $('steve-draft').readOnly = false; $('speech-interim').textContent = ''; micButton();
    $('mic-status').textContent = 'Microphone is off. Review or edit your draft.';
  }
  $('mic-toggle').onclick = () => {
    if (micWanted) { stopMic(); return; }
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    micWanted = true; micButton(); startMic();
  };
  $('language').onchange = () => { if (micWanted) { stopMic(); micWanted = true; micButton(); startMic(); } };
  if (!Recognition) { $('mic-toggle').disabled = true; $('mic-status').textContent = 'Speech recognition unavailable. Use Chrome, or type your reply.'; }
  if (!window.speechSynthesis) $('tts').disabled = true;
  $('tts').onchange = () => { if (!$('tts').checked && window.speechSynthesis) window.speechSynthesis.cancel(); };
  const heartbeat = setInterval(() => send({type: 'ping'}), 10000);
  window.addEventListener('pagehide', () => {
    stopped = true; clearInterval(heartbeat); clearTimeout(retryTimer); clearTimeout(previewTimer); doneFrame();
    stopCamera(); stopMic(); if (window.speechSynthesis) window.speechSynthesis.cancel(); if (ws) ws.close();
  });
  controls(); connect();
})();
