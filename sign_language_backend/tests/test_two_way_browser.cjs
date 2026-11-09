const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
function setup() {
  const elements = new Map(), sockets = [], speech = [], timers = new Map(), handlers = {}, spoken = [];
  let timerId = 0;
  const el = id => {
    if (!elements.has(id)) elements.set(id, {value:'',textContent:'',children:[],setAttribute(){},getContext:()=>({clearRect(){},drawImage(){}}),
      appendChild(child){this.children.push(child);},scrollTo(options){this.scrollOptions=options;},readOnly:false});
    return elements.get(id);
  };
  el('language').value='en-US';
  class Socket {
    static OPEN=1;
    constructor(url){this.url=url;this.readyState=1;this.sent=[];sockets.push(this);}
    send(data){this.sent.push(JSON.parse(data));}
    close(){this.readyState=3;this.onclose({code:1000});}
  }
  class Speech {
    constructor(){speech.push(this);}
    start(){this.onstart();}
    abort(){this.aborted=true;if(this.onend)this.onend();}
  }
  let serial=0;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../frontend/app.js'),'utf8'),{
    document:{getElementById:el,createElement:()=>({children:[],appendChild(child){this.children.push(child);}})},
    window:{matchMedia:()=>({matches:false}),webkitSpeechRecognition:Speech,speechSynthesis:{speak:x=>spoken.push(x.text),cancel(){}},addEventListener:(key,fn)=>handlers[key]=fn},
    location:{hash:'#test-room',host:'localhost:8000',protocol:'http:'},history:{replaceState(){}},
    crypto:{randomUUID:()=>`request-${++serial}`},WebSocket:Socket,SpeechSynthesisUtterance:class{constructor(text){this.text=text;}},
    setTimeout:(fn,ms)=>{timers.set(++timerId,{fn,ms});return timerId;},clearTimeout:id=>timers.delete(id),
    setInterval:()=>++timerId,clearInterval(){},requestAnimationFrame(){},navigator:{}
  });
  const ws=sockets[0];ws.onopen();
  return {el,ws,sockets,speech,timers,spoken,handlers,receive:data=>ws.onmessage({data:JSON.stringify(data)})};
}
test('both drafts require confirmation; acknowledgements clear only the matching sender',()=>{
  const s=setup();
  s.el('steve-draft').value='Hello Sam.';s.el('steve-draft').oninput();
  assert.equal(s.ws.sent.length,0);
  assert.equal(s.el('chat-stream').children.length,0);
  s.el('send-steve').onclick();
  const request=s.ws.sent.at(-1);
  assert.equal(request.sender,'steve');assert.equal(request.text,'Hello Sam.');
  assert.equal(s.el('steve-draft').value,'Hello Sam.');
  s.receive({type:'message',message_id:'m1',sender:'steve',recipient:'sam',text:'Hello Sam.',request_id:request.request_id});
  assert.equal(s.el('steve-draft').value,'');
  assert.equal(s.el('chat-stream').children[0].className,'chat-message from-steve');
  assert.equal(s.el('chat-stream').children[0].children[0].textContent,'Steve (Voice)');
  assert.equal(s.el('chat-stream').children[0].children[1].textContent,'Hello Sam.');
  s.receive({type:'draft',sentence:'Please bring water.',glosses:['water','please']});
  assert.equal(s.el('sam-draft').value,'Please bring water.');assert.equal(s.ws.sent.length,1);
  assert.equal(s.el('chat-stream').children.length,1); // Preview is not a chat message.
  s.el('sam-draft').value='Please bring cold water.';s.el('send-sam').onclick();
  assert.equal(s.ws.sent.at(-1).text,'Please bring cold water.');
  s.receive({type:'sent',sender:'sam',glosses:[]});
  assert.equal(s.el('sam-draft').value,'');
});
test('unchanged frame glosses do not postpone preview; speech revisions stay staged; TTS stops capture',()=>{
  const s=setup();
  s.receive({type:'frame_result',status:'prediction',glosses:['water']});
  const preview=[...s.timers.values()].find(t=>t.ms===900);
  s.receive({type:'frame_result',status:'prediction',glosses:['water']});
  assert.equal([...s.timers.values()].filter(t=>t.ms===900).length,1);
  preview.fn();assert.equal(s.ws.sent.at(-1).type,'stage');
  s.el('mic-toggle').onclick();
  const rec=s.speech[0];assert.equal(rec.interimResults,true);
  const result=text=>rec.onresult({results:[Object.assign([{transcript:text}],{isFinal:false})]});
  result('Hello wor');result('Hello world');
  assert.equal(s.el('steve-draft').value,'Hello world');
  assert.equal(s.ws.sent.length,1);
  s.el('tts').checked=true;
  s.receive({type:'message',message_id:'m2',sender:'sam',recipient:'steve',text:'Hello Steve.'});
  assert.equal(rec.aborted,true);assert.deepEqual(s.spoken,['Hello Steve.']);
  s.receive({type:'message',message_id:'m2',sender:'sam',recipient:'steve',text:'Hello Steve.'});
  assert.equal(s.el('chat-stream').children.length,1);
  assert.equal(s.el('chat-stream').children[0].className,'chat-message from-sam');
  assert.equal(s.el('chat-stream').children[0].children[0].textContent,'Sam (Sign)');
  assert.equal(s.el('chat-stream').scrollOptions.behavior,'smooth');
});
test('disconnect preserves unsent drafts without replay; cleanup closes connection',()=>{
  const s=setup();s.el('sam-draft').value='Unsent words.';s.ws.close();
  assert.equal(s.el('sam-draft').value,'Unsent words.');
  assert.match(s.el('connection').textContent,/not retried/);
  [...s.timers.values()].find(t=>t.ms===500).fn();
  assert.equal(s.sockets[1].sent.length,0);
  s.handlers.pagehide();assert.equal(s.sockets[1].readyState,3);
});
