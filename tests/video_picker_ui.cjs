// Exercise the real dashboard script and event handlers with a small DOM adapter.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children = []; this.listeners = {}; this.files = []; this.style = {}; this.value = ''; this.hidden = false; this.disabled = false; this.textContent = ''; this.classList = { add() {}, remove() {}, toggle() {} }; }
  append(...items) { this.children.push(...items); }
  prepend(item) { this.children.unshift(item); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  setAttribute() {}
  removeAttribute() {}
  get firstElementChild() { return this.children[0]; }
}
const elements = new Map();
const element = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
let activeRequest;
class Upload {
  constructor() { this.upload = {}; this.headers = {}; activeRequest = this; }
  open(method, url) { this.method = method; this.url = url; }
  setRequestHeader(key, value) { this.headers[key] = value; }
  send(file) { this.file = file; }
  abort() { this.onabort(); this.onloadend(); }
}
class Stream {
  constructor(url) { this.url = url; this.listeners = {}; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  close() { this.closed = true; }
}
const context = vm.createContext({
  document: { getElementById: element, createElement: () => new Element(), createTextNode: (text) => ({textContent:text}), querySelectorAll: () => [] },
  window: { location: { protocol: 'file:' } },
  Option: function(name, value) { this.name=name; this.value=value; },
  XMLHttpRequest: Upload, EventSource: Stream,
  performance: { now: () => 1000 }, Intl, console, encodeURIComponent, setTimeout,
});
vm.runInContext(fs.readFileSync('frontend/app.js', 'utf8'), context);
const run = (code) => vm.runInContext(code, context);
run('liveConfig = {max_upload_bytes: 536870912, semantic_model: false};');
element('video-input').listeners.change({target:{files:[{name:'huge.mp4',size:536870913}]}});
assert.equal(element('video-analyze').disabled, true);
const file = {name:'my gallery clip.mp4',size:2000};
element('video-input').files = [file];
element('video-input').listeners.change({target:{files:[file]}});
assert.equal(element('video-analyze').disabled, false);
element('video-analyze').listeners.click();
assert.equal(activeRequest.method,'POST');
assert.equal(activeRequest.url,'/api/videos');
assert.equal(activeRequest.headers['X-Video-Name'],'my%20gallery%20clip.mp4');
assert.equal(activeRequest.file,file);
activeRequest.upload.onprogress({lengthComputable:true,loaded:100,total:200});
assert.match(element('video-selection').textContent,/50%/);
activeRequest.status=201;
activeRequest.responseText=JSON.stringify({video_id:'abc123',source:file.name,codec:'h264',width:640,height:360});
activeRequest.onload(); activeRequest.onloadend();
assert.equal(run('liveStream.url'),'/api/stream?video=abc123');
assert.equal(element('scenario-name').textContent,file.name);
assert.equal(element('video-analyze').disabled,false);
run(`renderLiveFrame({type:'frame_result',generation:1,status:'READY',end_to_end_frame_seconds:0.01,source_codec:'h264',source_size:[640,360],pages:[],tensor_storage:{payload_bytes:16},kv_reuse:{pages_reused:3,pages_recomputed:4},baseline_comparison:{same_top1:true}});`);
run(`renderLiveFrame({type:'frame_result',generation:2,status:'READY',end_to_end_frame_seconds:0.03,source_codec:'h264',source_size:[640,360],metadata_fallback:true,pages:[],tensor_storage:{payload_bytes:32},kv_reuse:{pages_reused:2,pages_recomputed:5},baseline_comparison:{same_top1:false}});`);
run("liveStream.listeners.done()");
const values = element('video-result-metrics').children.map((card)=>card.children[1].textContent);
assert.deepEqual(values,['2','20 ms','1 / 2','32 B','5','1 / 2']);
assert.equal(element('video-results').hidden,false);
element('live-restart').listeners.click();
assert.equal(run('liveStream.url'),'/api/stream?video=abc123');
assert.equal(run('liveRecords.length'),0);
assert.equal(element('video-results').hidden,true);
element('video-analyze').listeners.click();
activeRequest.status=400;
activeRequest.responseText=JSON.stringify({error:'Invalid video'});
activeRequest.onload(); activeRequest.onloadend();
assert.equal(element('video-selection').textContent,'Invalid video');
assert.equal(run('liveVideoId'),'abc123');
assert.equal(element('video-input').disabled,false);
element('video-analyze').listeners.click();
element('video-cancel').listeners.click();
assert.match(element('video-selection').textContent,/canceled/);
assert.equal(element('video-analyze').disabled,false);
console.log('PASS: file selection, upload progress, source switching, results arithmetic, replay reset, invalid upload and cancellation.');
