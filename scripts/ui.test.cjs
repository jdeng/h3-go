const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function page() {
  const elements = new Map();
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      value: '', hidden: false, disabled: false, checked: false, textContent: '',
      listeners: {}, classList: { toggle() {}, add() {}, remove() {} },
      addEventListener(name, handler) { this.listeners[name] = handler; },
      removeAttribute(name) { delete this[name]; }, pause() {}, load() {},
    });
    return elements.get(selector);
  };
  element('#form').elements = Object.fromEntries(Object.entries({
    aspect_ratio: 'source', resolution: '640', width: '640', height: '384',
    duration: '10', steps: '8', cfg: '1', negative_prompt: '', lora_strength: '1',
    turbo: '', audio: '', seed: '',
  }).map(([name, value]) => [name, { ...element(name), value, checked: true }]));
  const images = [];
  const urls = new Set();
  let sequence = 0;
  const context = vm.createContext({
    document: { querySelector: element, addEventListener() {} },
    Image: class { constructor() { images.push(this); } },
    DataTransfer: class { items = { add() {} }; files = []; },
    URL: {
      createObjectURL() { const url = `blob:${++sequence}`; urls.add(url); return url; },
      revokeObjectURL(url) { urls.delete(url); },
    },
    fetch: async () => ({ ok: true, json: async () => ({ busy: false }) }),
    confirm: () => true, setInterval() {}, clearInterval() {},
  });
  const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  return {
    context, element, urls,
    evaluate: (source) => vm.runInContext(source, context),
    load(slot, name, size = 1) {
      return vm.runInContext(`loadFiles('${slot}', [{name:'${name}',type:'image/png',size:${size}}])`, context);
    },
    decode(index, width = 640, height = 384) {
      Object.assign(images[index], { naturalWidth: width, naturalHeight: height });
      images[index].onload();
    },
    fail(index) { images[index].onerror(); },
  };
}

test('the latest selection wins when decoding completes out of order', async () => {
  const view = page();
  const older = view.load('first', 'older.png');
  const newer = view.load('first', 'newer.png');
  view.decode(1);
  await newer;
  view.decode(0);
  await older;
  assert.equal(view.evaluate('references.first.file.name'), 'newer.png');
  assert.equal(view.urls.size, 1);
});

test('removing a pending keyframe prevents it from reappearing', async () => {
  const view = page();
  const loading = view.load('first', 'removed.png');
  assert.equal(view.element('#first-remove').hidden, false);
  view.element('#first-remove').listeners.click();
  view.decode(0);
  await loading;
  assert.equal(view.evaluate('references.first'), null);
  assert.equal(view.element('#status').textContent, 'Keyframe removed.');
  assert.equal(view.urls.size, 0);
});

test('generation stays disabled during decoding, including health refreshes', async () => {
  const view = page();
  const loading = view.load('last', 'pending.png');
  await view.evaluate('checkHealth()');
  assert.equal(view.element('#submit').disabled, true);
  view.decode(0);
  await loading;
  assert.equal(view.element('#submit').disabled, false);
});

for (const [name, size, width, height, message] of [
  ['bytes', 32 * 1024 * 1024, 640, 384, /63 MiB/],
  ['pixels', 1, 5000, 5000, /40 MP/],
]) {
  test(`simultaneous first/last uploads enforce combined ${name}`, async () => {
    const view = page();
    const first = view.load('first', 'first.png', size);
    const last = view.load('last', 'last.png', size);
    view.decode(0, width, height);
    await first;
    view.decode(1, width, height);
    await last;
    assert.equal(view.evaluate('references.last'), null);
    assert.match(view.element('#status').textContent, message);
    assert.equal(view.urls.size, 1);
  });
}

test('a failed replacement preserves the selected image and releases its URL', async () => {
  const view = page();
  const selected = view.load('first', 'selected.png');
  view.decode(0);
  await selected;
  const replacement = view.load('first', 'broken.png');
  view.fail(1);
  await replacement;
  assert.equal(view.evaluate('references.first.file.name'), 'selected.png');
  assert.equal(view.urls.size, 1);
  assert.equal(view.element('#submit').disabled, false);
});

test('deleting an old result does not clear a newer preview', async () => {
  const view = page();
  let finish;
  view.context.fetch = () => new Promise((resolve) => { finish = resolve; });
  view.evaluate("currentResult = {filename:'old.mp4'}");
  const deleting = view.element('#delete').listeners.click();
  view.evaluate("currentResult = {filename:'new.mp4'}; video.src = '/outputs/new.mp4'");
  finish({ ok: true });
  await deleting;
  assert.equal(view.evaluate('currentResult.filename'), 'new.mp4');
  assert.equal(view.element('#video').src, '/outputs/new.mp4');
});
