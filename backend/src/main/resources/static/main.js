/* ════════════════════════════════════════════════════════════════
   KazGPT — Main JS (single-page: landing + chat in one)
   ════════════════════════════════════════════════════════════════ */

const STORAGE_KEY   = 'kazgpt-history-v2';
const MODEL_KEY     = 'kazgpt-model';

// ── Elements ─────────────────────────────────────────────────
const app         = document.getElementById('app');
const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('input');
const sendBtn     = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const statusDot   = document.getElementById('statusDot');
const statusText  = document.getElementById('statusText');
const newChatBtn  = document.getElementById('newChatBtn');
const heroEl      = document.getElementById('hero');
const footerEl    = document.getElementById('footer');
const headerMid   = document.getElementById('headerMid');

// ── State ─────────────────────────────────────────────────────
let history     = loadHistory();
let isStreaming = false;
let inChatMode  = false;

// ── Init model ────────────────────────────────────────────────
modelSelect.value = localStorage.getItem(MODEL_KEY) || 'v5';
modelSelect.addEventListener('change', () => localStorage.setItem(MODEL_KEY, modelSelect.value));

// ── Restore chat if history exists ───────────────────────────
if (history.length > 0) {
  enterChatMode();
  history.forEach(m => renderMsg(m.role === 'user' ? 'user' : 'bot', m.content));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Typewriter ────────────────────────────────────────────────
if (!inChatMode) {
  const lines = [
    'Қазақ тілінде сөйлейтін жасанды интеллект',
    'Интернетсіз · Толығымен локальды жүйе',
    'QLoRA Fine-tuned Qwen2.5-7B · 4.4 GB',
  ];
  let li = 0, ci = 0, del = false, pause = false;
  const subEl = document.getElementById('heroSub');

  function type() {
    if (!subEl || inChatMode) return;
    const cur = lines[li];
    if (pause) return;
    if (!del) {
      ci++;
      subEl.innerHTML = esc(cur.slice(0, ci)) + '<span class="tw-cursor"></span>';
      if (ci === cur.length) { pause = true; setTimeout(() => { pause = false; del = true; type(); }, 2800); return; }
      setTimeout(type, 44);
    } else {
      ci--;
      subEl.innerHTML = esc(cur.slice(0, ci)) + '<span class="tw-cursor"></span>';
      if (ci === 0) {
        del = false; li = (li + 1) % lines.length;
        pause = true; setTimeout(() => { pause = false; setTimeout(type, 80); }, 350); return;
      }
      setTimeout(type, 22);
    }
  }
  setTimeout(type, 900);
}

// ── Chips ─────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(c => {
  c.addEventListener('click', () => {
    inputEl.value = c.dataset.q;
    resize();
    inputEl.focus();
    updateSendBtn();
  });
});

// ── Input ─────────────────────────────────────────────────────
inputEl.addEventListener('input', () => { resize(); updateSendBtn(); });
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

function resize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}
function updateSendBtn() {
  sendBtn.disabled = !inputEl.value.trim() || isStreaming;
}

// ── New chat ──────────────────────────────────────────────────
newChatBtn.addEventListener('click', () => {
  if (isStreaming) return;
  history = []; saveHistory();
  messagesEl.innerHTML = '';
  exitChatMode();
});

// ── Mode transitions ──────────────────────────────────────────
function enterChatMode() {
  if (inChatMode) return;
  inChatMode = true;
  app.classList.add('app--chat');
  newChatBtn.style.display = 'flex';

  // inject model select into header center
  if (!document.getElementById('headerModelSel')) {
    const sel = document.createElement('select');
    sel.id = 'headerModelSel';
    sel.className = 'header-model';
    sel.innerHTML = modelSelect.innerHTML;
    sel.value = modelSelect.value;
    sel.addEventListener('change', () => {
      modelSelect.value = sel.value;
      localStorage.setItem(MODEL_KEY, sel.value);
    });
    modelSelect.addEventListener('change', () => { sel.value = modelSelect.value; });
    headerMid.appendChild(sel);
  }
}
function exitChatMode() {
  inChatMode = false;
  app.classList.remove('app--chat');
  newChatBtn.style.display = 'none';
  const sel = document.getElementById('headerModelSel');
  if (sel) sel.remove();
  // restart typewriter
  const subEl = document.getElementById('heroSub');
  if (subEl) subEl.innerHTML = '&nbsp;';
}

// ── Send ──────────────────────────────────────────────────────
async function send() {
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;

  if (!inChatMode) enterChatMode();

  inputEl.value = '';
  resize();
  updateSendBtn();

  addMsg('user', text);

  // typing indicator
  const typingEl = addTyping();
  setStreaming(true);

  let acc = '';
  let botEl = null;

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: history.slice(0, -1),
        model: modelSelect.value
      })
    });

    typingEl.remove();
    botEl = addMsg('bot', '');
    botEl.classList.add('streaming');

    if (!res.ok) throw new Error('HTTP ' + res.status);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const tok = parseSSE(buf.slice(0, idx));
        buf = buf.slice(idx + 2);
        if (tok) { acc += tok; updateMsg(botEl, acc); }
      }
    }
    if (buf.trim()) { const tok = parseSSE(buf); if (tok) { acc += tok; updateMsg(botEl, acc); } }

    history[history.length - 1].content = acc || '[бос жауап]';
    saveHistory();
    setStatus('ok', 'Дайын');
  } catch (err) {
    typingEl?.remove();
    if (!botEl) botEl = addMsg('bot', '');
    updateMsg(botEl, '[Қате: ' + err.message + '. Ollama іске қосылғанын тексеріңіз.]');
    setStatus('err', 'Қате');
  } finally {
    botEl?.classList.remove('streaming');
    setStreaming(false);
  }
}

// ── DOM helpers ───────────────────────────────────────────────
function addMsg(role, text) {
  const el = document.createElement('div');
  el.className = `msg msg--${role}`;
  el.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'Сіз' : 'K'}</div>
    <div class="msg-bubble"></div>
  `;
  el.querySelector('.msg-bubble').innerHTML = md(text);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  if (role === 'user') history.push({ role: 'user', content: text });
  else                 history.push({ role: 'assistant', content: text });
  saveHistory();
  return el;
}

function renderMsg(role, text) {
  const el = document.createElement('div');
  el.className = `msg msg--${role}`;
  el.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'Сіз' : 'K'}</div>
    <div class="msg-bubble">${md(text)}</div>
  `;
  messagesEl.appendChild(el);
}

function updateMsg(el, text) {
  el.querySelector('.msg-bubble').innerHTML = md(text);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTyping() {
  const el = document.createElement('div');
  el.className = 'msg msg--bot';
  el.innerHTML = `
    <div class="msg-avatar">K</div>
    <div class="msg-bubble">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

// ── Markdown ──────────────────────────────────────────────────
function md(text) {
  const blocks = [];
  let r = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push({ lang: lang || '', code: code.replace(/\n$/, '') });
    return `\x02B${blocks.length - 1}\x03`;
  });
  r = esc(r);
  r = r.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  r = r.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  r = r.replace(/\n/g, '<br>');
  r = r.replace(/\x02B(\d+)\x03/g, (_, i) => {
    const { lang, code } = blocks[parseInt(i)];
    const label = lang ? `<span class="code-lang">${lang}</span>` : '';
    return `<pre>${label}<code>${esc(code)}</code></pre>`;
  });
  return r;
}
function esc(s) {
  return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
}

// ── SSE parser ────────────────────────────────────────────────
function parseSSE(raw) {
  const lines = raw.split('\n').filter(l => l.startsWith('data:'));
  if (!lines.length) return null;
  const joined = lines.map(l => l.slice(5)).join('\n');
  return joined === '[DONE]' ? null : joined;
}

// ── Streaming state ───────────────────────────────────────────
function setStreaming(on) {
  isStreaming = on;
  sendBtn.disabled = on || !inputEl.value.trim();
  inputEl.disabled = on;
  if (on) setStatus('', 'Жазып жатыр...');
}

// ── Status ────────────────────────────────────────────────────
function setStatus(cls, text) {
  statusDot.className = 'status-dot' + (cls ? ' ' + cls : '');
  statusText.textContent = text;
}

// ── History ───────────────────────────────────────────────────
function saveHistory() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(history)); } catch {} }
function loadHistory() { try { const r = localStorage.getItem(STORAGE_KEY); return r ? JSON.parse(r) : []; } catch { return []; } }

// ── Health ────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const d = await fetch('/api/health').then(r => r.json());
    if (d.ollamaUp) setStatus('ok', 'Дайын');
    else setStatus('err', 'Ollama жоқ');
  } catch { setStatus('err', 'Офлайн'); }
}
checkHealth();
setInterval(checkHealth, 15000);
