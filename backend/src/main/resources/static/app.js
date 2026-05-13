const STORAGE_KEY = 'kazgpt-history-v1';
const SELECTED_MODEL_KEY = 'kazgpt-model';

const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const modelSelect = document.getElementById('modelSelect');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

let history = loadHistory();
let isStreaming = false;

modelSelect.value = localStorage.getItem(SELECTED_MODEL_KEY) || 'base';
modelSelect.addEventListener('change', () => {
    localStorage.setItem(SELECTED_MODEL_KEY, modelSelect.value);
});

if (history.length > 0) {
    document.querySelector('.welcome')?.remove();
    history.forEach(m => renderMessage(m.role === 'user' ? 'user' : 'bot', m.content));
}

document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
        inputEl.value = btn.dataset.q;
        inputEl.focus();
        autosize();
    });
});

inputEl.addEventListener('input', autosize);
inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});
sendBtn.addEventListener('click', send);

newChatBtn.addEventListener('click', () => {
    if (isStreaming) return;
    history = [];
    saveHistory();
    chatEl.innerHTML = '';
    chatEl.appendChild(buildWelcome());
    bindSuggestions();
});

async function send() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    document.querySelector('.welcome')?.remove();

    addMessage('user', text);
    inputEl.value = '';
    autosize();

    const botEl = addMessage('bot', '');
    botEl.classList.add('streaming');
    setStreaming(true);

    let accumulator = '';

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

        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            const tokens = parseSSE(chunk);
            for (const t of tokens) {
                accumulator += t;
                updateMessage(botEl, accumulator);
            }
        }

        history[history.length - 1].content = accumulator || '[бос жауап]';
        saveHistory();
        setStatus('ok', 'Дайын');
    } catch (err) {
        console.error(err);
        updateMessage(botEl, '[Қате: бэкэндке қол жеткізу мүмкін болмады. Ollama іске қосылғанын тексеріңіз.]');
        setStatus('error', 'Қате');
    } finally {
        botEl.classList.remove('streaming');
        setStreaming(false);
    }
}

function parseSSE(chunk) {
    const tokens = [];
    const lines = chunk.split('\n');
    for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;
        if (line.startsWith('data:')) {
            const data = line.slice(5).trimStart();
            if (data && data !== '[DONE]') tokens.push(data);
        } else if (line === ':') {
            continue;
        }
    }
    return tokens;
}

function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + role;
    el.innerHTML = `
        <div class="avatar">${role === 'user' ? 'Сіз' : 'K'}</div>
        <div class="bubble"></div>
    `;
    el.querySelector('.bubble').innerHTML = renderMarkdown(text);
    chatEl.appendChild(el);
    chatEl.scrollTop = chatEl.scrollHeight;

    if (role === 'user') {
        history.push({ role: 'user', content: text });
    } else {
        history.push({ role: 'assistant', content: text });
    }
    saveHistory();
    return el;
}

function renderMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + role;
    el.innerHTML = `
        <div class="avatar">${role === 'user' ? 'Сіз' : 'K'}</div>
        <div class="bubble">${renderMarkdown(text)}</div>
    `;
    chatEl.appendChild(el);
}

function updateMessage(el, text) {
    el.querySelector('.bubble').innerHTML = renderMarkdown(text);
    chatEl.scrollTop = chatEl.scrollHeight;
}

function renderMarkdown(text) {
    return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(s) {
    return s
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
}

function autosize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}

function setStreaming(on) {
    isStreaming = on;
    sendBtn.disabled = on;
    inputEl.disabled = on;
    if (on) setStatus('', 'Жазып жатыр...');
}

function setStatus(cls, text) {
    statusDot.className = 'status-dot ' + (cls || '');
    statusText.textContent = text;
}

function saveHistory() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(history)); } catch (e) {}
}

function loadHistory() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
}

function buildWelcome() {
    const w = document.createElement('div');
    w.className = 'welcome';
    w.innerHTML = `
        <div class="welcome-icon">🤖</div>
        <h1>KazGPT</h1>
        <p>Қазақ тілінде еркін сөйлейтін жасанды интеллект көмекшісі</p>
        <div class="suggestions">
            <button class="suggestion" data-q="Сәлем! Өзің туралы айтып бер">Өзің туралы айт</button>
            <button class="suggestion" data-q="Алматы туралы қысқаша мәлімет бер">Алматы туралы</button>
            <button class="suggestion" data-q="Қазақстанның астанасы қандай?">Астана туралы</button>
            <button class="suggestion" data-q="Нейрондық желі дегеніміз не?">Нейрондық желі</button>
        </div>
    `;
    return w;
}

function bindSuggestions() {
    document.querySelectorAll('.suggestion').forEach(btn => {
        btn.addEventListener('click', () => {
            inputEl.value = btn.dataset.q;
            inputEl.focus();
            autosize();
        });
    });
}

checkHealth();
setInterval(checkHealth, 15000);

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.ollamaUp) setStatus('ok', 'Дайын · Ollama қосылған');
        else setStatus('error', 'Ollama қол жетімсіз');
    } catch (e) {
        setStatus('error', 'Бэкэнд жауап бермейді');
    }
}
