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

modelSelect.value = localStorage.getItem(SELECTED_MODEL_KEY) || 'v5';
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
    window.location.href = '/';
});

// ── Auto-send prompt from landing page ?q= param ─────────────
window.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const initialQ = params.get('q');
    if (initialQ) {
        history = [];
        saveHistory();
        history.replaceState({}, '', 'chat.html');
        document.querySelector('.welcome')?.remove();
        inputEl.value = initialQ;
        autosize();
        setTimeout(() => send(), 350);
    }
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
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let idx;
            while ((idx = buffer.indexOf('\n\n')) >= 0) {
                const event = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                const token = parseSSEEvent(event);
                if (token) {
                    accumulator += token;
                    updateMessage(botEl, accumulator);
                }
            }
        }
        if (buffer.trim()) {
            const token = parseSSEEvent(buffer);
            if (token) {
                accumulator += token;
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

function parseSSEEvent(eventText) {
    const lines = eventText.split('\n');
    const dataLines = [];
    for (const raw of lines) {
        if (raw.startsWith('data:')) {
            // Spring's SSE encoder writes "data:<value>" without a cosmetic space,
            // so leading whitespace in the value is genuine content (BPE tokens often
            // begin with " "). Preserve it exactly.
            dataLines.push(raw.slice(5));
        }
    }
    if (dataLines.length === 0) return null;
    const joined = dataLines.join('\n');
    if (joined === '[DONE]') return null;
    return joined;
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
    // 1. Сохраняем код-блоки ДО экранирования HTML —
    //    иначе escapeHtml испортит содержимое и regex не сработает.
    const codeBlocks = [];
    let result = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        codeBlocks.push({ lang: lang || '', code: code.replace(/\n$/, '') });
        return `\x02CODE${codeBlocks.length - 1}\x03`;
    });

    // 2. Экранируем HTML во всём остальном тексте
    result = escapeHtml(result);

    // 3. Inline-код: `однострочник`
    result = result.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // 4. Жирный текст **bold**
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // 5. Переносы строк → <br>
    result = result.replace(/\n/g, '<br>');

    // 6. Возвращаем код-блоки оформленными как <pre><code>
    result = result.replace(/\x02CODE(\d+)\x03/g, (_, i) => {
        const { lang, code } = codeBlocks[parseInt(i)];
        const escaped = escapeHtml(code);
        const label = lang ? `<span class="code-lang">${lang}</span>` : '';
        return `<pre>${label}<code>${escaped}</code></pre>`;
    });

    return result;
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
