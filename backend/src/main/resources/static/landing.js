const SELECTED_MODEL_KEY = 'kazgpt-model';

// ── Elements ──────────────────────────────────────────────────
const promptInput = document.getElementById('promptInput');
const sendBtn     = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const promptCard  = document.getElementById('promptCard');
const taglineEl   = document.getElementById('tagline');
const navStatus   = document.getElementById('navStatus');
const navStatusText = document.getElementById('navStatusText');

// ── Restore model selection ───────────────────────────────────
modelSelect.value = localStorage.getItem(SELECTED_MODEL_KEY) || 'v5';
modelSelect.addEventListener('change', () => {
    localStorage.setItem(SELECTED_MODEL_KEY, modelSelect.value);
});

// ── Typewriter tagline ────────────────────────────────────────
const lines = [
    'Қазақ тілінде сөйлейтін жасанды интеллект',
    'Толығымен локальды · Интернетсіз жұмыс істейді',
    'QLoRA Fine-tuned Qwen2.5-7B · 4.4 GB GGUF',
];
let lineIdx = 0, charIdx = 0, deleting = false, pausing = false;

function typewriter() {
    const current = lines[lineIdx];
    if (pausing) return;

    if (!deleting) {
        charIdx++;
        taglineEl.innerHTML = escapeHtml(current.slice(0, charIdx)) + '<span class="cursor"></span>';
        if (charIdx === current.length) {
            pausing = true;
            setTimeout(() => { pausing = false; deleting = true; }, 2800);
        }
        setTimeout(typewriter, 42);
    } else {
        charIdx--;
        taglineEl.innerHTML = escapeHtml(current.slice(0, charIdx)) + '<span class="cursor"></span>';
        if (charIdx === 0) {
            deleting = false;
            lineIdx = (lineIdx + 1) % lines.length;
            pausing = true;
            setTimeout(() => { pausing = false; setTimeout(typewriter, 60); }, 300);
        } else {
            setTimeout(typewriter, 22);
        }
    }
}
setTimeout(typewriter, 800);

function escapeHtml(s) {
    return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

// ── Prompt card focus glow ────────────────────────────────────
promptInput.addEventListener('focus', () => promptCard.classList.add('focused'));
promptInput.addEventListener('blur',  () => promptCard.classList.remove('focused'));

// ── Autosize textarea ─────────────────────────────────────────
promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
});

// ── Suggestion chips ──────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        promptInput.value = chip.dataset.q;
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
        promptInput.focus();
    });
});

// ── Send / navigate to chat ───────────────────────────────────
function goToChat() {
    const q = promptInput.value.trim();
    if (!q) { promptInput.focus(); return; }

    localStorage.setItem(SELECTED_MODEL_KEY, modelSelect.value);

    // Fade out then navigate
    document.body.classList.add('fade-out');
    setTimeout(() => {
        window.location.href = 'chat.html?q=' + encodeURIComponent(q);
    }, 280);
}

sendBtn.addEventListener('click', goToChat);
promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        goToChat();
    }
});

// ── Health check ──────────────────────────────────────────────
async function checkHealth() {
    try {
        const data = await fetch('/api/health').then(r => r.json());
        if (data.ollamaUp) {
            navStatus.className = 'status-dot ok';
            navStatusText.textContent = 'Дайын';
        } else {
            navStatus.className = 'status-dot error';
            navStatusText.textContent = 'Ollama жоқ';
        }
    } catch {
        navStatus.className = 'status-dot error';
        navStatusText.textContent = 'Офлайн';
    }
}
checkHealth();
setInterval(checkHealth, 15000);
