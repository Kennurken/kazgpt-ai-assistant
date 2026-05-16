#!/usr/bin/env python3
"""
KazGPT — Максимальный сборщик казахских данных
Собирает ВСЁ: HuggingFace + Wikipedia + Законы РК + Новости + Литература

Запуск:
    python collect_all.py
    python collect_all.py --limit 50000   # лимит на источник
    python collect_all.py --skip-scrape   # только HF, без парсинга
"""

import json, re, hashlib, time, argparse, sys, random
from pathlib import Path
from datetime import datetime

random.seed(42)
DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = Path(__file__).parent / "data" / "kazgpt_full_raw.jsonl"

STATS = {}

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_jsonl(records, name):
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    STATS[name] = len(records)
    log(f"  ✓ {name}: {len(records):,} примеров → {path.name}")
    return records

def to_instruction(prompt, output, source=""):
    return {
        "prompt": prompt.strip(),
        "completion": " " + output.strip(),
        "source": source
    }

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)          # убрать HTML теги
    text = re.sub(r'\[\d+\]', '', text)           # убрать [1] [2]
    text = re.sub(r'http\S+', '', text)           # убрать URL
    return text.strip()

def is_kazakh(text, min_kk_chars=0.05):
    """Простая проверка — казахские буквы (ә,ғ,қ,ң,ө,ұ,ү,һ,і)"""
    kk_specific = set('әғқңөұүһі')
    total = len([c for c in text.lower() if c.isalpha()])
    if total == 0: return False
    kk_count = sum(1 for c in text.lower() if c in kk_specific)
    return (kk_count / total) >= min_kk_chars

def is_quality(text, min_len=80, max_len=8000):
    text = text.strip()
    if len(text) < min_len or len(text) > max_len: return False
    if not is_kazakh(text): return False
    words = text.split()
    if len(words) < 15: return False
    return True

def deduplicate(records):
    seen = set()
    out = []
    for r in records:
        h = hashlib.md5((r["prompt"] + r["completion"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out

# ══════════════════════════════════════════════════════════════════
# 1. HUGGINGFACE DATASETS
# ══════════════════════════════════════════════════════════════════

def load_kazqad(limit=999999):
    log("📚 [1/8] issai/kazqad — QA датасет ISSAI/NU")
    try:
        from datasets import load_dataset
        ds = load_dataset("issai/kazqad", split="train")
        records = []
        for item in ds:
            q = item.get("question") or item.get("instruction") or ""
            a = item.get("answers") or item.get("answer") or item.get("output") or ""
            if isinstance(a, dict):
                a = (a.get("text") or [""])[0]
            elif isinstance(a, list):
                a = a[0] if a else ""
            q, a = clean_text(str(q)), clean_text(str(a))
            if q and a and len(a) > 20:
                records.append(to_instruction(q, a, "kazqad"))
            if len(records) >= limit: break
        return save_jsonl(records, "kazqad")
    except Exception as e:
        log(f"  ✗ KazQAD ошибка: {e}"); return []

def load_wikipedia_kk(limit=100000):
    log("🌐 [2/8] wikipedia (kk) — Казахская Уикипедия")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20231101.kk", split="train", streaming=True)
        records = []
        for item in ds:
            title = clean_text(item.get("title", ""))
            text  = clean_text(item.get("text", ""))
            if not title or not text or not is_quality(text, min_len=200):
                continue
            # → задача: суммаризация
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
            if len(sentences) >= 3:
                summary = '. '.join(sentences[:3]) + '.'
                records.append(to_instruction(
                    f"«{title}» туралы қысқаша айтып бер.",
                    summary, "wikipedia_kk"
                ))
            # → задача: продолжить текст
            words = text.split()
            if len(words) > 60:
                cut = len(words) // 3
                records.append(to_instruction(
                    ' '.join(words[:cut]),
                    ' '.join(words[cut:cut+120]),
                    "wikipedia_kk_completion"
                ))
            if len(records) >= limit: break
        return save_jsonl(records, "wikipedia_kk")
    except Exception as e:
        log(f"  ✗ Wikipedia ошибка: {e}"); return []

def load_mc4_kk(limit=50000):
    log("🔷 [3/8] mc4 (kk) — Веб-корпус на казахском")
    try:
        from datasets import load_dataset
        ds = load_dataset("mc4", "kk", split="train", streaming=True, trust_remote_code=True)
        records = []
        for item in ds:
            text = clean_text(item.get("text", ""))
            if not is_quality(text, min_len=150): continue
            words = text.split()
            if len(words) < 40: continue
            cut = min(len(words) // 2, 100)
            records.append(to_instruction(
                ' '.join(words[:cut]),
                ' '.join(words[cut:cut+150]),
                "mc4_kk"
            ))
            if len(records) >= limit: break
        return save_jsonl(records, "mc4_kk")
    except Exception as e:
        log(f"  ✗ mc4 ошибка: {e}"); return []

def load_culturax_kk(limit=50000):
    log("✨ [4/8] CulturaX (kk) — Очищенный веб-корпус")
    try:
        from datasets import load_dataset
        ds = load_dataset("uonlp/CulturaX", "kk", split="train",
                         streaming=True, trust_remote_code=True)
        records = []
        for item in ds:
            text = clean_text(item.get("text", ""))
            if not is_quality(text, min_len=150): continue
            words = text.split()
            if len(words) < 40: continue
            cut = min(len(words) // 2, 100)
            records.append(to_instruction(
                ' '.join(words[:cut]),
                ' '.join(words[cut:cut+150]),
                "culturax_kk"
            ))
            if len(records) >= limit: break
        return save_jsonl(records, "culturax_kk")
    except Exception as e:
        log(f"  ✗ CulturaX ошибка: {e}"); return []

def load_oscar_kk(limit=50000):
    log("📖 [5/8] OSCAR-2301 (kk) — Common Crawl Казахстан")
    try:
        from datasets import load_dataset
        ds = load_dataset("oscar-corpus/OSCAR-2301", "kk", split="train",
                         streaming=True, trust_remote_code=True)
        records = []
        for item in ds:
            text = clean_text(item.get("content") or item.get("text") or "")
            if not is_quality(text, min_len=100): continue
            words = text.split()
            if len(words) < 30: continue
            cut = min(len(words) // 2, 80)
            records.append(to_instruction(
                ' '.join(words[:cut]),
                ' '.join(words[cut:cut+120]),
                "oscar_kk"
            ))
            if len(records) >= limit: break
        return save_jsonl(records, "oscar_kk")
    except Exception as e:
        log(f"  ✗ OSCAR ошибка: {e}"); return []

def load_multidomain_kk(limit=30000):
    log("🗂️ [6/8] multidomain-kazakh-dataset")
    try:
        from datasets import load_dataset
        ds = load_dataset("kz-transformers/multidomain-kazakh-dataset",
                         split="train", streaming=True)
        records = []
        for item in ds:
            text = clean_text(item.get("text") or item.get("content") or "")
            if not is_quality(text, min_len=100): continue
            words = text.split()
            if len(words) < 30: continue
            cut = min(len(words) // 2, 80)
            records.append(to_instruction(
                ' '.join(words[:cut]),
                ' '.join(words[cut:cut+120]),
                "multidomain_kk"
            ))
            if len(records) >= limit: break
        return save_jsonl(records, "multidomain_kk")
    except Exception as e:
        log(f"  ✗ multidomain ошибка: {e}"); return []

# ══════════════════════════════════════════════════════════════════
# 2. ВЕБ СКРАПИНГ
# ══════════════════════════════════════════════════════════════════

def scrape_adilet(limit=5000):
    log("⚖️ [7/8] adilet.zan.kz — Законы Республики Казахстан")
    try:
        import requests
        from bs4 import BeautifulSoup

        base = "https://adilet.zan.kz/kaz/docs"
        headers = {"User-Agent": "Mozilla/5.0 (academic research bot)"}

        # Список категорий законов
        categories = [
            "/code",      # Кодексы
            "/zakon",     # Законы
            "/ukaz",      # Указы
        ]

        records = []
        for cat in categories:
            if len(records) >= limit: break
            try:
                url = base + cat
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = [a['href'] for a in soup.find_all('a', href=True)
                        if '/kaz/docs/' in a['href']][:30]

                for link in links:
                    if len(records) >= limit: break
                    try:
                        full_url = f"https://adilet.zan.kz{link}" if link.startswith('/') else link
                        r = requests.get(full_url, headers=headers, timeout=10)
                        s = BeautifulSoup(r.text, 'html.parser')

                        title_el = s.find('h1') or s.find('h2')
                        title = clean_text(title_el.get_text()) if title_el else ""

                        content_el = s.find('div', class_='document-text') or s.find('div', id='content')
                        if not content_el:
                            content_el = s.find('div', class_='content')
                        if not content_el: continue

                        text = clean_text(content_el.get_text())
                        if not title or not is_quality(text, min_len=300): continue

                        records.append(to_instruction(
                            f"«{title}» заңы туралы қысқаша айтып бер.",
                            text[:1500],
                            "adilet_laws"
                        ))
                        records.append(to_instruction(
                            f"Мына заңды оқып, негізгі мазмұнын түсіндір: {title}",
                            text[:800],
                            "adilet_laws"
                        ))
                        time.sleep(0.5)
                    except Exception:
                        continue
            except Exception:
                continue

        return save_jsonl(records, "adilet_laws")
    except Exception as e:
        log(f"  ✗ Adilet ошибка: {e}"); return []

def scrape_kazakh_news(limit=10000):
    log("📰 [8/8] Казахские новостные сайты")
    try:
        import requests
        from bs4 import BeautifulSoup

        sources = [
            ("https://tengrinews.kz/kaz/", "tengri"),
            ("https://www.egemen.kz/", "egemen"),
            ("https://abai.kz/", "abai"),
        ]

        headers = {"User-Agent": "Mozilla/5.0 (academic research bot)"}
        records = []

        for base_url, source_name in sources:
            if len(records) >= limit: break
            try:
                resp = requests.get(base_url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')

                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/') and len(href) > 10:
                        full = base_url.rstrip('/') + href
                        links.append(full)
                    elif base_url in href:
                        links.append(href)

                links = list(set(links))[:50]

                for link in links:
                    if len(records) >= limit // len(sources): break
                    try:
                        r = requests.get(link, headers=headers, timeout=10)
                        s = BeautifulSoup(r.text, 'html.parser')

                        title_el = s.find('h1')
                        title = clean_text(title_el.get_text()) if title_el else ""

                        for tag in s(['script', 'style', 'nav', 'footer', 'header']):
                            tag.decompose()

                        article = s.find('article') or s.find('div', class_=re.compile('article|content|text|body'))
                        if not article: continue

                        text = clean_text(article.get_text())
                        if not title or not is_quality(text, min_len=200): continue

                        records.append(to_instruction(
                            f"Мына жаңалықты қысқаша баяндап бер: {title}",
                            text[:1000],
                            source_name
                        ))
                        time.sleep(0.3)
                    except Exception:
                        continue
            except Exception as e:
                log(f"    {source_name} қол жетімсіз: {e}")
                continue

        return save_jsonl(records, "kazakh_news")
    except Exception as e:
        log(f"  ✗ News ошибка: {e}"); return []

# ══════════════════════════════════════════════════════════════════
# 3. РУЧНЫЕ КАЧЕСТВЕННЫЕ ПРИМЕРЫ
# ══════════════════════════════════════════════════════════════════

def add_manual_seeds():
    log("✍️ [+] Ручные seed примеры (высокое качество)")
    records = [
        to_instruction("Сәлем!", "Сәлеметсіз! Мен KazGPT — қазақ тілінде сөйлейтін жасанды интеллект. Сізге қалай көмектесе аламын?", "manual"),
        to_instruction("Өзің туралы айтып бер", "Мен KazGPT V5 — Қыдырбек Елдос жасаған локальды жасанды интеллект. QLoRA әдісімен Qwen2.5-7B моделін KazQAD деректерімен дообучал. Толығымен оффлайн жұмыс істеймін.", "manual"),
        to_instruction("Python тілінде 'Сәлем, Әлем!' деп шығару кодын жаз", "```python\nprint('Сәлем, Әлем!')\n```\nБұл қарапайым Python кодысы. `print()` функциясы мәтінді экранға шығарады.", "manual"),
        to_instruction("Қазақстан туралы 3 қызықты факт айт", "1. Қазақстан — әлемдегі ең үлкен жерге байланысты мемлекет (2,7 млн км²).\n2. Байқоңыр ғарыш айлағы — адамзат тарихындағы алғашқы ғарыш айлағы.\n3. Қазақстанда 130-дан астам ұлт тұрады.", "manual"),
        to_instruction("Машиналық оқыту деген не?", "Машиналық оқыту (Machine Learning) — компьютерлерді мысалдардан үйренуге мүмкіндік беретін жасанды интеллект саласы. Модель деректерден заңдылықтарды тауып, болжам жасайды.", "manual"),
        to_instruction("Жасанды интеллект пен машиналық оқытудың айырмашылығы?", "Жасанды интеллект (AI) — кеңірек ұғым, машиналардың ақылды мінез-құлқы. Машиналық оқыту (ML) — AI-дың бір бөлімі, мысалдардан үйрену арқылы жұмыс істейді.", "manual"),
        to_instruction("Нейрондық желі қалай жұмыс істейді?", "Нейрондық желі — адам миын үлгілейтін алгоритм. Кіріс деректері қабаттар арқылы өтіп, салмақтар реттеледі. Жаттығу барысында қателер азайтылып, дәлдік артады.", "manual"),
        to_instruction("Алматы туралы мәлімет бер", "Алматы — Қазақстанның ең үлкен қаласы және мәдени-экономикалық орталығы. Іле Алатауының солтүстік баурайында орналасқан. Халқы 2 миллионнан астам. 1997 жылға дейін астана болған.", "manual"),
        to_instruction("LoRA fine-tuning дегеніміз не?", "LoRA (Low-Rank Adaptation) — үлкен модельдерді аз параметрмен дообучать ету әдісі. Барлық салмақтарды емес, арнайы қосымша матрицаларды ғана өзгертеді. Жылдам, арзан, тиімді.", "manual"),
        to_instruction("QLoRA мен LoRA айырмашылығы?", "QLoRA — квантизацияланған LoRA. Модель 4-бит форматта сақталады, бұл жадты 4 есе азайтады. 16GB RAM-да 7B модельді fine-tune жасауға мүмкіндік береді. LoRA-дан баяуырақ бірақ анағұрлым экономды.", "manual"),
    ]
    return save_jsonl(records * 5, "manual_seeds")  # x5 для весомости

# ══════════════════════════════════════════════════════════════════
# 4. ФИНАЛЬНАЯ СБОРКА
# ══════════════════════════════════════════════════════════════════

def merge_and_finalize(all_records):
    log(f"\n🔧 Финальная обработка — всего записей: {len(all_records):,}")

    # Фильтрация
    filtered = [r for r in all_records if is_quality(r.get("completion",""), min_len=20)]
    log(f"  После фильтрации: {len(filtered):,}")

    # Дедупликация
    deduped = deduplicate(filtered)
    log(f"  После дедупликации: {len(deduped):,}")

    # Перемешать
    random.shuffle(deduped)

    # Сохранить финальный файл
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"\n{'='*55}")
    log(f"✅ ГОТОВО: {OUT_FILE}")
    log(f"{'='*55}")
    log(f"ИТОГО ПРИМЕРОВ: {len(deduped):,}")
    log(f"\nПо источникам:")
    for name, cnt in sorted(STATS.items(), key=lambda x: -x[1]):
        bar = '█' * min(30, cnt // max(1, max(STATS.values()) // 30))
        log(f"  {name:<30} {cnt:>7,}  {bar}")
    log(f"{'='*55}")
    return deduped

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=999999, help="Лимит на HF источник")
    p.add_argument("--skip-scrape", action="store_true", help="Пропустить веб-скрапинг")
    p.add_argument("--skip-hf", action="store_true", help="Пропустить HuggingFace (только scrape)")
    p.add_argument("--fast", action="store_true", help="Быстрый режим (лимит 5000 на источник)")
    return p.parse_args()

def main():
    args = parse_args()
    limit = 5000 if args.fast else args.limit

    log("=" * 55)
    log("  KazGPT — Максимальный сборщик датасетов")
    log("=" * 55)

    all_records = []

    if not args.skip_hf:
        all_records += load_kazqad(limit)
        all_records += load_wikipedia_kk(limit)
        all_records += load_mc4_kk(limit)
        all_records += load_culturax_kk(limit)
        all_records += load_oscar_kk(limit)
        all_records += load_multidomain_kk(limit)

    if not args.skip_scrape:
        all_records += scrape_adilet(5000)
        all_records += scrape_kazakh_news(10000)

    all_records += add_manual_seeds()

    merge_and_finalize(all_records)

if __name__ == "__main__":
    main()
