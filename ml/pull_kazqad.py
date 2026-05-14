"""KazGPT V3 — production data preparation.

Заменяет старый prepare_data.py, который падал на гейтированный KazQAD и брал
случайные текстовые куски вместо Q&A пар.

Стратегия (multi-source):
  1. KazQAD от ISSAI NU (главный): требует `hf auth login`. Реальные Q&A на казахском.
  2. KazNERD: NER датасет — мы конвертируем sentence → "Что упоминается?" stub примеры.
  3. kk-Wikipedia: первые предложения → "X деген не?" автогенерация.
  4. Ручные seed примеры: высокого качества, под стиль KazGPT.

Формат output: ChatML-совместимый JSONL для train_cuda.py:
  {"messages": [
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}
  ]}

Использование:
  hf auth login   # один раз
  python ml/pull_kazqad.py --output ./data --kazqad --wiki --synthetic
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

random.seed(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./data")
    p.add_argument("--kazqad", action="store_true", help="скачать issai/kazqad (нужен hf auth)")
    p.add_argument("--kaznerd", action="store_true", help="скачать issai/kaznerd")
    p.add_argument("--wiki", action="store_true", help="скачать kk-Wikipedia dump")
    p.add_argument("--multidomain", action="store_true", help="фоллбек: kz-transformers/multidomain-kazakh-dataset")
    p.add_argument("--synthetic", action="store_true", help="добавить ручные seed примеры")
    p.add_argument("--max-per-source", type=int, default=10000)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--valid-ratio", type=float, default=0.10)
    p.add_argument("--min-prompt-tokens", type=int, default=3)
    p.add_argument("--min-completion-tokens", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2000)
    return p.parse_args()


def to_chatml(prompt: str, completion: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": completion.strip()},
        ]
    }


def load_kazqad(limit: int):
    """KazQAD: official Kazakh Q&A dataset. Поля: question, answers.text"""
    from datasets import load_dataset
    print("=> Loading issai/kazqad ...")
    try:
        ds = load_dataset("issai/kazqad", split="train")
    except Exception as e:
        print(f"   [SKIP] KazQAD failed: {e}")
        print("   → Запусти 'hf auth login' и проверь access к ISSAI.")
        return []

    records = []
    for item in ds:
        q = item.get("question") or item.get("instruction") or ""
        ans_field = item.get("answers") or item.get("answer") or item.get("output")
        if isinstance(ans_field, dict):
            texts = ans_field.get("text") or []
            a = texts[0] if texts else ""
        elif isinstance(ans_field, list) and ans_field:
            a = str(ans_field[0])
        else:
            a = str(ans_field or "")
        if q.strip() and a.strip():
            records.append(to_chatml(q, a))
        if len(records) >= limit:
            break
    print(f"   [+] {len(records)} examples from KazQAD")
    return records


def load_wiki_kk(limit: int):
    """kk-Wikipedia → автогенерация Q&A через first sentence."""
    from datasets import load_dataset
    print("=> Loading kk Wikipedia (wikimedia/wikipedia, 20231101.kk) ...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.kk", split="train", streaming=True)
    except Exception as e:
        print(f"   [SKIP] Wiki failed: {e}")
        return []

    records = []
    for item in ds:
        title = (item.get("title") or "").strip()
        text = (item.get("text") or "").strip()
        if not title or not text:
            continue
        # Первое предложение как «определение»
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) < 30 or len(first_sentence) > 600:
            continue

        # Генерируем несколько форм вопроса
        templates = [
            f"{title} деген не?",
            f"{title} туралы қысқаша айтып бер.",
            f"{title} жайында мәлімет бер.",
        ]
        q = random.choice(templates)
        records.append(to_chatml(q, first_sentence + "."))

        if len(records) >= limit:
            break
    print(f"   [+] {len(records)} examples from kk-Wikipedia")
    return records


def load_multidomain(limit: int):
    """kz-transformers/multidomain — случайные параграфы.
    НЕ для Q&A: используем как continuation training (prompt = первая половина,
    completion = вторая). Это работает плохо, но как фоллбек ОК."""
    from datasets import load_dataset
    print("=> Loading kz-transformers/multidomain-kazakh-dataset (fallback) ...")
    try:
        ds = load_dataset("kz-transformers/multidomain-kazakh-dataset",
                          split="train", streaming=True)
    except Exception as e:
        print(f"   [SKIP] multidomain failed: {e}")
        return []

    records = []
    for item in ds:
        text = (item.get("text") or item.get("content") or "").strip()
        if len(text) < 100:
            continue
        words = text.split()
        if len(words) < 30:
            continue
        half = len(words) // 2
        prompt = " ".join(words[:half])
        completion = " ".join(words[half:half + 100])
        records.append(to_chatml(
            f"Жалғастыр: {prompt}",
            completion,
        ))
        if len(records) >= limit:
            break
    print(f"   [+] {len(records)} examples from multidomain")
    return records


def synthetic_seed():
    """Ручные high-quality примеры под стиль KazGPT. Дублируются N раз для веса."""
    examples = [
        ("Сәлем!", "Сәлеметсіз! Мен KazGPT көмекшісімін. Сізге қалай көмектесе аламын?"),
        ("Қалың қалай?", "Жақсы, рахмет. Сіз қалайсыз? Не айтайын деп едіңіз?"),
        ("Привет, как дела?", "Сәлеметсіз! Бәрі жақсы, рахмет. Сіз қалайсыз?"),
        ("Hello, how are you?", "Сәлеметсіз! Жақсы, рахмет. Бірақ мен тек қазақша жауап беремін. Сізге не керек?"),
        ("Өзің туралы айтып бер",
         "Мен KazGPT — қазақ тілінде сөйлейтін жасанды интеллект көмекшісімін. "
         "Сұрақтарға жауап беремін, ақпаратпен бөлісемін және мәтінмен жұмыс істеуге көмектесемін."),
        ("Сен кімсің?", "Мен KazGPT, қазақ тіліндегі AI-көмекші. Сізге қалай көмектесе аламын?"),
        ("Қазақстанның астанасы қандай қала?",
         "Қазақстанның астанасы — Астана қаласы. 1997 жылы ол Алматыдан астана мәртебесін алды."),
        ("Алматы туралы қысқаша айтып бер",
         "Алматы — Қазақстанның оңтүстігіндегі ірі қала, Іле Алатауының баурайында орналасқан. "
         "1997 жылға дейін ел астанасы болған, қазір — мәдени, қаржылық және ғылыми орталық."),
        ("Кім Абай Құнанбайұлы?",
         "Абай Құнанбайұлы (1845–1904) — қазақтың ұлы ақыны, ағартушы, философ. "
         "«Қара сөздер» атты прозалық шығармасымен танымал."),
        ("Қазақстан туралы 3 қызықты факт айт",
         "1) Қазақстан — әлемдегі ең үлкен теңізге шықпайтын ел. "
         "2) Байқоңыр ғарыш айлағы Қазақстанда орналасқан. "
         "3) Қазақ тілінде 42 әріп бар (кирилица нұсқасы)."),
        ("Translate to Kazakh: I love my homeland.",
         "«Мен өз Отанымды сүйемін.»"),
        ("Переведи на казахский: Спасибо большое.",
         "«Үлкен рахмет.» немесе «Көп рахмет.»"),
        ("Бір қазақша мақал айт",
         "«Еңбек — бақыттың кілті.» Мағынасы: бақытқа жету үшін еңбек ету қажет."),
        ("Маған қазақша мақал-мәтел үш мысал бер",
         "1) «Еңбек — бақыттың кілті». "
         "2) «Білекті бірді жығады, білімді мыңды жығады». "
         "3) «Жеті рет өлшеп, бір рет кес»."),
        ("«Достық» сөзінің мағынасы қандай?",
         "«Достық» — адамдар арасындағы шынайы, жақын қарым-қатынас, бір-біріне сенім мен қолдау білдіру."),
        ("Нейрондық желі дегеніміз не?",
         "Нейрондық желі — адам миының жұмысын үлгілейтін математикалық модель. "
         "Ол көп санды деректерден заңдылықтар үйренеді."),
        ("Жасанды интеллект қалай жұмыс істейді?",
         "Жасанды интеллект үлкен деректерден заңдылықтарды үйренеді, содан кейін жаңа жағдайларда болжам жасайды немесе шешім қабылдайды."),
        ("Бүгінгі ауа райы туралы білесің бе?",
         "Кешіріңіз, мен нақты уақыттағы ауа райын білмеймін. Ауа райы сайттарын тексеріңіз."),
        ("Менің атым кім?",
         "Сіз әлі есіміңізді айтқан жоқсыз. Қалай жүгінгенімді қалайсыз?"),
        ("Әзіл айт",
         "Бір студент мұғаліміне: «Маған көмек керек, мен математиканы түсінбеймін» дейді. "
         "Мұғалімі: «Менде де сондай мәселе бар, бірақ маған еңбекақы төлейді» дейді."),
        ("Мына мәтінді қысқарт: «Бүгін мен таңертең тұрып, сыртқа шығып, дүкенге барып, нан, сүт және жұмыртқа сатып алдым.»",
         "Бүгін таңертең дүкенге барып, нан, сүт, жұмыртқа сатып алдым."),
        ("Сөйлемді өзгертіп жаз: «Бүгін ауа райы суық.»",
         "«Бүгінгі күн қатты салқын.» немесе «Бүгін күн жайсыз салқын.»"),
        ("Не білмейсің?", "Көп нәрсені білмеймін: нақты уақыттағы жаңалықтарды, жеке адамдар туралы деректерді, болашақты болжай алмаймын. Білмесем — «білмеймін» деймін."),
        ("Ағылшын тілінде жауап бере аласың ба?",
         "Кешіріңіз, мен тек қазақ тілінде жауап беремін. Бұл — менің мақсатым: қазақ тілін қолдау."),
        ("KazGPT кімдер жасады?",
         "KazGPT — IS-магистранттың жоба жұмысы. ISSAI зертханасының ашық деректері мен Qwen2.5 негізіндегі LoRA адаптерімен жасалды."),
    ]
    records = []
    for q, a in examples:
        # Дублируем по 5 раз для усиления стиля
        for _ in range(5):
            records.append(to_chatml(q, a))
    print(f"   [+] {len(records)} synthetic seed examples (each ×5)")
    return records


def dedup(records):
    """Удаляет дубликаты по (user_content, assistant_content) хэшу."""
    seen = set()
    out = []
    for r in records:
        msgs = r["messages"]
        u = msgs[0]["content"].strip().lower()
        a = msgs[1]["content"].strip().lower()
        h = hashlib.md5((u + "||" + a).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def filter_quality(records, min_p_tok: int, min_c_tok: int, max_tok: int):
    """Удаляет слишком короткие/длинные примеры."""
    out = []
    for r in records:
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        if len(u.split()) < min_p_tok or len(a.split()) < min_c_tok:
            continue
        if len(u.split()) > max_tok or len(a.split()) > max_tok:
            continue
        out.append(r)
    return out


def split_and_save(records, output_dir: Path, train_ratio: float, valid_ratio: float):
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    splits = {
        "train": records[:n_train],
        "valid": records[n_train:n_train + n_valid],
        "test": records[n_train + n_valid:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   {path.name}: {len(data)} examples")
    return splits


def main():
    args = parse_args()
    print("KazGPT V3 — Production data preparation")
    print("=" * 50)

    all_records = []
    if args.synthetic:
        all_records.extend(synthetic_seed())
    if args.kazqad:
        all_records.extend(load_kazqad(args.max_per_source))
    if args.wiki:
        all_records.extend(load_wiki_kk(args.max_per_source))
    if args.multidomain:
        all_records.extend(load_multidomain(args.max_per_source))

    if not all_records:
        print("[FATAL] No sources enabled. Use --kazqad --wiki --synthetic")
        sys.exit(1)

    print(f"\n=> Pre-filter: {len(all_records)} records")
    all_records = filter_quality(all_records, args.min_prompt_tokens, args.min_completion_tokens, args.max_tokens)
    print(f"=> After quality filter: {len(all_records)}")
    all_records = dedup(all_records)
    print(f"=> After dedup: {len(all_records)}")

    splits = split_and_save(all_records, Path(args.output), args.train_ratio, args.valid_ratio)

    print("\n=> Sample (3 random from train):")
    for r in random.sample(splits["train"], min(3, len(splits["train"]))):
        u = r["messages"][0]["content"][:80]
        a = r["messages"][1]["content"][:80]
        print(f"   USER: {u}{'...' if len(u) == 80 else ''}")
        print(f"   ASST: {a}{'...' if len(a) == 80 else ''}")
        print()

    print("=> Готово. Следующий шаг:")
    print(f"   python ml/train_cuda.py --data {args.output} --output ./adapters_v3")


if __name__ == "__main__":
    main()
