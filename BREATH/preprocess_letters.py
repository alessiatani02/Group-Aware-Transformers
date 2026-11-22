

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

DEFAULT_TOKENIZER = 'Musixmatch/umberto-commoncrawl-cased-v1'

# ---------------------------------------------------------------------------
# Scrub 
# ---------------------------------------------------------------------------

_DATE = re.compile(r'\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b')
_TIME = re.compile(r'(?<![\d:])(?:ore\s*)?\d{1,2}:\d{2}(?::\d{2})?(?!\d)|\bore\s*\d{1,2}[.,]\d{2}\b', re.I)
_SPACED_DIGITS = re.compile(r'(?:\b\d\s){5,}\d\b')
_LONG_ID = re.compile(r'\b[A-Za-z]{0,4}[/_-]?\d{5,}[A-Za-z0-9/_-]*\b')
_EPISODE_ID = re.compile(r'\b[A-Z]{1,4}/\d{2}/\d+\b')
_URL_MAIL = re.compile(r'\S+@\S+|(?:https?://|www\.)\S+', re.I)
_BIRTH = re.compile(r'\bnat[oa](?:/[ao])?\s+il\b:?|\bnat[oa]\s+a\s*:', re.I)
_DOCTOR = re.compile(
    r"\b(?i:dott\.?\s?ssa|dottssa|dott\.?|dr\.?\s?ssa|dr\.?|prof\.?\s?ssa|prof\.?|sig\.?\s?ra|sig\.?)\s+"
    r"[A-ZÀ-Ý][\w'’]+(?:\s+[A-ZÀ-Ý][\w'’]+){0,2}"
)
_PHONE = re.compile(r'\b(?:tel|fax|cell)\b\.?\s*(?:\(?\+?\d[\d\s./()-]*)?', re.I)


_TRAILING_NAME = re.compile(r'(?<=[a-z.,;)])\s+(?:[A-ZÀ-Ý]{3,}\s+){1,2}[A-ZÀ-Ý]{3,}\s*$')


_HAS_DIGIT = re.compile(r'\d').search
_SPACES = re.compile(r'\s+')


def scrub(line):
    low = line.lower()
    if '@' in line or 'www' in low or 'http' in low:
        line = _URL_MAIL.sub(' ', line)
    if _HAS_DIGIT(line):
        for pattern in (_DATE, _TIME, _SPACED_DIGITS, _EPISODE_ID, _LONG_ID):
            line = pattern.sub(' ', line)
    if 'nat' in low:
        line = _BIRTH.sub(' ', line)
    if 'dr' in low or 'dott' in low or 'prof' in low or 'sig' in low:
        line = _DOCTOR.sub(' ', line)
    if 'tel' in low or 'fax' in low or 'cell' in low:
        line = _PHONE.sub(' ', line)
    line = _TRAILING_NAME.sub('', line)
    return _SPACES.sub(' ', line).strip(' \t-_,;|')


# ---------------------------------------------------------------------------
# Header / footer / righe amministrative
# ---------------------------------------------------------------------------

# Primo contenuto clinico: tutto cio' che lo precede e' header.
_ANCHOR = re.compile(
    r'\b(?:motivo|anamnesi|anamnestic\w*|sintomatologia|esame obiettivo|giunge|giunta|giunto|'
    r'accede|riferisc\w*|riferit\w*|diagnosi|referto|problema specifico|psm|storia)\b',
    re.I,
)

# Etichette amministrative ("Cognome e Nome: ...", "Data/Ora di Ammissione: ..."):
# valgono solo con i due punti entro 3 parole o se sono sole sulla riga, cosi'
# non colpiscono frasi come "Paziente di 3 anni ...".
_ADMIN_LABEL = re.compile(
    r"""^\W*(?:
        cognome(?:\ e\ nome)?|nome|sesso|data\ di\ nascita|luogo\ di\ nascita|comune|provincia|prov|
        cittadinanza|nazionalit[àa]|indirizzo|domicilio|residenza|via|tel|telefono|fax|e-?mail|mail|cap|
        codice\ fiscale|cod\.?\ fiscale|c\.?\ ?f\.?|tess(?:era)?\.?\ ?san\w*\.?|cod\.?\ ?sanit\.?|id|
        id\ documento|id\ paz\.?/c\.?f\.?|identificativo\ \w+|episodio(?:\ n\.?)?|numero\ verbale|
        n\.?\ ?accettazione|n°\ ?accettazione|codice\ accesso|assistito(?:/a)?|paziente|dettagli\ paziente|
        data(?:/ora|\ e\ ora|\ ora)?|stampat[oa]|cod(?:ice)?\ operatore|operatore(?:\ al\ triage)?|
        modalit[àa]\ di\ \w+|mezzo\ di\ trasporto|provenienza|parte\ riservata\ \w+|prestazioni\ erogate|
        ripresentarsi\ il|medico\ accettante|infermiere\ di\ triage
    )\b(?:(?:\s+[\w./°]+){0,3}\s*:|\s*$)""",
    re.I | re.X,
)

# Righe che iniziano con questi termini sono sempre amministrative; "apertura" e
# "chiusura" lo sono solo se restano da sole dopo la rimozione di data e ora.
_ADMIN_START = re.compile(
    r"^\W*(?:segreteria|direttore|responsabile|firmat\w+|firma\b|dimesso dal p\.?s\.? il|"
    r"referto redatto il)|^\W*(?:apertura|chiusura)\W*$",
    re.I,
)

# Footer legali / di sistema: bastano in qualunque punto della riga.
_BOILERPLATE = re.compile(
    r"""firmat|conservat[oa]\ (?:in\ formato\ )?digital|normativa\ vigente|ai\ sensi\ d|
        rappresentazione\ di\ un|certificato\ numero|numero\ di\ certificato|actalis|
        stampat[oa]\ il|stampa\ in\ originale|\bpagina\b|\bpag\.?\s*\d|desideriamo\ render|
        servizio\ sanitario\ regionale|\bticket\b|quota\ fissa|totale\ impegnato|
        raccomandiamo\ di\ esibire|presentat[ao]\ (?:al|il)\ pi[uù]\ presto|orario\ di\ alcune\ prestazioni|
        presente\ verbale\ costituisce|consegnare\ al\ proprio\ pediatra|\binformazione\b|
        gentile\ signor|trattamento\ dei\ dati|\bcodice\ colore\b|\binfermier[ei]\b|
        richiesta\ di\ pagamento|casse\ automatiche|\bcup\b|riscuotitori|spese\ di\ gestione""",
    re.I | re.X,
)

# Intestazioni di struttura nelle prime righe, se brevi e senza contenuto clinico.
_HEADER_LINE = re.compile(
    r"\b(?:regione|azienda|ulss|ospedale|presidio ospedaliero|dipartimento|struttura complessa|"
    r"area assistenziale|pronto soccorso|verbale|modulo|relazione sanitaria|unit[aà] operativa)\b|"
    r"\b(?:u\.o\.[cs]?\.?|p\.o\.|u\.l\.s\.s\.?|dip\.)(?=\s|$)",
    re.I,
)

# Campi di modulo senza valore ("Tipo invio:", "Temper.: Sat: Glicemia:"), tranne le
# etichette di sezione il cui valore sta sulla riga dopo ("Diagnosi:" / "virosi").
_EMPTY_FORM = re.compile(r'^(?:[^:\d]{1,40}:\s*[/\-]*\s*)+$')
_SECTION_LABEL = re.compile(
    r'^\W*(?:diagnosi|impress\w*|esito|terapia|anamnesi|osservazion\w*|conclusioni|prescrizion\w*|'
    r'consigli\w*|referto|motivo\w*|note|decorso)\b[^:]*:?\s*$',
    re.I,
)

# Firma: riga breve "Il Medico ..." seguita spesso da una riga con solo il nome.
_SIGNATURE = re.compile(r"^\W*(?:[A-Za-zÀ-ÿ' ]{3,20},\s*)?(?:il|la)?\s*(?:medico|dottore|dottoressa)\b", re.I)
_NAME_ONLY = re.compile(r"^[A-ZÀ-Ý][A-Za-zÀ-ÿ'’]+(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’.]+){0,3}$")

DIAG_RE = re.compile(r'\b(?:diagnosi|impress(?:ione)?|conclusioni)\b', re.I)
_DIAG_LABEL = re.compile(
    r'^\W*(?:codice\s+)?(?:descrizione\s+)?(?:diagnosi|impress\w*|conclusioni)\b[^:]{0,45}:?\s*$', re.I)


def split_lines(text):
    text = text.replace('\r', '\n').replace('\t', ' ')
    return [re.sub(r'[  ]+', ' ', l).strip() for l in text.split('\n') if l.strip()]


def cut_header(lines):
    """Elimina le righe prima del primo contenuto clinico, se compare presto."""
    limit = min(40, max(8, len(lines) // 2))
    for i, line in enumerate(lines[:limit]):
        if _ANCHOR.search(line):
            return lines[i:]
    return lines


_HAS_WORD = re.compile(r'[A-Za-zÀ-ÿ]{2}').search


def is_admin(line):
    if not _HAS_WORD(line):
        return True
    if _EMPTY_FORM.match(line) and not _SECTION_LABEL.match(line):
        return True
    return bool(_ADMIN_START.match(line) or _ADMIN_LABEL.match(line) or _BOILERPLATE.search(line))


def is_header_line(line):
    """Intestazione di struttura: breve, senza ancore cliniche e senza parole
    minuscole di contenuto (tutta maiuscola o Title Case)."""
    if not _HEADER_LINE.search(line) or _ANCHOR.search(line) or len(line.split()) > 12:
        return False
    return not any(len(w) > 3 and w[0].islower() for w in line.split())


def mask(line):
    return re.sub(r'\s+', ' ', re.sub(r'\d', '0', line.lower())).strip(' .:;-')


def clean_lines(text, templates=frozenset()):
    """Righe pulite, senza header, footer, righe amministrative e duplicati."""
    lines = cut_header(split_lines(text))
    out, seen, skip_name, after_diag, dropped = [], set(), False, False, False
    for pos, raw in enumerate(lines):
        line = scrub(raw)
        if not line:
            continue
        value_of_diagnosis, after_diag = after_diag, bool(_DIAG_LABEL.match(line))
        # una riga in minuscolo che continua una riga appena scartata va via con lei
        if dropped and line[0].islower():
            continue
        dropped = True
        if pos < 15 and is_header_line(line):
            continue
        if skip_name:
            skip_name = False
            if len(line.split()) <= 4 and _NAME_ONLY.match(line):
                continue
        if _SIGNATURE.match(line) and len(line.split()) <= 8:
            skip_name = True
            continue
        if is_admin(line):
            continue
        key = mask(line)
        if key in seen or (key in templates and not DIAG_RE.search(line) and not value_of_diagnosis):
            continue
        seen.add(key)
        out.append(line)
        dropped = False
    return out


def reflow(lines):
    """Riunisce le righe spezzate dal layout PDF e le coppie 'Etichetta:' / valore."""
    paras = []
    for line in lines:
        prev = paras[-1] if paras else ''
        is_label = len(prev.split()) <= 3 and (prev.endswith(':') or _SECTION_LABEL.match(prev))
        if prev and (line[0].islower() or is_label):
            paras[-1] = f'{prev} {line}'
        else:
            paras.append(line)
    return paras


# ---------------------------------------------------------------------------
# Template di header/footer appresi dal dataset
# ---------------------------------------------------------------------------

def learn_templates(texts, min_count, edge_frac=0.6, dup_frac=0.3, edge=0.2, min_lines=25,
                    neighbor_frac=0.7, max_words=12):
    """Righe (con cifre mascherate) che fanno parte di header, footer o etichette
    di modulo, riconosciute dal dataset con tre criteri:
      - stanno ai margini del documento (solo nelle lettere con almeno `min_lines`
        righe: in quelle corte ogni riga e' "ai margini");
      - si ripetono nello stesso documento (header/footer di pagina);
      - sono brevi, frequenti e compaiono quasi sempre con la stessa riga prima o
        dopo (le etichette di un modulo hanno sempre la stessa sequenza, le frasi
        cliniche no).
    Le righe di diagnosi e di esito (ricovero, dimissione a domicilio...) non
    vengono mai considerate template."""
    docs = []
    df, df_long, at_edge, dup = Counter(), Counter(), Counter(), Counter()
    for t in texts:
        lines = [scrub(l) for l in split_lines(t)]
        keys = [mask(l) for l in lines if l]
        docs.append(keys)
        n = len(keys)
        per_doc = Counter(keys)
        edge_keys = {k for i, k in enumerate(keys) if i < n * edge or i >= n * (1 - edge)}
        for k in per_doc:
            df[k] += 1
            dup[k] += per_doc[k] > 1
            if n >= min_lines:
                df_long[k] += 1
                at_edge[k] += k in edge_keys

    frequent = {k for k, c in df.items() if c >= min_count}
    neighbors = {k: (Counter(), Counter()) for k in frequent}
    for keys in docs:
        for i, k in enumerate(keys):
            if k in frequent:
                neighbors[k][0][keys[i - 1] if i else '<start>'] += 1
                neighbors[k][1][keys[i + 1] if i + 1 < len(keys) else '<end>'] += 1

    templates = set()
    for k in frequent:
        if len(k) <= 3 or DIAG_RE.search(k) or _DISPOSITION.search(k):
            continue
        c = df[k]
        by_dup = dup[k] / c >= dup_frac
        by_edge = df_long[k] >= min_count and at_edge[k] / df_long[k] >= edge_frac
        prev_c, next_c = neighbors[k]
        follows_diagnosis = bool(DIAG_RE.search(prev_c.most_common(1)[0][0]))
        by_neighbor = not follows_diagnosis and len(k.split()) <= max_words and max(
            max(prev_c.values()) / sum(prev_c.values()),
            max(next_c.values()) / sum(next_c.values()),
        ) >= neighbor_frac
        if by_dup or by_edge or by_neighbor:
            templates.add(k)
    return frozenset(templates)


# ---------------------------------------------------------------------------
# Budget di token
# ---------------------------------------------------------------------------

def load_tokenizer(name_or_path):
    path = Path(name_or_path)
    if path.is_file():
        return Tokenizer.from_file(str(path))
    from huggingface_hub import hf_hub_download
    try:
        file = hf_hub_download(name_or_path, 'tokenizer.json', local_files_only=True)
    except Exception:
        file = hf_hub_download(name_or_path, 'tokenizer.json')
    return Tokenizer.from_file(file)


class TokenCounter:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.n_special = len(tokenizer.encode('').ids)

    def count(self, texts):
        if not texts:
            return []
        return [len(e.ids) for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def total(self, text):
        return len(self.tok.encode(text).ids)


def split_sentences(text):
    return [s for s in re.split(r'(?<=[.!?;])\s+(?=[A-ZÀ-Ý0-9])', text) if s]


def fit_sentences(text, budget, counter, from_end):
    """Frasi di `text` (dalla fine o dall'inizio) che stanno in `budget` token."""
    sents = split_sentences(text)
    counts = counter.count(sents)
    order = range(len(sents) - 1, -1, -1) if from_end else range(len(sents))
    kept, used = [], 0
    for i in order:
        if used + counts[i] > budget:
            break
        kept.append(i)
        used += counts[i]
    kept.sort()
    return ' '.join(sents[i] for i in kept)


def diagnosis_groups(paras):
    """Indici dei paragrafi con 'diagnosi', seguiti dalle righe successive quando
    il paragrafo e' solo un'etichetta (es. "Diagnosi" / "Codice Descrizione" / "493.82 Asma")."""
    groups, i = [], 0
    while i < len(paras):
        if not DIAG_RE.search(paras[i]):
            i += 1
            continue
        group = [i]
        while _DIAG_LABEL.match(paras[group[-1]]) and group[-1] + 1 < len(paras):
            group.append(group[-1] + 1)
        groups.append(group)
        i = group[-1] + 1
    return groups


_DISPOSITION = re.compile(r'esito|ricover|trasferit|a domicilio|rinvio|dimess[oa]', re.I)
_KEEP_SHORT = re.compile(r'esito|ricover|dimission|domicilio|trasferit|prognosi', re.I)


def is_low_value(par):
    """Righe di 4 parole o meno (valori di tabelle, etichette): usate solo se avanza spazio."""
    return len(par.split()) <= 4 and not DIAG_RE.search(par) and not _KEEP_SHORT.search(par)


def select_paragraphs(paras, limit, tail_fraction, counter, drop_diagnosis):
    """Sceglie (con priorita' diagnosi > coda > testa > resto) i paragrafi che
    stanno in `limit` token, preservando l'ordine originale."""
    if drop_diagnosis:
        paras = [p for p in paras if not DIAG_RE.search(p)]
    counts = counter.count(paras)
    if sum(counts) <= limit:
        return paras

    chosen, left = {}, limit

    def take(i, budget, from_end):
        nonlocal left
        if counts[i] <= budget:
            chosen[i], cost = paras[i], counts[i]
        else:
            part = fit_sentences(paras[i], budget, counter, from_end)
            if not part:
                return False
            chosen[i], cost = part, sum(counter.count(split_sentences(part)))
        left -= cost
        return counts[i] <= budget

    # Come l'estrattore della repo, l'ultima diagnosi e' la piu' rilevante; il tetto
    # complessivo evita che molte menzioni di "diagnosi" esauriscano il budget.
    diag_left = int(limit * 0.4)
    for group in reversed(diagnosis_groups(paras)):
        for i in group:
            before = left
            take(i, min(diag_left, 100), False)
            diag_left -= before - left

    low = [is_low_value(p) for p in paras]

    tail_left = min(left, int(limit * tail_fraction))
    for i in range(len(paras) - 1, -1, -1):
        if i in chosen or low[i]:
            continue
        before = left
        if not take(i, tail_left, True):
            tail_left = 0
            break
        tail_left -= before - left

    for i in range(len(paras)):
        if i in chosen or low[i]:
            continue
        if not take(i, left, False):
            break

    for i in range(len(paras)):
        if i not in chosen and counts[i] <= left:
            chosen[i] = paras[i]
            left -= counts[i]

    return [chosen[i] for i in sorted(chosen)]


def preprocess_letter(text, counter, max_tokens, tail_fraction, drop_diagnosis, templates):
    if pd.isna(text):
        return None
    paras = reflow(clean_lines(text, templates))
    limit = max_tokens - counter.n_special
    for _ in range(3):
        out = '\n'.join(select_paragraphs(paras, limit, tail_fraction, counter, drop_diagnosis))
        over = counter.total(out) - max_tokens
        if over <= 0:
            return out
        limit -= over + 2
    enc = counter.tok.encode(out)
    return out[:enc.offsets[max_tokens - counter.n_special - 1][1]]


# ---------------------------------------------------------------------------
# Pipeline e I/O
# ---------------------------------------------------------------------------

def add_preprocessed(df, text_col, output_col, tokenizer, max_tokens=512, tail_fraction=0.5,
                     drop_diagnosis=False, detect_templates=True, template_min_count=None):
    if text_col not in df.columns:
        raise KeyError(f"Colonna '{text_col}' non trovata. Colonne disponibili: {list(df.columns)}")

    counter = TokenCounter(tokenizer)
    texts = df[text_col].tolist()

    templates = frozenset()
    if detect_templates and len(texts) >= 200:
        sample = df[text_col].dropna().sample(min(15000, df[text_col].notna().sum()), random_state=0)
        min_count = template_min_count or max(30, int(0.002 * len(sample)))
        templates = learn_templates(sample.tolist(), min_count)
        print(f"Template di header/footer appresi dal dataset: {len(templates)} righe.")

    df = df.copy()
    df[output_col] = [
        preprocess_letter(t, counter, max_tokens, tail_fraction, drop_diagnosis, templates)
        for t in texts
    ]
    return df


EXCEL_EXTENSIONS = {'.xlsx', '.xls'}
CSV_EXTENSIONS = {'.csv'}


def read_dataset(path):
    ext = path.suffix.lower()
    if ext in EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    if ext in CSV_EXTENSIONS:
        return pd.read_csv(path)
    raise ValueError(f"Formato file non supportato: '{ext}'. Usa .csv, .xlsx o .xls.")


def write_dataset(df, path):
    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in EXCEL_EXTENSIONS:
        df.to_excel(path, index=False)
    elif ext in CSV_EXTENSIONS:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Formato file non supportato: '{ext}'. Usa .csv, .xlsx o .xls.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Riduce le lettere di dimissione a un massimo di N token, togliendo header e footer."
    )
    parser.add_argument('input_file', type=Path, help="File di input (.csv, .xlsx, .xls)")
    parser.add_argument('text_column', help="Nome della colonna con il testo della lettera")
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help="File di output (.csv, .xlsx, .xls). Default: <input>_512.<ext>")
    parser.add_argument('--output-col', default='testo_512', help="Nome della colonna di output")
    parser.add_argument('--max-tokens', type=int, default=512, help="Massimo di token, speciali inclusi")
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER,
                        help="Repo HF o percorso di un tokenizer.json (default: UmBERTo)")
    parser.add_argument('--tail-fraction', type=float, default=0.5,
                        help="Quota del budget riservata alla parte finale della lettera")
    parser.add_argument('--drop-diagnosis', action='store_true',
                        help="Rimuove anche le righe con 'diagnosi'/'impressione'/'conclusioni'")
    parser.add_argument('--no-template-detection', action='store_true',
                        help="Disattiva l'apprendimento dei template di header/footer dal dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_file.exists():
        raise FileNotFoundError(f"File non trovato: {args.input_file}")

    output_path = args.output or args.input_file.with_name(
        f"{args.input_file.stem}_{args.max_tokens}{args.input_file.suffix}")

    print(f"Lettura di: {args.input_file}")
    df = read_dataset(args.input_file)
    print(f"{len(df)} righe caricate.")

    tokenizer = load_tokenizer(args.tokenizer)
    df = add_preprocessed(
        df, args.text_column, args.output_col, tokenizer,
        max_tokens=args.max_tokens, tail_fraction=args.tail_fraction,
        drop_diagnosis=args.drop_diagnosis, detect_templates=not args.no_template_detection,
    )

    counter = TokenCounter(tokenizer)
    n_tok = pd.Series([counter.total(t) if isinstance(t, str) else 0 for t in df[args.output_col]])
    print(f"Token per lettera dopo il pre-processing: mediana {n_tok.median():.0f}, max {n_tok.max()}.")

    write_dataset(df, output_path)
    print(f"Output salvato in: {output_path}")


if __name__ == '__main__':
    main()
