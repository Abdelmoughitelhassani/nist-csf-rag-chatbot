"""Parsing structuré du PDF NIST Cybersecurity Framework v1.1.

Extrait la hiérarchie Function > Category > Subcategory du "Table 2:
Framework Core" (le PDF n'a pas de table des matières exploitable via
PyMuPDF). Logique validée dans notebooks/01_chunking_chromadb_test.ipynb,
étendue ici avec l'extraction des DESCRIPTIONS de catégorie (le paragraphe
d'intro avant la liste de ses subcategories), nécessaire pour construire
des chunks au niveau catégorie (rag/chunking.py).
"""

from __future__ import annotations

import re

FUNCTION_NAMES = {
    "ID": "IDENTIFY",
    "PR": "PROTECT",
    "DE": "DETECT",
    "RS": "RESPOND",
    "RC": "RECOVER",
}

SUBCAT_ID_RE = re.compile(r"\b(ID|PR|DE|RS|RC)\.([A-Z]{2})-(\d+)\b")

# pas de chiffres dans la classe de caracteres : ca evite de "manger" les
# codes de references informatives (ex. "CP-2, PS-7, PM-11") qui precedent
# parfois le vrai header de categorie dans le flux de texte aplati
CATEGORY_HEADER_RE = re.compile(
    r"([A-Z][A-Za-z ,/\-'’\n]{2,100}?)\s*\(((?:ID|PR|DE|RS|RC)\.[A-Z]{2})\):"
)

REFERENCE_PREFIXES = (
    "CIS CSC", "COBIT 5", "ISA 62443", "ISO/IEC", "NIST SP",
    "NIST Privacy Framework", "NIST Cybersecurity Framework",
)

# en-tete + pied de page + repetition du header du tableau, presents en
# debut de chaque page du document
PAGE_BOILERPLATE_RE = re.compile(
    r"^April 16, 2018\s+Cybersecurity Framework\s+Version 1\.1\s+"
    r"This publication is available free of charge from:\s*\S+\s+"
    r"\d+\s+"
    r"(?:Function\s+Category\s+Subcategory\s+Informative References\s+)?"
)


def strip_page_boilerplate(text: str) -> str:
    return PAGE_BOILERPLATE_RE.sub("", text)


def find_core_table_pages(doc) -> list[int]:
    """Ne garde que le plus long bloc de pages CONSECUTIVES contenant des
    subcategory IDs (les mentions isolees ailleurs dans le document sont
    ainsi ecartees)."""
    pages_with_subcats = [i for i in range(doc.page_count) if SUBCAT_ID_RE.search(doc[i].get_text())]
    if not pages_with_subcats:
        return []

    runs, current = [], [pages_with_subcats[0]]
    for p in pages_with_subcats[1:]:
        if p == current[-1] + 1:
            current.append(p)
        else:
            runs.append(current)
            current = [p]
    runs.append(current)
    return max(runs, key=len)


def build_page_text_index(doc, pages: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Concatene le texte des pages ciblees (apres nettoyage du
    header/footer repete), en gardant un mapping
    offset_de_caractere -> numero_de_page (1-indexed)."""
    full_text = ""
    offsets = []
    for p in pages:
        offsets.append((len(full_text), p + 1))
        full_text += strip_page_boilerplate(doc[p].get_text())
    return full_text, offsets


def offset_to_page(offset: int, offsets: list[tuple[int, int]]) -> int:
    page = offsets[0][1]
    for start, p in offsets:
        if offset >= start:
            page = p
        else:
            break
    return page


def clean_text_block(raw: str) -> str:
    """Retire les lignes de references informatives collees a la
    description/subcategory dans le flux de texte aplati par PyMuPDF, et
    le ':' residuel apres un ID de subcategory (ex 'ID.AM-1:')."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    kept = []
    for line in lines:
        if line.startswith(REFERENCE_PREFIXES):
            break
        kept.append(line)
    return " ".join(kept).strip().lstrip(":").strip()


def parse_nist_csf_structure(doc) -> tuple[list[dict], dict[str, dict]]:
    """Retourne (sections, category_info).

    sections : une entree par subcategory -- function, category,
    subcategory_id, content, page.

    category_info : category_code ("ID.AM") -> {function, category,
    category_code, description, page}. La description est le paragraphe
    entre le header "Nom (CODE):" et la premiere subcategory qui suit.
    """
    pages = find_core_table_pages(doc)
    full_text, offsets = build_page_text_index(doc, pages)

    category_matches = list(CATEGORY_HEADER_RE.finditer(full_text))
    category_names = {}
    for m in category_matches:
        name, code = m.group(1), m.group(2)
        category_names[code] = re.sub(r"\s+", " ", name).strip()

    subcat_matches = list(SUBCAT_ID_RE.finditer(full_text))

    sections = []
    seen_ids = set()
    for idx, m in enumerate(subcat_matches):
        func_code, cat_letters, num = m.group(1), m.group(2), m.group(3)
        subcategory_id = f"{func_code}.{cat_letters}-{num}"
        if subcategory_id in seen_ids:
            continue
        seen_ids.add(subcategory_id)

        start = m.end()
        end = subcat_matches[idx + 1].start() if idx + 1 < len(subcat_matches) else len(full_text)
        content = clean_text_block(full_text[start:end])

        category_code = f"{func_code}.{cat_letters}"
        sections.append({
            "function": FUNCTION_NAMES.get(func_code, func_code),
            "category": category_names.get(category_code, category_code),
            "subcategory_id": subcategory_id,
            "content": content,
            "page": offset_to_page(m.start(), offsets),
        })

    category_info = {}
    for cm in category_matches:
        code = cm.group(2)
        func_code = code.split(".")[0]
        start = cm.end()
        end = next((sm.start() for sm in subcat_matches if sm.start() >= start), len(full_text))
        description = clean_text_block(full_text[start:end])
        category_info[code] = {
            "function": FUNCTION_NAMES.get(func_code, func_code),
            "category": category_names.get(code, code),
            "category_code": code,
            "description": description,
            "page": offset_to_page(cm.start(), offsets),
        }

    return sections, category_info
