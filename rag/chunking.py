"""Chunking contextuel : par subcategory (validé dans le notebook 01) et,
nouveau, par category.

Les chunks de catégorie permettent de répondre à des questions portant sur
une catégorie entière (ex: "What does the Information Protection Processes
category cover?"), que les chunks de subcategory seuls -- trop fins, un
seul par pratique -- ne couvrent pas bien individuellement.

Le chunk de catégorie contient volontairement une phrase de renvoi
explicite vers ses subcategories ("voir PR.IP-1, PR.IP-2, ...") : c'est un
renvoi interne réel (pas fabriqué), qui sert de cas de test pour la boucle
de résolution en deux passes (rag/orchestrator.py, Principe 3 du MD).
"""

from __future__ import annotations

from collections import defaultdict


def build_contextual_chunks(sections: list[dict]) -> list[dict]:
    """Un chunk par subcategory, avec le contexte parent en en-tête."""
    chunks = []
    for section in sections:
        contextual_text = f"""Document : NIST Cybersecurity Framework v1.1
Fonction : {section['function']}
Categorie : {section['category']}
Sous-section : {section['subcategory_id']}

{section['content']}""".strip()

        chunks.append({
            "text": contextual_text,
            "page": section["page"],
            "level": "subcategory",
            "subcategory_id": section["subcategory_id"],
            "category_code": section["subcategory_id"].split("-")[0],
            "parent_category": section["category"],
            "parent_function": section["function"],
        })
    return chunks


def build_category_chunks(sections: list[dict], category_info: dict[str, dict]) -> list[dict]:
    """Un chunk par category : description + renvoi explicite vers ses subcategories."""
    subcats_by_category: dict[str, list[str]] = defaultdict(list)
    for s in sections:
        code = s["subcategory_id"].split("-")[0]
        subcats_by_category[code].append(s["subcategory_id"])

    chunks = []
    for code, info in category_info.items():
        subcat_ids = sorted(
            subcats_by_category.get(code, []),
            key=lambda sid: int(sid.split("-")[1]),
        )
        ref_sentence = ""
        if subcat_ids:
            ref_sentence = (
                f"\n\nCette categorie regroupe {len(subcat_ids)} sous-categories, "
                f"chacune decrivant une pratique de securite specifique : voir "
                f"{', '.join(subcat_ids)} pour le detail complet."
            )

        text = f"""Document : NIST Cybersecurity Framework v1.1
Fonction : {info['function']}
Categorie : {info['category']} ({code})

{info['description']}{ref_sentence}""".strip()

        chunks.append({
            "text": text,
            "page": info["page"],
            "level": "category",
            "category_code": code,
            "parent_category": info["category"],
            "parent_function": info["function"],
            "subcategory_ids": ",".join(subcat_ids),  # metadata Chroma = scalaire uniquement
        })
    return chunks
