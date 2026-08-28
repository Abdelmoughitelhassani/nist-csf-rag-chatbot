"""Schémas Pydantic pour l'answer contract (Principe 1 + Principe 3 du MD).

La réponse du LLM doit toujours inclure : la valeur, le span de preuve
(passage exact du document), et deux booléens (answer_found,
complete_answer_found) plutôt que d'inventer une réponse quand le document
ne la contient pas.

Les champs pending_references / answer_completeness (Principe 3, résolution
des renvois internes) sont présents dès cette version du contrat, même si
la boucle de résolution en deux passes n'est pas encore implémentée
(prochaine étape après ce notebook).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Span(BaseModel):
    page: int
    text: str  # passage exact du document


class PendingReference(BaseModel):
    raw_text: str  # "Section 5.2", "Table 3 row (E)"
    ref_type: str  # "section" | "table" | "figure"
    origin_page: int


class ReferenceAwareAnswer(BaseModel):
    value: Any = Field(description="La réponse à la question. null si answer_found=False.")
    evidence: list[Span] = Field(default_factory=list, description="Passages exacts du document qui justifient la réponse, avec leur numéro de page.")
    answer_found: bool = Field(description="False si le document ne contient pas de réponse à la question.")
    complete_answer_found: bool = Field(description="False si la réponse trouvée est partielle (ex: renvoi non résolu).")
    confidence: float = Field(ge=0, le=1)
    answer_completeness: Literal["complete", "references_unresolved", "partial"]
    pending_references: list[PendingReference] = Field(default_factory=list)
