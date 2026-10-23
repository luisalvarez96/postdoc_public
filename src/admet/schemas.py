"""Pydantic schemas for the ADMET annotation step.

Design (see README.md): the final score is computed in Python from deterministic features plus a small table of
points (scoring_points.csv). The LLM only fills this annotation record: short human-readable fields, a coarse safety
tier and evidence-backed annotation flags. It never sees ModelAvg and never produces a score.

The Field descriptions below are sent to the model by `instructor` together with the system prompt, so they are part
of the prompt: keep them consistent with prompts/annotate_system.txt.
"""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

NOT_REPORTED = "Not reported"
_EMPTY_LIKE = {"", "unknown", "n/a", "na", "none", "not available", "not specified", "not reported"}


def _norm_text(v: str) -> str:
    v = (v or "").strip()
    return NOT_REPORTED if v.lower() in _EMPTY_LIKE else v


class EvidencedFlag(BaseModel):
    """A yes/no annotation that must be backed by a quote from the input."""

    known: bool = Field(
        default=False,
        description=(
            "True ONLY if the input text states that the drug acts on, or is used for, this tissue/function. "
            "False if the text is silent, or only mentions it as an adverse effect or as a disease the drug "
            "prevents (e.g. a vaccine)."
        ),
    )
    evidence: str = Field(
        default="",
        description="Verbatim quote (at most 30 words) from the input supporting `known`. Empty if `known` is false.",
    )

    @model_validator(mode="after")
    def _evidence_required(self):
        # An unsupported "True" is treated as False rather than trusted.
        self.evidence = (self.evidence or "").strip()
        if self.known and not self.evidence:
            self.known = False
        if not self.known:
            self.evidence = ""
        return self


class DrugAnnotation(BaseModel):
    """Output of the single LLM call per drug."""

    drug_class: str = Field(
        description=(
            "Short phrase (at most 12 words) naming the pharmacological or therapeutic class, using only the input. "
            f"'{NOT_REPORTED}' if the input does not support one."
        )
    )
    mechanism_of_action: str = Field(
        description=(
            "1-2 sentences: targets and what the drug does to them (inhibitor, agonist, ...), using only the input. "
            f"'{NOT_REPORTED}' if absent."
        )
    )
    toxicity_summary: str = Field(
        description=(
            "If the input gives numeric toxicity values (LD50, TD50, NOAEL, ...), quote them VERBATIM with species, "
            "route and units. Otherwise one sentence summarising the toxicity information. "
            f"'{NOT_REPORTED}' if the input has none."
        )
    )
    potency_summary: str = Field(
        description=(
            "If the input gives numeric potency values (ED50, IC50, EC50, Ki, ...), quote them VERBATIM with units "
            "and target/system. Otherwise one sentence on potency or effective dose if the input states it. "
            f"'{NOT_REPORTED}' if absent."
        )
    )
    safety_tier: Literal["low", "moderate", "high", "unknown"] = Field(
        description=(
            "Coarse safety tier from the toxicity text and toxicity-related category tags, per the rubric in the "
            "system prompt. 'unknown' if the input has no toxicity information."
        )
    )
    safety_evidence: str = Field(
        default="",
        description="Verbatim quote (at most 30 words) supporting the tier. Required unless safety_tier is 'unknown'.",
    )
    rna_viruses_inhibited: List[str] = Field(
        default_factory=list,
        description="Names of RNA viruses the input says the drug inhibits or treats. Empty if none are named.",
    )
    dna_viruses_inhibited: List[str] = Field(
        default_factory=list,
        description="Names of DNA viruses the input says the drug inhibits or treats. Empty if none are named.",
    )
    endothelial: EvidencedFlag = Field(
        default_factory=EvidencedFlag,
        description="Known to treat endothelial dysfunction, support endothelial function or block endothelial inflammation.",
    )
    neuronal: EvidencedFlag = Field(
        default_factory=EvidencedFlag,
        description="Known to treat neuronal dysfunction or encephalitis, or to be neuroprotective.",
    )
    respiratory: EvidencedFlag = Field(
        default_factory=EvidencedFlag,
        description="Known to treat respiratory dysfunction or support respiratory function.",
    )

    @field_validator("drug_class", "mechanism_of_action", "toxicity_summary", "potency_summary")
    @classmethod
    def _text_fields(cls, v: str) -> str:
        return _norm_text(v)

    @field_validator("rna_viruses_inhibited", "dna_viruses_inhibited")
    @classmethod
    def _dedupe(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for x in v:
            x = (x or "").strip()
            if x and x.lower() not in seen:
                seen.add(x.lower())
                out.append(x)
        return out

    @model_validator(mode="after")
    def _tier_needs_evidence(self):
        # Raising here makes `instructor` retry the call with this message.
        self.safety_evidence = (self.safety_evidence or "").strip()
        if self.safety_tier == "unknown":
            self.safety_evidence = ""
        elif not self.safety_evidence:
            raise ValueError(
                "safety_evidence is required when safety_tier is 'low', 'moderate' or 'high'. "
                "Quote the input, or set safety_tier to 'unknown' if it has no toxicity information."
            )
        return self
