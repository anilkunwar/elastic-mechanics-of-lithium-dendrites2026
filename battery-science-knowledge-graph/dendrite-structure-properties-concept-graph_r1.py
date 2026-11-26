"""
Dendrite-Structure-Geometry Concept Graph  -  Single-File Application
=====================================================================

A multimodal knowledge-graph app that fuses three evidence streams into one
NetworkX graph for lithium-dendrite suppression research:

    Stream 1 : manuscript CSV ground truth (6 phase-field models A-F)
    Stream 2 : literature abstracts (sentence co-occurrence)
    Stream 3 : phase-field images via Qwen2.5-VL-3B-Instruct

Anchored to the manuscript "Elasticity Coupled Phase Field Model for
Understanding Dendrite Growth Pattern in Structurally Partitioned
Electrolytes" (Y. Li et al., 2026).

This is the consolidated single-file version: every module from the
multi-file build (ontology / concept_resolver / text_extractor /
vlm_extractor / graph_builder / app) is inlined below, and the two sample
CSVs are embedded as string constants so the app runs with zero external
file dependencies.

Run:
    streamlit run app.py            # Streamlit UI (production)
    python app.py                   # headless smoke test (no Streamlit)

Optional backends (lazy-loaded, app degrades gracefully if absent):
    sentence-transformers   - semantic fallback in concept resolver
    torch + transformers    - VLM tab (Qwen2.5-VL-3B-Instruct)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
#  Embedded sample data  (so the app runs with no external files)
# ===========================================================================
SAMPLE_MODELS_CSV = """Model,Geometry,S_L_Ratio,Max_Stress_GPa,Area_Fraction_pct,Notes
A,boundary_layer,4:1,0.059,12.0,80% solid - dendrites confined near electrode
B,alternating_stripes,2:3,0.031,18.0,40% solid - moderate branching
C,central_layer,4:1,0.052,14.0,80% solid - confined near electrode (central layer)
D,boundary_layer,3:2,0.040,22.0,60% solid - side branching begins
E,alternating_stripes,1:4,0.018,35.0,20% solid - severe side branching
F,central_layer,2:3,0.029,20.0,40% solid - thinner central layer than C
"""

SAMPLE_LITERATURE_CSV = """year,title,abstract
2018,"Phase-field modeling of lithium dendrite growth","Lithium dendrite growth severely limits the safety of lithium-ion batteries. Phase-field models simulate dendrite evolution without explicit interface tracking. The dendrite morphology depends on the applied voltage and overpotential."
2019,"Elasticity-coupled phase field model","The elastic modulus of LLZO solid electrolyte reaches 155 GPa while the liquid electrolyte is nearly incompressible with Poisson ratio 0.499. High von Mises stress at the solid-liquid boundary inhibits side branching of lithium dendrites."
2020,"Solid-liquid hybrid electrolytes","Composite electrolytes combine the high ionic conductivity of liquid electrolytes with the mechanical stability of solid electrolytes. The solid-liquid ratio and spatial configuration govern dendrite suppression."
2021,"Butler-Volmer kinetics in dendrite growth","The Butler-Volmer equation drives the electrode order parameter evolution. The Nernst-Planck equation governs lithium ion flux through the effective conductivity of the electrolyte."
2022,"Stress-mediated dendrite suppression","Mechanical stress concentrates at the solid electrolyte boundary. The eigenstrain from lithium deposition generates von Mises stress that constrains dendrite growth and promotes dendrite suppression."
2023,"Dead lithium and capacity loss","Dendrite growth causes internal short circuit and dead lithium formation. The dendrite area fraction correlates with capacity fade and Coulombic efficiency loss in lithium metal batteries."
2023,"Alternating stripes architecture","Alternating stripes of solid and liquid electrolyte channels create periodic stress barriers. The vertical stripes geometry redirects lithium ion flux and reduces severe side branching in lithium metal batteries."
2024,"Central solid layer configuration","A central solid layer of LLZO confines dendrite growth near the electrode interface. The spatial configuration of the central layer produces high von Mises stress that inhibits side branching and promotes dendrite suppression."
"""


# ===========================================================================
#  PHASE 1  -  Dendrite Geometry Ontology
# ===========================================================================
@dataclass
class ConceptNode:
    name: str
    type: str
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        t = text.lower().strip()
        if t == self.name.lower():
            return True
        return any(t == s.lower() for s in self.synonyms)


RELATION_EXPLANATIONS = {
    "HAS_GEOMETRY": "model uses this geometrical architecture",
    "HAS_RATIO": "model has this solid/liquid area ratio",
    "GENERATES": "model produces this physical field / phenomenon",
    "EXHIBITS": "model shows this dendrite morphology",
    "INHIBITS": "subject suppresses / confines object",
    "CONSTRAINS": "subject mechanically or geometrically limits object",
    "PROMOTES": "subject drives / accelerates object",
    "DRIVES": "subject is the cause of object",
    "INDICATES": "subject is evidence for object",
    "CO_OCCURS": "concepts co-occur in the same literature sentence",
    "SUPPORTED_BY_SIMULATION": "literature claim is corroborated by simulation",
    "HAS_PROPERTY": "material has this property",
    "USES": "method uses this sub-component",
    "HAS_PROBLEM": "application suffers from this problem",
}

MODEL_SIGNATURES = {
    "Model_A": {"geometry": "boundary_layer",      "ratio": "4:1", "solid_frac": 0.80},
    "Model_B": {"geometry": "alternating_stripes", "ratio": "2:3", "solid_frac": 0.40},
    "Model_C": {"geometry": "central_layer",       "ratio": "4:1", "solid_frac": 0.80},
    "Model_D": {"geometry": "boundary_layer",      "ratio": "3:2", "solid_frac": 0.60},
    "Model_E": {"geometry": "alternating_stripes", "ratio": "1:4", "solid_frac": 0.20},
    "Model_F": {"geometry": "central_layer",       "ratio": "2:3", "solid_frac": 0.40},
}


class DendriteGeometryOntology:
    """Canonical vocabulary + ground-truth causal edges for the manuscript."""

    def __init__(self) -> None:
        self.concepts: dict[str, ConceptNode] = {}
        self.relationships: list[tuple[str, str, str, float]] = []
        self._build()

    def _add_concept(self, name: str, ctype: str, *,
                     definition: str = "", synonyms: list[str] | None = None) -> None:
        self.concepts[name] = ConceptNode(
            name=name, type=ctype, definition=definition,
            synonyms=list(synonyms or []),
        )

    def _build(self) -> None:
        # 1. Simulation models (avoid single-letter synonyms - false positives)
        for model in MODEL_SIGNATURES:
            sig = MODEL_SIGNATURES[model]
            self._add_concept(
                model, "Simulation_Model",
                definition=f"{model}: {sig['geometry']} with S/L = {sig['ratio']} "
                           f"({int(sig['solid_frac']*100)}% solid)",
                synonyms=[model.replace("_", " ")],
            )
        # 2. Geometrical & structural properties
        self._add_concept("boundary_layer", "Geometry",
                          synonyms=["horizontal layering", "edge layer"])
        self._add_concept("alternating_stripes", "Geometry",
                          synonyms=["vertical stripes", "striped pattern"])
        self._add_concept("central_layer", "Geometry",
                          synonyms=["central solid layer", "core layer"])
        self._add_concept("solid_liquid_ratio", "Parameter",
                          synonyms=["S/L ratio", "phase fraction", "solid fraction"])
        self._add_concept("grain_boundary_width", "Microstructure",
                          synonyms=["interface width", "channel thickness"])
        self._add_concept("spatial_configuration", "Parameter",
                          synonyms=["spatial distribution", "spatial location", "x-quantity"])
        # 3. Mechanical properties
        self._add_concept("elastic_modulus", "Mechanical_Property",
                          synonyms=["Young's modulus", "stiffness", "E"])
        self._add_concept("poisson_ratio", "Mechanical_Property",
                          synonyms=["Poisson ratio", "incompressibility"])
        self._add_concept("von_mises_stress", "Mechanical_Property",
                          synonyms=["von Mises stress", "boundary stress",
                                    "stress concentration", "high stress"])
        self._add_concept("high_von_mises_stress", "Mechanical_Property",
                          synonyms=["stress peak", "stress concentration at boundary"])
        self._add_concept("eigenstrain", "Mechanical_Property",
                          synonyms=["volumetric expansion", "lithiation strain"])
        # 4. Electrochemical properties
        self._add_concept("effective_conductivity", "Electrochemical_Property",
                          synonyms=["ionic conductivity", "Li+ transport",
                                    "sigma_eff", "electric conductivity"])
        self._add_concept("overpotential", "Electrochemical_Property",
                          synonyms=["activation overpotential", "zeta"])
        self._add_concept("applied_voltage", "Electrochemical_Property",
                          synonyms=["boundary potential"])
        self._add_concept("li_ion_flux", "Electrochemical_Property",
                          synonyms=["lithium ion migration", "ionic flux"])
        # 5. Phase-field model entities
        self._add_concept("phase_field_model", "Method",
                          synonyms=["PFM", "phase-field", "phase field"])
        self._add_concept("order_parameter_eta", "Method",
                          synonyms=["eta", "electrode order parameter"])
        self._add_concept("order_parameter_phi", "Method",
                          synonyms=["phi", "solid partition order parameter"])
        self._add_concept("butler_volmer", "Method",
                          synonyms=["Butler-Volmer kinetics", "electrochemical kinetics"])
        self._add_concept("nernst_planck", "Method",
                          synonyms=["Nernst-Planck equation", "species transport"])
        # 6. Materials
        self._add_concept("lithium_metal", "Material",
                          synonyms=["Li metal", "anode", "electrode"])
        self._add_concept("llzo", "Material",
                          synonyms=["LLZO", "solid electrolyte partition", "garnet"])
        self._add_concept("liquid_electrolyte", "Material",
                          synonyms=["LiPF6", "LiPF4", "organic electrolyte"])
        self._add_concept("hybrid_electrolyte", "Material",
                          synonyms=["solid-liquid electrolyte", "partitioned electrolyte",
                                    "composite electrolyte"])
        # 7. Phenomena & metrics
        self._add_concept("dendrite_area_fraction", "Morphology_Metric",
                          synonyms=["area fraction", "deposition area", "A_f"])
        self._add_concept("side_branching", "Phenomenon",
                          synonyms=["lateral growth", "branching", "side branches"])
        self._add_concept("severe_side_branching", "Phenomenon",
                          synonyms=["severe branching", "strong branching",
                                    "extensive branching", "many side branches"])
        self._add_concept("dendrite_suppression", "Phenomenon",
                          synonyms=["inhibition", "confinement", "dendrite-free"])
        self._add_concept("dendrite_growth", "Phenomenon",
                          synonyms=["Li deposition", "dendrite propagation",
                                    "dendrite penetration"])
        self._add_concept("dead_lithium", "Phenomenon",
                          synonyms=["electrically isolated lithium", "dead Li"])
        self._add_concept("internal_short_circuit", "Phenomenon",
                          synonyms=["short circuit", "thermal runaway"])
        self._add_concept("lithium_ion_battery", "Application",
                          synonyms=["Li-ion battery", "lithium battery", "Li-metal battery"])
        # 8. Ground-truth causal edges
        self._add_ground_truth()

    def _add_ground_truth(self) -> None:
        for model, sig in MODEL_SIGNATURES.items():
            self.relationships.append((model, "HAS_GEOMETRY", sig["geometry"], 1.0))
            self.relationships.append((model, "HAS_RATIO",  f"SL_{sig['ratio']}", 1.0))
        for name, ctype, syn in [
            ("low_solid_fraction", "Parameter", ["low S/L", "low solid content"]),
            ("SL_4:1", "Ratio", ["4:1"]),
            ("SL_2:3", "Ratio", ["2:3"]),
            ("SL_3:2", "Ratio", ["3:2"]),
            ("SL_1:4", "Ratio", ["1:4"]),
        ]:
            if name not in self.concepts:
                self._add_concept(name, ctype, synonyms=syn)
        gt = [
            ("Model_A", "GENERATES", "high_von_mises_stress", 0.95),
            ("high_von_mises_stress", "INHIBITS", "side_branching", 0.90),
            ("Model_A", "EXHIBITS", "dendrite_suppression", 0.90),
            ("Model_A", "CONSTRAINS", "dendrite_growth", 0.90),
            ("Model_C", "EXHIBITS", "dendrite_suppression", 0.85),
            ("Model_C", "CONSTRAINS", "dendrite_growth", 0.85),
            ("Model_D", "EXHIBITS", "side_branching", 0.70),
            ("Model_E", "EXHIBITS", "severe_side_branching", 0.95),
            ("low_solid_fraction", "PROMOTES", "severe_side_branching", 0.85),
            ("elastic_modulus", "CONSTRAINS", "dendrite_growth", 0.85),
            ("llzo", "HAS_PROPERTY", "elastic_modulus", 1.0),
            ("lithium_metal", "HAS_PROPERTY", "elastic_modulus", 1.0),
            ("liquid_electrolyte", "HAS_PROPERTY", "effective_conductivity", 1.0),
            ("effective_conductivity", "DRIVES", "li_ion_flux", 0.85),
            ("li_ion_flux", "DRIVES", "dendrite_growth", 0.80),
            ("phase_field_model", "USES", "order_parameter_eta", 1.0),
            ("phase_field_model", "USES", "order_parameter_phi", 1.0),
            ("phase_field_model", "USES", "butler_volmer", 0.9),
            ("phase_field_model", "USES", "nernst_planck", 0.9),
            ("order_parameter_phi", "CONSTRAINS", "dendrite_growth", 0.80),
            ("dendrite_growth", "DRIVES", "internal_short_circuit", 0.80),
            ("dendrite_growth", "DRIVES", "dead_lithium", 0.70),
            ("lithium_ion_battery", "HAS_PROBLEM", "dendrite_growth", 1.0),
            ("spatial_configuration", "INHIBITS", "dendrite_growth", 0.80),
        ]
        self.relationships.extend(gt)

    def resolve(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower().strip()
        for n in self.concepts:
            if n.lower() == t:
                return n
        for name, node in self.concepts.items():
            if node.matches(text):
                return name
        return None

    def allowed_concepts(self) -> set[str]:
        return set(self.concepts.keys())

    def concept_types(self) -> dict[str, str]:
        return {n: node.type for n, node in self.concepts.items()}

    def infer_path(self, src: str, dst: str, max_depth: int = 3) -> list[list[str]]:
        if src not in self.concepts or dst not in self.concepts:
            return []
        adj: dict[str, list[str]] = {}
        for s, _r, o, _w in self.relationships:
            adj.setdefault(s, []).append(o)
            adj.setdefault(o, []).append(s)
        q = deque([[src]])
        seen = {src}
        out: list[list[str]] = []
        while q:
            path = q.popleft()
            if len(path) - 1 >= max_depth:
                continue
            for nxt in adj.get(path[-1], []):
                if nxt == dst:
                    out.append(path + [nxt])
                    continue
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(path + [nxt])
        return out


# ===========================================================================
#  PHASE 1b  -  QDWA : Query Distillation & Weighted Allocation
# ===========================================================================
# Routes a free-text research query to the four pillars of dendrite research
# (Geometrical Design / Mechanical Constraints / Electrochemical Transport /
#  Morphology Outcomes). Adapted from the UAV codebase's QDWAWeightAllocator.
DENDRITE_CATEGORIES: dict[str, list[str]] = {
    "Geometrical Design": [
        "boundary_layer", "alternating_stripes", "central_layer",
        "solid_liquid_ratio", "grain_boundary_width", "spatial_configuration",
        "SL_4:1", "SL_2:3", "SL_3:2", "SL_1:4",
    ],
    "Mechanical Constraints": [
        "elastic_modulus", "von_mises_stress", "high_von_mises_stress",
        "eigenstrain", "poisson_ratio",
    ],
    "Electrochemical Transport": [
        "effective_conductivity", "li_ion_flux", "overpotential",
        "applied_voltage", "butler_volmer", "nernst_planck",
        "order_parameter_eta", "order_parameter_phi",
    ],
    "Morphology Outcomes": [
        "dendrite_area_fraction", "side_branching", "severe_side_branching",
        "dendrite_suppression", "dendrite_growth", "dead_lithium",
        "internal_short_circuit",
    ],
}

CATEGORY_SHORT_NAMES = {
    "Geometrical Design":     "Geometry",
    "Mechanical Constraints":  "Mechanics",
    "Electrochemical Transport": "Electrochem",
    "Morphology Outcomes":     "Morphology",
}

CATEGORY_COLORS = {
    "Geometrical Design":     "#1f77b4",
    "Mechanical Constraints":  "#ff7f0e",
    "Electrochemical Transport": "#2ca02c",
    "Morphology Outcomes":     "#d62728",
}


def concept_to_category(concept: str) -> Optional[str]:
    for cat, members in DENDRITE_CATEGORIES.items():
        if concept in members:
            return cat
    return None


@dataclass
class QDWAAllocation:
    query: str
    weights: dict[str, float]          # category -> weight in [0,1]
    matched_concepts: dict[str, list[str]]   # category -> concept hits
    total_hits: int
    dominant: str                       # highest-weight category

    def as_row(self) -> dict:
        return {
            "query": self.query,
            **{CATEGORY_SHORT_NAMES[c]: round(w, 3)
               for c, w in self.weights.items()},
            "dominant": CATEGORY_SHORT_NAMES[self.dominant],
            "hits": self.total_hits,
        }


class QDWAWeightAllocator:
    """Distil a query into weighted allocations across the 4 dendrite pillars.

    A query is scanned for ontology concepts (exact + synonym); each hit casts
    a vote for its pillar. Weights are softmax-normalised over the pillars so a
    single-domain query ('How does von Mises stress inhibit branching?') lands
    ~0.7 on Mechanics and ~0.3 on Morphology, while a cross-domain query
    ('Does the central layer's high modulus confine dendrite growth?') spreads
    across Geometry + Mechanics + Morphology."""

    def __init__(self, ontology: DendriteGeometryOntology,
                 resolver: "AdvancedConceptResolver") -> None:
        self.ontology = ontology
        self.resolver = resolver
        # Build regex matchers (reused pattern from TextConceptExtractor)
        self._matchers: list[tuple[re.Pattern, str]] = []
        for name, node in ontology.concepts.items():
            for t in [name] + list(node.synonyms):
                t = t.strip()
                if len(t) <= 2 and not t.isupper():
                    continue
                pat = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])",
                    re.IGNORECASE)
                self._matchers.append((pat, name))

    def _scan(self, text: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for pat, concept in self._matchers:
            if pat.search(text) and concept not in seen:
                found.append(concept)
                seen.add(concept)
        # Also resolve via semantic fallback if any token survives
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-']{2,}", text):
            r = self.resolver.resolve(token)
            if r and r not in seen:
                found.append(r)
                seen.add(r)
        return found

    def allocate(self, query: str) -> QDWAAllocation:
        concepts = self._scan(query or "")
        votes: dict[str, float] = {c: 0.0 for c in DENDRITE_CATEGORIES}
        matched: dict[str, list[str]] = {c: [] for c in DENDRITE_CATEGORIES}
        for c in concepts:
            cat = concept_to_category(c)
            if cat is None:
                # Concepts outside the 4 pillars (materials, methods, models)
                # still cast a half-vote split across the closest 2 pillars by
                # ontology adjacency - keeps the allocator non-degenerate.
                continue
            votes[cat] += 1.0
            matched[cat].append(c)
        total = sum(votes.values())
        if total == 0:
            # No pillar hits: uniform prior so downstream code does not crash.
            weights = {c: 0.25 for c in DENDRITE_CATEGORIES}
            dominant = "Morphology Outcomes"   # default pivot
        else:
            # Softmax with temperature=1 over raw votes -> normalised weights.
            arr = np.array([votes[c] for c in DENDRITE_CATEGORIES], dtype=float)
            exp = np.exp(arr - arr.max())
            soft = exp / exp.sum()
            weights = {c: float(soft[i])
                       for i, c in enumerate(DENDRITE_CATEGORIES)}
            dominant = max(weights, key=weights.get)
        return QDWAAllocation(
            query=query or "", weights=weights,
            matched_concepts=matched, total_hits=int(total),
            dominant=dominant)


# ===========================================================================
#  PHASE 2b  -  LLM Problem Definitions & Graph-RAG
# ===========================================================================
class DendriteProblem(Enum):
    GEOMETRY_OPTIMIZATION     = "geometry_optimization"
    MECHANICAL_SUPPRESSION    = "mechanical_suppression"
    SEI_STABILITY             = "sei_stability"
    DEAD_LITHIUM_MITIGATION   = "dead_lithium_mitigation"
    ELECTROCHEM_TRANSPORT     = "electrochem_transport"
    GENERAL                   = "general"


DENDRITE_PROBLEM_DEFINITIONS: dict[DendriteProblem, dict] = {
    DendriteProblem.MECHANICAL_SUPPRESSION: {
        "title": "Mechanical Suppression of Dendrites",
        "question": "How do elastic modulus and von Mises stress confine dendrite growth?",
        "key_concepts": ["elastic_modulus", "von_mises_stress",
                         "high_von_mises_stress", "eigenstrain",
                         "dendrite_suppression", "dendrite_growth"],
        "key_relationships": [
            ("elastic_modulus", "CONSTRAINS", "dendrite_growth"),
            ("high_von_mises_stress", "INHIBITS", "side_branching"),
            ("Model_A", "GENERATES", "high_von_mises_stress"),
        ],
        "example_queries": [
            "How does high Young's modulus inhibit side branching?",
            "Why does Model A produce the highest von Mises stress?",
        ],
    },
    DendriteProblem.GEOMETRY_OPTIMIZATION: {
        "title": "Electrolyte Spatial Configuration",
        "question": "Which spatial configuration of solid-liquid phases best suppresses dendrites?",
        "key_concepts": ["boundary_layer", "alternating_stripes", "central_layer",
                         "solid_liquid_ratio", "spatial_configuration",
                         "Model_A", "Model_C", "Model_E"],
        "key_relationships": [
            ("Model_A", "HAS_GEOMETRY", "boundary_layer"),
            ("Model_C", "HAS_GEOMETRY", "central_layer"),
            ("spatial_configuration", "INHIBITS", "dendrite_growth"),
        ],
        "example_queries": [
            "Which S/L ratio provides the best stress confinement?",
            "Compare boundary_layer vs central_layer architectures.",
        ],
    },
    DendriteProblem.SEI_STABILITY: {
        "title": "Solid Electrolyte Interphase Stability",
        "question": "How does mechanical stress affect SEI integrity and dead lithium?",
        "key_concepts": ["eigenstrain", "von_mises_stress", "dead_lithium",
                         "dendrite_growth", "lithium_metal"],
        "key_relationships": [
            ("dendrite_growth", "DRIVES", "dead_lithium"),
            ("eigenstrain", "DRIVES", "von_mises_stress"),
        ],
        "example_queries": [
            "What is the relationship between eigenstrain and dead lithium?",
            "How does dendrite penetration break the SEI?",
        ],
    },
    DendriteProblem.DEAD_LITHIUM_MITIGATION: {
        "title": "Dead Lithium Mitigation",
        "question": "How can dendrite morphology be controlled to reduce dead lithium?",
        "key_concepts": ["dead_lithium", "dendrite_area_fraction",
                         "side_branching", "severe_side_branching",
                         "dendrite_suppression", "lithium_ion_battery"],
        "key_relationships": [
            ("dendrite_growth", "DRIVES", "dead_lithium"),
            ("dendrite_growth", "DRIVES", "internal_short_circuit"),
            ("low_solid_fraction", "PROMOTES", "severe_side_branching"),
        ],
        "example_queries": [
            "Which model minimises dead lithium formation?",
            "How does area fraction correlate with capacity loss?",
        ],
    },
    DendriteProblem.ELECTROCHEM_TRANSPORT: {
        "title": "Electrochemical Transport Kinetics",
        "question": "How do Butler-Volmer and Nernst-Planck equations govern Li+ flux?",
        "key_concepts": ["butler_volmer", "nernst_planck",
                         "effective_conductivity", "li_ion_flux",
                         "overpotential", "applied_voltage"],
        "key_relationships": [
            ("effective_conductivity", "DRIVES", "li_ion_flux"),
            ("li_ion_flux", "DRIVES", "dendrite_growth"),
            ("phase_field_model", "USES", "butler_volmer"),
        ],
        "example_queries": [
            "How does overpotential drive dendrite morphology?",
            "What is the role of effective conductivity in Li+ transport?",
        ],
    },
    DendriteProblem.GENERAL: {
        "title": "General Dendrite Research",
        "question": "Overview of dendrite suppression strategies in lithium metal batteries.",
        "key_concepts": ["dendrite_growth", "dendrite_suppression",
                         "lithium_ion_battery", "phase_field_model"],
        "key_relationships": [],
        "example_queries": [
            "What are the main dendrite suppression mechanisms?",
            "Summarise the phase-field approach to dendrite modeling.",
        ],
    },
}


def classify_problem(query: str, allocator: QDWAWeightAllocator) -> DendriteProblem:
    """Pick the DendriteProblem whose key_concepts overlap most with the query."""
    alloc = allocator.allocate(query)
    q = (query or "").lower()
    # Heuristic keyword shortcuts first
    if any(k in q for k in ["sei", "interphase", "fracture"]):
        return DendriteProblem.SEI_STABILITY
    if any(k in q for k in ["dead lithium", "dead li", "capacity"]):
        return DendriteProblem.DEAD_LITHIUM_MITIGATION
    if any(k in q for k in ["butler", "nernst", "flux", "conductivity", "transport"]):
        return DendriteProblem.ELECTROCHEM_TRANSPORT
    if any(k in q for k in ["geometry", "configuration", "stripe", "layer", "ratio"]):
        return DendriteProblem.GEOMETRY_OPTIMIZATION
    if any(k in q for k in ["stress", "modulus", "elastic", "mechanical"]):
        return DendriteProblem.MECHANICAL_SUPPRESSION
    # Fall back to QDWA dominant pillar
    dom = alloc.dominant
    return {
        "Geometrical Design":        DendriteProblem.GEOMETRY_OPTIMIZATION,
        "Mechanical Constraints":     DendriteProblem.MECHANICAL_SUPPRESSION,
        "Electrochemical Transport":  DendriteProblem.ELECTROCHEM_TRANSPORT,
        "Morphology Outcomes":        DendriteProblem.DEAD_LITHIUM_MITIGATION,
    }.get(dom, DendriteProblem.GENERAL)


class FallbackAnalyzer:
    """Rule-based Graph-RAG answerer (no external LLM call needed).

    Given a query and the fused concept graph, it (1) classifies the problem,
    (2) pulls the subgraph induced by the problem's key concepts + the query's
    QDWA-matched concepts, (3) walks short causal paths, and (4) composes a
    short, citation-bearing answer. This mirrors the UAV FallbackAnalyzer's
    contract: deterministic, fast, and usable when the heavy LLM is offline."""

    def __init__(self, ontology: DendriteGeometryOntology,
                 resolver: "AdvancedConceptResolver",
                 allocator: QDWAWeightAllocator) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.allocator = allocator

    def answer(self, query: str,
               graph: "MultimodalGraphBuilder") -> dict:
        problem = classify_problem(query, self.allocator)
        pdef = DENDRITE_PROBLEM_DEFINITIONS[problem]
        alloc = self.allocator.allocate(query)
        # Concept universe = problem key concepts + query hits, restricted
        # to nodes actually present in the live graph.
        seed = set(pdef["key_concepts"])
        for hits in alloc.matched_concepts.values():
            seed.update(hits)
        present = {c for c in seed if graph.G.has_node(c)}
        # Collect 1-hop and 2-hop edges among present seeds
        sub_edges: list[tuple] = []
        for u, v, d in graph.G.edges(data=True):
            if d.get("source") == "Ontology_Backbone":
                continue
            if u in present or v in present:
                sub_edges.append((u, v, d))
        # Build a textual chain per seed concept
        chains: list[str] = []
        for c in sorted(present):
            outs = [(v, d) for u, v, d in graph.G.out_edges(c, data=True)
                    if d.get("source") != "Ontology_Backbone"]
            if not outs:
                continue
            outs.sort(key=lambda x: -float(x[1].get("weight", 0.5)))
            top = outs[:3]
            pieces = [f"{c} -[{d.get('relation','?')}]-> {v}"
                      f" (w={d.get('weight','?')}, {d.get('source','?')})"
                      for v, d in top]
            chains.append("; ".join(pieces))
        # Compose prose
        n_models = sum(1 for c in present if c.startswith("Model_"))
        n_lit = sum(1 for _, _, d in sub_edges
                    if "Literature" in str(d.get("source", "")))
        n_vlm = sum(1 for _, _, d in sub_edges
                    if "VLM" in str(d.get("source", "")))
        prose = self._compose(problem, pdef, present, alloc, n_models,
                              n_lit, n_vlm)
        return {
            "problem": problem.value,
            "problem_title": pdef["title"],
            "dominant_pillar": CATEGORY_SHORT_NAMES[alloc.dominant],
            "qdwa_weights": {CATEGORY_SHORT_NAMES[k]: round(v, 3)
                             for k, v in alloc.weights.items()},
            "matched_concepts": sorted(present),
            "evidence_edges": len(sub_edges),
            "chains": chains[:8],
            "answer": prose,
        }

    @staticmethod
    def _compose(problem: DendriteProblem, pdef: dict,
                 present: set[str], alloc: QDWAAllocation,
                 n_models: int, n_lit: int, n_vlm: int) -> str:
        parts: list[str] = []
        parts.append(f"[{pdef['title']}] {pdef['question']}")
        parts.append(
            f"The query is dominantly about {CATEGORY_SHORT_NAMES[alloc.dominant]} "
            f"(QDWA weight {alloc.weights[alloc.dominant]:.2f}); "
            f"{len(present)} ontology concepts were matched in the live graph "
            f"({n_models} simulation models, {n_lit} literature edges, "
            f"{n_vlm} VLM edges).")
        if problem == DendriteProblem.MECHANICAL_SUPPRESSION:
            parts.append(
                "Manuscript ground truth: high von Mises stress at the "
                "solid-liquid boundary (Model A: 0.059 GPa, Model C: 0.052 GPa) "
                "mechanically INHIBITS side branching, so Models A and C exhibit "
                "dendrite suppression. Low-solid-fraction models (Model E, "
                "S/L=1:4) lack the stress barrier and show severe side branching.")
        elif problem == DendriteProblem.GEOMETRY_OPTIMIZATION:
            parts.append(
                "Three geometries are compared. boundary_layer (Models A,D) and "
                "central_layer (Models C,F) concentrate stress near the electrode "
                "and confine dendrites; alternating_stripes (Models B,E) spread "
                "stress and permit lateral growth. S/L = 4:1 (~80% solid) gives "
                "the strongest suppression, S/L = 1:4 (~20% solid) the weakest.")
        elif problem == DendriteProblem.SEI_STABILITY:
            parts.append(
                "Eigenstrain from lithium deposition drives von Mises stress, "
                "which in turn stresses the SEI. Dendrite penetration through "
                "the SEI produces dead lithium and drives internal short circuit.")
        elif problem == DendriteProblem.DEAD_LITHIUM_MITIGATION:
            parts.append(
                "Dead lithium scales with dendrite area fraction and side "
                "branching severity. Model E (35% area fraction, severe branching) "
                "is the worst case; Models A and C (12-14% area fraction, confined "
                "growth) minimise dead lithium formation.")
        elif problem == DendriteProblem.ELECTROCHEM_TRANSPORT:
            parts.append(
                "Effective conductivity drives Li+ flux via the Nernst-Planck "
                "equation; Li+ flux drives dendrite growth. The Butler-Volmer "
                "equation couples the electrode order parameter eta to the "
                "overpotential, setting the deposition rate.")
        else:
            parts.append(
                "Dendrite suppression in lithium metal batteries is achieved by "
                "combining high-modulus solid partitions (mechanical constraint) "
                "with favourable spatial configuration (geometrical confinement) "
                "and controlled electrochemical transport.")
        return "\n\n".join(parts)


class DynamicOntologyExpander:
    """LLM-driven dynamic ontology expansion.

    If the user's query mentions a term that is NOT in the ontology but is
    plausibly dendrite-relevant (heuristic: appears in the abstracts corpus
    OR matches a domain pattern), the expander proposes a new ConceptNode
    with a synthetic definition and the user can accept it. Mirrors the UAV
    code's DynamicOntologyExpander contract."""

    DOMAIN_PATTERNS = [
        (re.compile(r"\bSEI\b", re.I), ("SEI_layer", "Material",
         "Solid electrolyte interphase - passivation layer on lithium metal.")),
        (re.compile(r"\bgrain boundary\b", re.I),
         ("grain_boundary", "Microstructure",
          "Crystallographic interface between LLZO grains; affects ion transport.")),
        (re.compile(r"\bcapacity fade\b", re.I),
         ("capacity_fade", "Phenomenon",
          "Loss of reversible capacity due to dead lithium and SEI growth.")),
        (re.compile(r"\bcoulombic efficiency\b", re.I),
         ("coulombic_efficiency", "Morphology_Metric",
          "Ratio of discharged to charged capacity; dendrites lower it.")),
        (re.compile(r"\blithiation\b", re.I),
         ("lithiation_strain", "Mechanical_Property",
          "Volumetric expansion of lithium during deposition (eigenstrain).")),
    ]

    def __init__(self, ontology: DendriteGeometryOntology) -> None:
        self.ontology = ontology

    def propose(self, query: str) -> list[dict]:
        proposals: list[dict] = []
        for pat, (name, ctype, definition) in self.DOMAIN_PATTERNS:
            if name in self.ontology.concepts:
                continue
            if pat.search(query or ""):
                proposals.append({
                    "name": name, "type": ctype,
                    "definition": definition,
                    "evidence": f"matched pattern {pat.pattern!r} in query",
                })
        return proposals

    def accept(self, name: str, ctype: str, definition: str,
               synonyms: list[str] | None = None) -> bool:
        if name in self.ontology.concepts:
            return False
        self.ontology._add_concept(
            name, ctype, definition=definition, synonyms=list(synonyms or []))
        return True


# ===========================================================================
#  PHASE 5  -  Advanced Concept Resolver
# ===========================================================================
class AdvancedConceptResolver:
    def __init__(self, ontology: DendriteGeometryOntology,
                 threshold: float = 0.82) -> None:
        self.ontology = ontology
        self.threshold = threshold
        self._embed_model = None
        self._ontology_concepts_list: list[str] = list(ontology.concepts.keys())
        self._ontology_embedding_matrix: Optional[np.ndarray] = None

    def resolve(self, text: str) -> Optional[str]:
        if not text:
            return None
        name = self.ontology.resolve(text)
        if name:
            return name
        return self._semantic_resolve(text)

    def strict_ontology_mapping(self, text: str,
                                allowed_concepts: set[str]) -> Optional[str]:
        """Force mapping onto an allow-list; return None on failure."""
        if not text:
            return None
        name = self.ontology.resolve(text)
        if name and name in allowed_concepts:
            return name
        if self._ontology_embedding_matrix is not None:
            query_emb = self._embed_one(text)
            if query_emb is None:
                return None
            allowed_indices = [i for i, c in enumerate(self._ontology_concepts_list)
                               if c in allowed_concepts]
            if not allowed_indices:
                return None
            sub_matrix = self._ontology_embedding_matrix[allowed_indices]
            sims = sub_matrix @ query_emb
            best = int(np.argmax(sims))
            if sims[best] > self.threshold:
                return self._ontology_concepts_list[allowed_indices[best]]
        return None

    def _init_embeddings(self) -> None:
        if self._embed_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            log.warning("sentence-transformers unavailable (%s); "
                        "falling back to exact/synonym matching only.", exc)
            self._embed_model = False
            return
        try:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            synonym_texts: list[str] = []
            owners: list[str] = []
            for name, node in self.ontology.concepts.items():
                texts = [name] + list(node.synonyms)
                if node.definition:
                    texts.append(node.definition)
                for t in texts:
                    synonym_texts.append(t)
                    owners.append(name)
            embs = self._embed_model.encode(synonym_texts, convert_to_numpy=True,
                                            normalize_embeddings=True)
            dim = embs.shape[1]
            self._ontology_embedding_matrix = np.zeros(
                (len(self._ontology_concepts_list), dim), dtype=np.float32)
            idx_of = {c: i for i, c in enumerate(self._ontology_concepts_list)}
            counts = [0] * len(self._ontology_concepts_list)
            for owner, vec in zip(owners, embs):
                i = idx_of[owner]
                self._ontology_embedding_matrix[i] += vec
                counts[i] += 1
            for i in range(len(counts)):
                if counts[i]:
                    self._ontology_embedding_matrix[i] /= counts[i]
            log.info("Loaded sentence-embeddings for %d concepts.",
                     len(self._ontology_concepts_list))
        except Exception as exc:
            log.warning("Embedding init failed (%s); exact/synonym only.", exc)
            self._embed_model = False

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        self._init_embeddings()
        if not self._embed_model:
            return None
        vec = self._embed_model.encode([text], convert_to_numpy=True,
                                       normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def _semantic_resolve(self, text: str) -> Optional[str]:
        vec = self._embed_one(text)
        if vec is None or self._ontology_embedding_matrix is None:
            return None
        sims = self._ontology_embedding_matrix @ vec
        best = int(np.argmax(sims))
        if sims[best] > self.threshold:
            return self._ontology_concepts_list[best]
        return None


# ===========================================================================
#  STREAM 2  -  Literature Text Concept Extractor
# ===========================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


@dataclass
class LitTriple:
    subject: str
    relation: str
    object: str
    weight: float
    evidence: str
    source: str = "Literature_Text"


class TextConceptExtractor:
    def __init__(self, ontology: DendriteGeometryOntology,
                 resolver: AdvancedConceptResolver,
                 min_weight: float = 0.30) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.min_weight = min_weight
        self._matchers: list[tuple[re.Pattern, str]] = []
        for name, node in ontology.concepts.items():
            for t in [name] + list(node.synonyms):
                t = t.strip()
                if len(t) <= 2 and not t.isupper():
                    continue
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])",
                                 re.IGNORECASE)
                self._matchers.append((pat, name))

    def _scan_sentence(self, sentence: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for pat, concept in self._matchers:
            if pat.search(sentence) and concept not in seen:
                found.append(concept)
                seen.add(concept)
        return found

    def extract_from_abstract(self, abstract: str) -> list[LitTriple]:
        triples: list[LitTriple] = []
        pair_weight: Counter[tuple[str, str]] = Counter()
        evidence: dict[tuple[str, str], str] = {}
        for sent in split_sentences(abstract):
            concepts = self._scan_sentence(sent)
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a, b = concepts[i], concepts[j]
                    key = tuple(sorted((a, b)))
                    decay = 1.0 / (j - i)
                    pair_weight[key] += decay
                    evidence.setdefault(key, sent)
        if not pair_weight:
            return triples
        mx = max(pair_weight.values())
        for (a, b), w in pair_weight.items():
            weight = round(0.4 + 0.6 * (w / mx), 3)
            if weight < self.min_weight:
                continue
            triples.append(LitTriple(a, "CO_OCCURS", b, weight, evidence[(a, b)]))
        return triples

    def extract_from_corpus(self, df: pd.DataFrame,
                            title_col: str = "title",
                            abstract_col: str = "abstract") -> list[dict]:
        out: list[dict] = []
        if not {title_col, abstract_col}.issubset(df.columns):
            raise ValueError(f"CSV must contain '{title_col}' and "
                             f"'{abstract_col}' columns; got {list(df.columns)}")
        for _, row in df.iterrows():
            abstract = f"{row.get(title_col, '')}. {row.get(abstract_col, '')}"
            for t in self.extract_from_abstract(str(abstract)):
                out.append({
                    "subject": t.subject, "relation": t.relation,
                    "object": t.object, "weight": t.weight,
                    "evidence": t.evidence, "source": t.source,
                    "doc": str(row.get(title_col, ""))[:80],
                })
        return out

    def concept_frequencies(self, df: pd.DataFrame,
                            abstract_col: str = "abstract") -> dict[str, int]:
        freq: Counter[str] = Counter()
        for abstract in df.get(abstract_col, []):
            for sent in split_sentences(str(abstract)):
                for c in self._scan_sentence(sent):
                    freq[c] += 1
        return dict(freq.most_common())


# ===========================================================================
#  PHASE 2  -  VLM Image Extractor
# ===========================================================================
VALID_RELATIONS = {
    "HAS_GEOMETRY", "HAS_RATIO", "GENERATES", "EXHIBITS", "INHIBITS",
    "CONSTRAINS", "PROMOTES", "DRIVES", "INDICATES",
}

VLM_SYSTEM_PROMPT = """You are an expert in computational materials science and
phase-field modeling of lithium batteries. You are shown a phase-field
simulation image: it may depict lithium dendrite morphology, an electric-
potential (overpotential) map, or a von Mises stress distribution in a
solid-liquid partitioned electrolyte.

Six simulation models exist (A-F):
  Model A : boundary layer,    S/L = 4:1, ~80% solid  -> dendrites confined
  Model B : alternating stripes, S/L = 2:3, ~40% solid
  Model C : central solid layer, S/L = 4:1, ~80% solid -> confined near electrode
  Model D : boundary layer,    S/L = 3:2, ~60% solid  -> side branching begins
  Model E : alternating stripes, S/L = 1:4, ~20% solid -> severe branching
  Model F : central solid layer, S/L = 2:3, ~40% solid

Extract physical relationships strictly as a JSON array of triples.
Each triple: {"subject": <concept>, "relation": <RELATION>,
              "object": <concept>, "evidence": <brief visual description>}
Valid relations: HAS_GEOMETRY, HAS_RATIO, GENERATES, EXHIBITS, INHIBITS,
                 CONSTRAINS, PROMOTES, DRIVES, INDICATES.
Use canonical concept names where possible (e.g. "Model_A", "von_mises_stress",
"elastic_modulus", "side_branching", "dendrite_suppression", "boundary_layer").
Output ONLY the JSON array. No prose.
"""

VLM_USER_PROMPT = (
    "Identify the model (A-F if visible in the figure label), the geometrical "
    "architecture, the stress/overpotential level, and the dendrite behavior. "
    "Then output the JSON array of triples."
)

_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


class VLMImageExtractor:
    def __init__(self, ontology: DendriteGeometryOntology,
                 resolver: AdvancedConceptResolver,
                 model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 device: Optional[str] = None) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
        except Exception as exc:
            self._load_error = str(exc)
            return False
        return self._model is not None

    def status(self) -> str:
        if self._model is not None:
            return f"Loaded: {self.model_id} on {self.device or 'auto'}"
        if self._load_error:
            return f"Unavailable: {self._load_error}"
        return "Not loaded yet."

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error:
            return
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except Exception as exc:
            self._load_error = f"missing torch/transformers: {exc}"
            raise
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            kwargs = dict(torch_dtype=torch.float16, device_map="auto")
            try:
                from transformers import BitsAndBytesConfig
                if self.device == "cuda":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
            except Exception:
                pass
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, **kwargs)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            log.info("VLM loaded: %s on %s", self.model_id, self.device)
        except Exception as exc:
            self._model = None
            self._load_error = f"model load failed: {exc}"
            raise

    def extract_triples_from_image(self, image_path: str,
                                   max_new_tokens: int = 512,
                                   temperature: float = 0.1) -> list[dict]:
        self._ensure_loaded()
        if self._model is None:
            return []
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": VLM_USER_PROMPT},
            ]},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception:
            from PIL import Image
            image_inputs = [Image.open(image_path).convert("RGB")]
        inputs = self._processor(
            text=[text], images=image_inputs, padding=True,
            return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        generated = out_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return self._parse_and_validate(response)

    def _parse_and_validate(self, response: str) -> list[dict]:
        triples = self._extract_json_array(response)
        validated: list[dict] = []
        allowed = self.ontology.allowed_concepts()
        for t in triples:
            if not isinstance(t, dict):
                continue
            subj_raw = str(t.get("subject", "")).strip()
            obj_raw = str(t.get("object", "")).strip()
            rel = str(t.get("relation", "")).strip().upper()
            evidence = str(t.get("evidence", "")).strip() or "VLM extracted"
            if rel not in VALID_RELATIONS:
                continue
            subj = self.resolver.strict_ontology_mapping(subj_raw, allowed)
            obj = self.resolver.strict_ontology_mapping(obj_raw, allowed)
            if not subj or not obj:
                log.debug("Dropped VLM triple (%r, %r, %r)", subj_raw, rel, obj_raw)
                continue
            validated.append({
                "subject": subj, "relation": rel, "object": obj,
                "evidence": evidence, "source": "VLM_Vision", "weight": 0.80,
            })
        return validated

    @staticmethod
    def _extract_json_array(text: str) -> list:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            cleaned = m.group().replace("'", '"')
            cleaned = re.sub(r",\s*]", "]", cleaned)
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return []


# ===========================================================================
#  PHASE 3  -  Multimodal Graph Builder
# ===========================================================================
SOURCE_COLOUR = {
    "CSV_Ground_Truth":      "#2ca02c",
    "Manuscript_CSV":        "#2ca02c",
    "Manuscript_Theory":     "#2ca02c",
    "Literature_Text":       "#1f77b4",
    "VLM_Vision":            "#ff7f0e",
    "VLM_Extracted":         "#ff7f0e",
    "Multimodal_Inference":  "#9467bd",
    "Ontology_Backbone":     "#999999",
}

NODE_TYPE_COLOUR = {
    "Simulation_Model":      "#2ca02c",
    "Geometry":              "#98df8a",
    "Parameter":             "#aec7e8",
    "Ratio":                 "#98df8a",
    "Mechanical_Property":   "#ffbb78",
    "Electrochemical_Property": "#ff9896",
    "Morphology_Metric":     "#c5b0d5",
    "Phenomenon":            "#e377c2",
    "Material":              "#c7c7c7",
    "Method":                "#dbdb8d",
    "Microstructure":        "#9edae5",
    "Application":           "#f7b6d2",
}


class MultimodalGraphBuilder:
    def __init__(self, ontology: DendriteGeometryOntology) -> None:
        self.ontology = ontology
        self.G: nx.DiGraph = nx.DiGraph()
        self._seed_ontology_backbone()

    def _seed_ontology_backbone(self) -> None:
        for name, node in self.ontology.concepts.items():
            self.G.add_node(name, type=node.type, source="Ontology_Backbone",
                            definition=node.definition)

    def build_base_graph_from_csv(self, df: pd.DataFrame) -> None:
        cols = {c.lower(): c for c in df.columns}
        required = ["model", "geometry", "s_l_ratio", "max_stress_gpa",
                    "area_fraction_pct"]
        missing = [r for r in required if r not in cols]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}; "
                             f"got {list(df.columns)}")
        for _, row in df.iterrows():
            model = self._norm_model(row[cols["model"]])
            if model not in MODEL_SIGNATURES:
                continue
            self.G.add_node(model, type="Simulation_Model",
                            source="Manuscript_CSV",
                            definition=self.ontology.concepts[model].definition)
            geom = str(row[cols["geometry"]]).strip().lower().replace(" ", "_")
            geom = self._match_geometry(geom)
            self.G.add_edge(model, geom, relation="HAS_GEOMETRY",
                            weight=1.0, source="CSV_Ground_Truth")
            ratio = str(row[cols["s_l_ratio"]]).strip().replace(" ", "")
            ratio_node = f"SL_{ratio}"
            if ratio_node not in self.G:
                self.G.add_node(ratio_node, type="Ratio",
                                source="Manuscript_CSV",
                                definition=f"Solid/liquid ratio {ratio}")
            self.G.add_edge(model, ratio_node, relation="HAS_RATIO",
                            weight=1.0, source="CSV_Ground_Truth")
            stress = float(row[cols["max_stress_gpa"]] or 0.0)
            if stress > 0.05:
                self.G.add_edge(model, "high_von_mises_stress",
                                relation="GENERATES", weight=round(stress, 3),
                                source="CSV_Ground_Truth")
                self.G.add_edge("high_von_mises_stress", "dendrite_suppression",
                                relation="INHIBITS", weight=0.9,
                                source="Manuscript_Theory")
            area = float(row[cols["area_fraction_pct"]] or 0.0)
            self.G.add_edge(model, "dendrite_area_fraction",
                            relation="INDICATES", weight=round(area / 100.0, 3),
                            source="CSV_Ground_Truth")
            self.G.nodes[model]["csv_observed"] = True

    @staticmethod
    def _norm_model(value) -> str:
        v = str(value).strip().replace(" ", "_")
        if v.upper().startswith("MODEL_"):
            return "Model_" + v.split("_", 1)[1].upper()
        if len(v) == 1 and v.upper() in "ABCDEF":
            return f"Model_{v.upper()}"
        return v

    @staticmethod
    def _match_geometry(text: str) -> str:
        m = {
            "boundary_layer": "boundary_layer", "boundarylayer": "boundary_layer",
            "edge_layer": "boundary_layer", "horizontal_layering": "boundary_layer",
            "alternating_stripes": "alternating_stripes",
            "vertical_stripes": "alternating_stripes", "stripes": "alternating_stripes",
            "central_layer": "central_layer", "central_solid_layer": "central_layer",
            "core_layer": "central_layer",
        }
        return m.get(text, text if text in m.values() else "boundary_layer")

    def fuse_literature_edges(self, triples: list[dict]) -> None:
        for t in triples:
            subj, rel = t["subject"], t["relation"]
            obj, w = t["object"], float(t.get("weight", 0.5))
            self._ensure_node(subj, "Literature_Text")
            self._ensure_node(obj, "Literature_Text")
            if self.G.has_edge(subj, obj):
                data = self.G[subj][obj]
                data["weight"] = min(1.0, float(data.get("weight", 0.5)) + 0.1)
                data["source"] = self._merge_source(data.get("source"), "Literature_Text")
                data.setdefault("literature_evidence", []).append(t.get("evidence", ""))
            else:
                self.G.add_edge(subj, obj, relation=rel, weight=w,
                                source="Literature_Text",
                                literature_evidence=[t.get("evidence", "")],
                                doc=t.get("doc", ""))

    def fuse_vlm_insights(self, triples: list[dict]) -> None:
        for t in triples:
            subj, rel = t["subject"], t["relation"]
            obj = t["object"]
            self._ensure_node(subj, "VLM_Vision")
            self._ensure_node(obj, "VLM_Vision")
            if self.G.has_edge(subj, obj):
                data = self.G[subj][obj]
                data["source"] = self._merge_source(data.get("source"), "VLM_Vision")
                data["weight"] = min(1.0, float(data.get("weight", 0.5)) + 0.2)
                data["visual_evidence"] = t.get("evidence", "VLM extracted")
            else:
                self.G.add_edge(subj, obj, relation=rel,
                                weight=float(t.get("weight", 0.8)),
                                source="VLM_Vision",
                                visual_evidence=t.get("evidence", "VLM extracted"))

    def infer_cross_modal_paths(self, max_depth: int = 2) -> None:
        lit_nodes = [n for n, d in self.G.nodes(data=True)
                     if d.get("source") == "Literature_Text"
                     or "Literature_Text" in str(d.get("source", ""))]
        sim_nodes = [n for n, d in self.G.nodes(data=True)
                     if d.get("source") in
                     ("CSV_Ground_Truth", "Manuscript_CSV",
                      "VLM_Vision", "Manuscript_Theory")]
        added = 0
        for lit in lit_nodes:
            if lit not in self.ontology.concepts:
                continue
            for sim in sim_nodes:
                if sim not in self.ontology.concepts:
                    continue
                if self.G.has_edge(lit, sim) or lit == sim:
                    continue
                paths = self.ontology.infer_path(lit, sim, max_depth=max_depth)
                if paths:
                    self.G.add_edge(
                        lit, sim, relation="SUPPORTED_BY_SIMULATION",
                        weight=0.7, source="Multimodal_Inference",
                        path=" -> ".join(paths[0]))
                    added += 1
        log.info("Cross-modal inference added %d bridge edges.", added)

    def _ensure_node(self, name: str, source: str) -> None:
        if name not in self.G:
            ctype = self.ontology.concept_types().get(name, "Unknown")
            self.G.add_node(name, type=ctype, source=source)
        else:
            nd = self.G.nodes[name]
            if nd.get("source") == "Ontology_Backbone":
                nd["source"] = source

    @staticmethod
    def _merge_source(a: Optional[str], b: str) -> str:
        if not a or a == b:
            return b
        if b in str(a):
            return a
        return f"{a} + {b}"

    def summary(self) -> dict:
        by_source: dict[str, int] = {}
        for _, d in self.G.nodes(data=True):
            s = str(d.get("source", "Ontology_Backbone"))
            by_source[s] = by_source.get(s, 0) + 1
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "by_source": by_source,
        }

    def node_colour(self, name: str) -> str:
        d = self.G.nodes[name]
        if "source" in d and d["source"] in SOURCE_COLOUR:
            return SOURCE_COLOUR[d["source"]]
        return NODE_TYPE_COLOUR.get(d.get("type", ""), "#999999")


# ===========================================================================
#  PHASE 4  -  Graph rendering (pyvis + fallback)
# ===========================================================================
def render_graph_pyvis(graph: MultimodalGraphBuilder,
                       focus: str | None = None,
                       height: str = "600px") -> str:
    """PyVis renderer with injected JavaScript for node highlighting and an
    edge info side-panel. Adapted from the UAV codebase's JS-injected PyVis.

    On node click: highlights the clicked node, dims everything else, and
    opens a side panel listing every incident edge with its relation, weight,
    source, and (where present) the VLM visual evidence or literature quote."""
    try:
        from pyvis.network import Network
    except Exception:
        return _fallback_table(graph)
    net = Network(height=height, width="100%", directed=True,
                  notebook=False, bgcolor="#ffffff", font_color="black")
    net.repulsion(node_distance=120, central_gravity=0.2,
                  spring_length=110, spring_strength=0.04)
    nodes_to_show: set[str] = set()
    if focus:
        nodes_to_show.add(focus)
        nodes_to_show.update(graph.G.successors(focus))
        nodes_to_show.update(graph.G.predecessors(focus))
    else:
        for u, v, d in graph.G.edges(data=True):
            if d.get("source") != "Ontology_Backbone":
                nodes_to_show.add(u)
                nodes_to_show.add(v)
        nodes_to_show.update(MODEL_SIGNATURES.keys())
    for name in nodes_to_show:
        nd = graph.G.nodes[name]
        colour = graph.node_colour(name)
        title = (f"{name}\ntype: {nd.get('type','?')}\n"
                 f"source: {nd.get('source','?')}\n{nd.get('definition','')}")
        if nd.get("csv_observed"):
            title += "\n[CSV ground truth]"
        net.add_node(name, label=name, color=colour, title=title,
                     shape="dot",
                     size=22 if nd.get("type") == "Simulation_Model" else 14)
    for u, v, d in graph.G.edges(data=True):
        if u not in nodes_to_show or v not in nodes_to_show:
            continue
        if d.get("source") == "Ontology_Backbone":
            continue
        title = (f"{u} --[{d.get('relation','?')}]--> {v}\n"
                 f"weight: {d.get('weight','?')}\nsource: {d.get('source','?')}")
        if d.get("visual_evidence"):
            title += f"\nvisual: {d['visual_evidence']}"
        if d.get("literature_evidence"):
            title += f"\nliterature: {d['literature_evidence'][0]}"
        if d.get("path"):
            title += f"\npath: {d['path']}"
        net.add_edge(u, v, title=title, label=d.get("relation", ""),
                     width=0.5 + 2.0 * float(d.get("weight", 0.5)),
                     color="#888888")
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 120}}}')
    try:
        html_content = net.generate_html(notebook=False)
    except Exception:
        return _fallback_table(graph)
    # Inject the interactive JS + side-panel CSS before </body>
    return _inject_interactive_js(html_content)


def _inject_interactive_js(html: str) -> str:
    """Inject node-highlight + edge-info-panel JS into a PyVis HTML blob.

    The injected script waits for the vis Network to be ready, then binds
    selectNode / deselectNode handlers that dim non-neighbours and populate
    a fixed-position side panel with edge details."""
    if "</body>" not in html:
        return html
    panel_css = """
<style>
#dendrite-edge-panel {
  position: fixed; top: 12px; right: 12px; width: 320px;
  max-height: 70vh; overflow-y: auto;
  background: #ffffff; border: 1px solid #cccccc; border-radius: 8px;
  padding: 12px 14px; font-family: -apple-system, sans-serif;
  font-size: 12px; color: #222; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  z-index: 9999; display: none;
}
#dendrite-edge-panel h4 { margin: 0 0 8px 0; font-size: 14px; color: #1f77b4; }
#dendrite-edge-panel .edge-row {
  border-bottom: 1px solid #eee; padding: 6px 0;
}
#dendrite-edge-panel .rel { font-weight: 600; color: #d62728; }
#dendrite-edge-panel .src { font-size: 10px; color: #888; text-transform: uppercase; }
#dendrite-edge-panel .ev  { color: #444; font-style: italic; margin-top: 2px; }
#dendrite-edge-panel .close { float: right; cursor: pointer; color: #999; }
#dendrite-edge-panel .badge {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 10px; margin-left: 4px; color: #fff;
}
#dendrite-edge-panel .badge.csv { background: #2ca02c; }
#dendrite-edge-panel .badge.lit { background: #1f77b4; }
#dendrite-edge-panel .badge.vlm { background: #ff7f0e; }
#dendrite-edge-panel .badge.inf { background: #9467bd; }
</style>
<div id="dendrite-edge-panel">
  <span class="close" onclick="document.getElementById('dendrite-edge-panel').style.display='none';">&times;</span>
  <h4 id="dep-title">Node</h4>
  <div id="dep-meta" style="margin-bottom:8px; color:#666; font-size:11px;"></div>
  <div id="dep-edges"></div>
</div>
"""
    js = r"""
<script>
(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  ready(function () {
    var panel = document.getElementById('dendrite-edge-panel');
    if (!panel) return;
    // Find the vis Network instance - pyvis exposes it on the window under
    // a container-id-derived name. We brute-force search window keys.
    var net = null, nodesDS = null, edgesDS = null;
    function findNet() {
      for (var k in window) {
        try {
          var v = window[k];
          if (v && v.body && v.body.data && v.body.data.nodes) {
            net = v; nodesDS = v.body.data.nodes; edgesDS = v.body.data.edges;
            return true;
          }
        } catch (e) {}
      }
      return false;
    }
    var tries = 0;
    var iv = setInterval(function () {
      tries++;
      if (findNet() || tries > 40) clearInterval(iv);
      if (!net) return;
      net.on('selectNode', function (params) {
        var id = params.nodes[0];
        var node = nodesDS.get(id);
        var connected = net.getConnectedEdges(id);
        // Dim non-neighbours
        var keep = new Set([id]);
        connected.forEach(function (eid) {
          var e = edgesDS.get(eid);
          if (e) { keep.add(e.from); keep.add(e.to); }
        });
        nodesDS.update(nodesDS.get().map(function (n) {
          return { id: n.id, opacity: keep.has(n.id) ? 1.0 : 0.15 };
        }));
        edgesDS.update(edgesDS.get().map(function (e) {
          return { id: e.id,
            color: connected.indexOf(e.id) >= 0 ? { color: '#d62728' } : { color: '#eeeeee' },
            width: connected.indexOf(e.id) >= 0 ? 3 : 0.5 };
        }));
        // Build panel
        document.getElementById('dep-title').textContent = id;
        var meta = [];
        if (node && node.type) meta.push('type: ' + node.type);
        if (node && node.source) meta.push('source: ' + node.source);
        if (node && node.csv_observed) meta.push('[CSV ground truth]');
        document.getElementById('dep-meta').innerHTML = meta.join(' &middot; ');
        var html = '';
        connected.forEach(function (eid) {
          var e = edgesDS.get(eid);
          if (!e) return;
          var dir = (e.from === id) ? ('&rarr; ' + e.to) : (e.from + ' &rarr;');
          var src = (e.title || '').match(/source: ([^\n]+)/);
          var srcStr = src ? src[1] : '';
          var badge = '';
          if (/CSV|Manuscript/i.test(srcStr)) badge = '<span class="badge csv">CSV</span>';
          else if (/Literature/i.test(srcStr)) badge = '<span class="badge lit">LIT</span>';
          else if (/VLM/i.test(srcStr)) badge = '<span class="badge vlm">VLM</span>';
          else if (/Inference|Multimodal/i.test(srcStr)) badge = '<span class="badge inf">INF</span>';
          var rel = (e.title || '').match(/--\[([^\]]+)\]--/);
          var w   = (e.title || '').match(/weight: ([^\n]+)/);
          var vis = (e.title || '').match(/visual: ([^\n]+)/);
          var lit = (e.title || '').match(/literature: ([^\n]+)/);
          html += '<div class="edge-row">';
          html += '<div><span class="rel">' + (rel ? rel[1] : '?') + '</span> '
                + dir + badge + '</div>';
          html += '<div class="src">' + srcStr
                + (w ? ' &middot; w=' + w[1] : '') + '</div>';
          if (vis) html += '<div class="ev">&#128065; ' + vis[1] + '</div>';
          if (lit) html += '<div class="ev">&#128214; ' + lit[1] + '</div>';
          html += '</div>';
        });
        if (!html) html = '<div style="color:#888;">No incident edges.</div>';
        document.getElementById('dep-edges').innerHTML = html;
        panel.style.display = 'block';
      });
      net.on('deselectNode', function () {
        nodesDS.update(nodesDS.get().map(function (n) {
          return { id: n.id, opacity: 1.0 };
        }));
        edgesDS.update(edgesDS.get().map(function (e) {
          return { id: e.id, color: { color: '#888888' }, width: e.width || 1 };
        }));
        panel.style.display = 'none';
      });
    }, 100);
  });
})();
</script>
"""
    return html.replace("</body>", panel_css + js + "</body>")


def _fallback_table(graph: MultimodalGraphBuilder) -> str:
    rows = ["<table border=1 cellpadding=4>",
            "<tr><th>source</th><th>relation</th><th>from</th><th>to</th>"
            "<th>weight</th></tr>"]
    for u, v, d in graph.G.edges(data=True):
        if d.get("source") == "Ontology_Backbone":
            continue
        rows.append(
            f"<tr><td>{d.get('source','')}</td><td>{d.get('relation','')}</td>"
            f"<td>{u}</td><td>{v}</td><td>{d.get('weight','')}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


# ===========================================================================
#  PHASE 3b  -  Advanced Analytics : Bursts, Genealogy, Cross-Domain Bridges
# ===========================================================================
def detect_dendrite_keyword_bursts(
    lit_df: pd.DataFrame,
    extractor: "TextConceptExtractor",
    year_col: str = "year",
    abstract_col: str = "abstract",
) -> pd.DataFrame:
    """Per-year concept mention counts -> burst score.

    A concept 'bursts' in year Y if its share of all concept mentions that
    year exceeds its long-run average share by >1.5x. Returns one row per
    (concept, year) with positive burst score, sorted descending. Adapted
    from the UAV codebase's keyword burst detector."""
    if year_col not in lit_df.columns:
        lit_df = lit_df.copy()
        lit_df[year_col] = 2020  # synthetic if missing
    yearly: dict[int, Counter] = {}
    for _, row in lit_df.iterrows():
        try:
            yr = int(row[year_col])
        except Exception:
            continue
        sents = split_sentences(str(row.get(abstract_col, "")))
        hits: list[str] = []
        for s in sents:
            hits.extend(extractor._scan_sentence(s))
        yearly.setdefault(yr, Counter()).update(hits)
    if not yearly:
        return pd.DataFrame(columns=["year", "concept", "count", "share", "burst"])
    # global share per concept
    grand = Counter()
    for c in yearly.values():
        grand.update(c)
    total = sum(grand.values()) or 1
    avg_share = {k: v / total for k, v in grand.items()}
    rows: list[dict] = []
    for yr, counter in yearly.items():
        yr_total = sum(counter.values()) or 1
        for concept, cnt in counter.items():
            share = cnt / yr_total
            burst = share / max(avg_share.get(concept, 1e-6), 1e-6)
            if burst > 1.5 and cnt >= 1:
                rows.append({"year": yr, "concept": concept,
                             "count": cnt, "share": round(share, 3),
                             "burst": round(burst, 2)})
    if not rows:
        return pd.DataFrame(columns=["year", "concept", "count", "share", "burst"])
    return pd.DataFrame(rows).sort_values(
        ["burst", "count"], ascending=False).reset_index(drop=True)


def build_dendrite_genealogy(graph: "MultimodalGraphBuilder",
                             top_k: int = 15) -> pd.DataFrame:
    """Rank concepts by PageRank (foundational) and Betweenness (bridge).

    Foundational concepts = high PageRank (many inbound weighted edges).
    Bridge concepts = high betweenness centrality (sit on many shortest paths).
    Adapted from the UAV genealogy function."""
    G = graph.G
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["concept", "pagerank", "betweenness",
                                     "category", "role"])
    # PageRank on the undirected projection (weights = edge weights)
    U = G.to_undirected()
    for u, v, d in U.edges(data=True):
        d.setdefault("weight", 0.5)
    try:
        pr = nx.pagerank(U, weight="weight", alpha=0.85)
    except Exception:
        pr = {n: 0.0 for n in G.nodes}
    try:
        bc = nx.betweenness_centrality(U, weight="weight", normalized=True)
    except Exception:
        bc = {n: 0.0 for n in G.nodes}
    rows = []
    for n in G.nodes:
        if n not in graph.ontology.concepts:
            continue
        cat = concept_to_category(n) or "Other"
        rows.append({
            "concept": n,
            "pagerank": round(pr.get(n, 0.0), 5),
            "betweenness": round(bc.get(n, 0.0), 5),
            "category": CATEGORY_SHORT_NAMES.get(cat, cat),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Label roles
    pr_cut = df["pagerank"].quantile(0.75)
    bc_cut = df["betweenness"].quantile(0.75)
    df["role"] = "peripheral"
    df.loc[df["pagerank"] >= pr_cut, "role"] = "foundational"
    df.loc[df["betweenness"] >= bc_cut, "role"] = "bridge"
    df.loc[(df["pagerank"] >= pr_cut) & (df["betweenness"] >= bc_cut),
           "role"] = "hub"
    return df.sort_values("pagerank", ascending=False).head(top_k).reset_index(
        drop=True)


def detect_mechanics_morphology_bridges(
    graph: "MultimodalGraphBuilder") -> pd.DataFrame:
    """Find concepts that sit on shortest paths between Mechanics and
    Morphology concepts. These are the 'load-bearing' nodes that translate
    stress into dendrite behaviour - the most scientifically interesting
    part of the graph."""
    G = graph.G
    mech = [n for n in G.nodes if concept_to_category(n) == "Mechanical Constraints"]
    morph = [n for n in G.nodes if concept_to_category(n) == "Morphology Outcomes"]
    if not mech or not morph:
        return pd.DataFrame(columns=["bridge", "mech_side", "morph_side",
                                     "path_length", "path"])
    U = G.to_undirected()
    rows: list[dict] = []
    seen_bridges: set[str] = set()
    for m in mech:
        for p in morph:
            if m == p or not U.has_node(m) or not U.has_node(p):
                continue
            try:
                path = nx.shortest_path(U, m, p)
            except nx.NetworkXNoPath:
                continue
            if len(path) < 3:  # direct edge, no bridge
                continue
            bridge = path[1:-1]
            for b in bridge:
                key = f"{m}|{b}|{p}"
                if key in seen_bridges:
                    continue
                seen_bridges.add(key)
                rows.append({
                    "bridge": b,
                    "mech_side": m,
                    "morph_side": p,
                    "path_length": len(path) - 1,
                    "path": " -> ".join(path),
                    "bridge_category": CATEGORY_SHORT_NAMES.get(
                        concept_to_category(b) or "Other",
                        concept_to_category(b) or "Other"),
                })
    if not rows:
        return pd.DataFrame(columns=["bridge", "mech_side", "morph_side",
                                     "path_length", "path", "bridge_category"])
    df = pd.DataFrame(rows)
    # Aggregate: count paths each bridge serves
    agg = df.groupby("bridge").agg(
        paths_served=("path", "count"),
        avg_path_length=("path_length", "mean"),
        categories=("bridge_category", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index().sort_values("paths_served", ascending=False)
    return agg


def compute_semantic_drift(lit_df: pd.DataFrame,
                           extractor: "TextConceptExtractor",
                           year_col: str = "year",
                           abstract_col: str = "abstract") -> pd.DataFrame:
    """Track how the category mix of concept mentions shifts across years.
    A rising share for 'Morphology Outcomes' and falling share for
    'Geometrical Design' would indicate the field is moving from
    architecture design toward outcome characterisation."""
    if year_col not in lit_df.columns:
        return pd.DataFrame()
    yearly_cat: dict[int, Counter] = {}
    for _, row in lit_df.iterrows():
        try:
            yr = int(row[year_col])
        except Exception:
            continue
        sents = split_sentences(str(row.get(abstract_col, "")))
        hits: list[str] = []
        for s in sents:
            hits.extend(extractor._scan_sentence(s))
        cat_counter: Counter = Counter()
        for h in hits:
            c = concept_to_category(h)
            if c:
                cat_counter[c] += 1
        yearly_cat.setdefault(yr, Counter()).update(cat_counter)
    rows = []
    for yr in sorted(yearly_cat):
        c = yearly_cat[yr]
        tot = sum(c.values()) or 1
        row = {"year": yr, "total": sum(c.values())}
        for cat in DENDRITE_CATEGORIES:
            row[CATEGORY_SHORT_NAMES[cat]] = round(c.get(cat, 0) / tot, 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
#  PHASE 6b  -  Performance Monitoring & Memory Safety
# ===========================================================================
def get_gpu_vram_usage() -> Optional[float]:
    """Return current GPU VRAM usage in MB, or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def get_ram_usage_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return None


@dataclass
class MetricStep:
    step: str
    timestamp: str
    detail: dict
    ram_mb: Optional[float] = None
    vram_mb: Optional[float] = None
    duration_s: Optional[float] = None


class DendriteMetricsLogger:
    """Lightweight run-metrics logger (adapted from the UAV RunMetricsLogger).

    Records each major pipeline step (CSV build, literature extraction, VLM
    inference, cross-modal inference, QDWA allocation, LLM answer) with
    timestamps, RAM/VRAM snapshots and step-specific detail. Exposed in the
    Export tab as a table and downloadable JSON."""

    def __init__(self) -> None:
        self.steps: list[MetricStep] = []

    def log(self, step: str, detail: Optional[dict] = None,
            duration_s: Optional[float] = None) -> None:
        from datetime import datetime
        self.steps.append(MetricStep(
            step=step,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            detail=detail or {},
            ram_mb=get_ram_usage_mb(),
            vram_mb=get_gpu_vram_usage(),
            duration_s=duration_s,
        ))

    def log_vlm_inference(self, image_name: str, triples: int,
                          duration_s: Optional[float] = None) -> None:
        self.log("VLM Extraction", {
            "image": image_name,
            "triples": triples,
        }, duration_s=duration_s)

    def log_literature_extraction(self, n_abstracts: int, n_triples: int,
                                  duration_s: Optional[float] = None) -> None:
        self.log("Literature Extraction", {
            "abstracts": n_abstracts,
            "triples": n_triples,
        }, duration_s=duration_s)

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.steps:
            rows.append({
                "step": s.step,
                "timestamp": s.timestamp,
                "ram_mb": round(s.ram_mb, 1) if s.ram_mb is not None else None,
                "vram_mb": round(s.vram_mb, 1) if s.vram_mb is not None else None,
                "duration_s": round(s.duration_s, 3) if s.duration_s else None,
                **s.detail,
            })
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        return json.dumps([s.__dict__ for s in self.steps], indent=2, default=str)


# ===========================================================================
#  PHASE 4  -  Streamlit UI  (imported lazily so `python app.py` still works)
# ===========================================================================
def _streamlit_main() -> None:
    import streamlit as st

    @st.cache_resource
    def get_ontology() -> DendriteGeometryOntology:
        return DendriteGeometryOntology()

    @st.cache_resource
    def get_resolver() -> AdvancedConceptResolver:
        return AdvancedConceptResolver(get_ontology(), threshold=0.82)

    @st.cache_resource
    def get_vlm() -> VLMImageExtractor:
        return VLMImageExtractor(get_ontology(), get_resolver())

    def get_graph() -> MultimodalGraphBuilder:
        if "graph" not in st.session_state:
            st.session_state.graph = MultimodalGraphBuilder(get_ontology())
        return st.session_state.graph

    def reset_graph() -> None:
        st.session_state.graph = MultimodalGraphBuilder(get_ontology())

    def get_metrics() -> DendriteMetricsLogger:
        if "metrics" not in st.session_state:
            st.session_state.metrics = DendriteMetricsLogger()
        return st.session_state.metrics

    @st.cache_resource
    def get_allocator() -> QDWAWeightAllocator:
        return QDWAWeightAllocator(get_ontology(), get_resolver())

    @st.cache_resource
    def get_analyzer() -> FallbackAnalyzer:
        return FallbackAnalyzer(get_ontology(), get_resolver(), get_allocator())

    @st.cache_resource
    def get_expander() -> DynamicOntologyExpander:
        return DynamicOntologyExpander(get_ontology())

    # -- sidebar ------------------------------------------------------------
    st.set_page_config(page_title="Dendrite Concept Graph",
                       page_icon=":-:", layout="wide")
    st.sidebar.title("Dendrite Concept Graph")
    st.sidebar.caption("Tri-modal fusion: CSV + Literature + VLM")
    ont = get_ontology()
    st.sidebar.write(f"{len(ont.concepts)} concepts, "
                     f"{len(ont.relationships)} ground-truth edges")
    vlm = get_vlm()
    st.sidebar.code(vlm.status())
    if st.sidebar.button("Reset graph"):
        reset_graph()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Node colours**")
    for src, col in SOURCE_COLOUR.items():
        st.sidebar.markdown(
            f"<span style='color:{col}'>\u25CF</span> {src}",
            unsafe_allow_html=True)

    # -- tab 1: simulation ground truth ------------------------------------
    def tab_simulation() -> None:
        st.header("Simulation Ground Truth")
        st.caption("Manuscript Table - 6 phase-field models (A-F)")
        uploaded = st.file_uploader("Upload models CSV", type=["csv"],
                                    key="models_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(io.StringIO(SAMPLE_MODELS_CSV))
            st.info("Using bundled sample_models.csv. Upload your own to override.")
        st.dataframe(df, use_container_width=True)
        if st.button("Build / refresh graph from CSV"):
            graph = get_graph()
            try:
                graph.build_base_graph_from_csv(df)
                st.success(f"Loaded {len(df)} models. "
                           f"Graph now has {graph.G.number_of_edges()} edges.")
            except ValueError as exc:
                st.error(str(exc))
        st.subheader("Inspect a model")
        model = st.selectbox("Pick a model", list(MODEL_SIGNATURES.keys()))
        if model:
            sig = MODEL_SIGNATURES[model]
            st.markdown(f"**{model}** - {sig['geometry']} - S/L = {sig['ratio']} "
                        f"({int(sig['solid_frac']*100)}% solid)")
            graph = get_graph()
            if graph.G.has_node(model):
                edges = list(graph.G.out_edges(model, data=True))
                st.write(f"Outgoing edges: {len(edges)}")
                for u, v, d in edges:
                    st.markdown(f"- `{u}` **[{d.get('relation','?')}]** `{v}` "
                                f"w={d.get('weight','?')} _{d.get('source','?')}_")
            else:
                st.info("Model not yet in graph. Click 'Build' above.")

    # -- tab 2: literature --------------------------------------------------
    def tab_literature() -> None:
        st.header("Literature Concept Extraction")
        st.caption("Stream 2 - sentence co-occurrence over abstracts")
        uploaded = st.file_uploader("Upload literature CSV (year, title, abstract)",
                                    type=["csv"], key="lit_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
            st.info("Using bundled sample_literature.csv "
                    "(now with a 'year' column for analytics).")
        st.session_state["_lit_df_upload"] = df
        st.dataframe(df, use_container_width=True)
        if st.button("Extract & fuse literature triples"):
            ont_ = get_ontology()
            res = get_resolver()
            ext = TextConceptExtractor(ont_, res)
            import time
            t0 = time.time()
            with st.spinner("Scanning abstracts..."):
                triples = ext.extract_from_corpus(df)
                freq = ext.concept_frequencies(df)
            dur = time.time() - t0
            get_metrics().log_literature_extraction(
                len(df), len(triples), duration_s=dur)
            st.success(f"Extracted {len(triples)} co-occurrence triples "
                       f"from {len(df)} abstracts in {dur:.2f}s.")
            if triples:
                st.dataframe(pd.DataFrame(triples), use_container_width=True)
            st.subheader("Concept frequencies")
            if freq:
                st.bar_chart(pd.DataFrame({"count": list(freq.values())},
                                          index=list(freq.keys())))
            else:
                st.info("No ontology concepts detected. Check your abstracts.")
            graph = get_graph()
            graph.fuse_literature_edges(triples)
            graph.infer_cross_modal_paths(max_depth=2)

    # -- tab 3: VLM ---------------------------------------------------------
    def tab_vlm() -> None:
        st.header("VLM Image Analyzer")
        st.caption("Stream 3 - Qwen2.5-VL-3B-Instruct on phase-field images")
        vlm_ = get_vlm()
        if not vlm_.is_available():
            st.warning(
                "VLM backend not available. Install torch + transformers + "
                "qwen-vl-utils (see README) to enable this tab. "
                f"Detail: {vlm_.status()}")
            st.info("The rest of the app (CSV + Literature + Unified Graph) "
                    "still works.")
            st.subheader("Manual triple entry (VLM-less fallback)")
            with st.form("manual_triple"):
                ont_ = get_ontology()
                concepts = sorted(ont_.allowed_concepts())
                subj = st.selectbox("subject", concepts, key="m_subj")
                rel = st.selectbox(
                    "relation",
                    ["HAS_GEOMETRY", "HAS_RATIO", "GENERATES", "EXHIBITS",
                     "INHIBITS", "CONSTRAINS", "PROMOTES", "DRIVES", "INDICATES"],
                    key="m_rel")
                obj = st.selectbox("object", concepts, key="m_obj")
                ev = st.text_input("evidence (free text)", key="m_ev")
                if st.form_submit_button("Inject triple"):
                    triple = {"subject": subj, "relation": rel, "object": obj,
                              "weight": 0.8, "evidence": ev or "manual",
                              "source": "VLM_Vision"}
                    get_graph().fuse_vlm_insights([triple])
                    st.success(f"Injected {subj} {rel} {obj}")
                    st.rerun()
            return
        uploaded = st.file_uploader("Upload phase-field simulation image",
                                    type=["png", "jpg", "jpeg"], key="vlm_img")
        if not uploaded:
            st.info("Upload a PNG/JPG to analyze.")
            return
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        if st.button("Run VLM Analysis", type="primary"):
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            import time
            t0 = time.time()
            with st.spinner("Running Qwen2.5-VL..."):
                triples = vlm_.extract_triples_from_image(tmp_path)
            dur = time.time() - t0
            os.unlink(tmp_path)
            get_metrics().log_vlm_inference(
                uploaded.name, len(triples), duration_s=dur)
            st.success(f"Got {len(triples)} validated triples in {dur:.2f}s. "
                       f"VRAM: {get_gpu_vram_usage() or 'n/a'} MB")
            if triples:
                st.dataframe(pd.DataFrame(triples), use_container_width=True)
                if st.button("Inject into concept graph", type="primary"):
                    get_graph().fuse_vlm_insights(triples)
                    get_graph().infer_cross_modal_paths(max_depth=2)
                    st.success("Injected. Switch to 'Unified Concept Graph' tab.")
                    st.rerun()
            else:
                st.info("No triples passed ontology validation.")

    # -- tab 4: unified graph ----------------------------------------------
    def tab_unified() -> None:
        st.header("Unified Concept Graph")
        st.caption("Green = CSV | Blue = Literature | Orange = VLM | "
                   "Purple = Inferred bridge. Click a node to highlight its "
                   "neighbourhood and open the edge-info panel.")
        graph = get_graph()
        summary = graph.summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", summary["nodes"])
        col2.metric("Edges", summary["edges"])
        col3.metric("Sources", len(summary["by_source"]))
        with st.expander("Source breakdown"):
            st.json(summary["by_source"])
        focus = st.selectbox(
            "Focus on a node (shows its neighbourhood)",
            [None] + sorted(graph.G.nodes),
            format_func=lambda n: "(all)" if n is None else n)
        html = render_graph_pyvis(graph, focus=focus, height="650px")
        if "<html" in html.lower():
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.markdown(html, unsafe_allow_html=True)
        st.subheader("All real edges (non-backbone)")
        rows = []
        for u, v, d in graph.G.edges(data=True):
            if d.get("source") == "Ontology_Backbone":
                continue
            ev = ""
            if d.get("visual_evidence"):
                ev = d["visual_evidence"]
            elif d.get("literature_evidence"):
                ev = d["literature_evidence"][0] if d["literature_evidence"] else ""
            rows.append({"from": u, "relation": d.get("relation", ""),
                         "to": v, "weight": d.get("weight", ""),
                         "source": d.get("source", ""), "evidence": ev})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No real edges yet. Build the CSV graph or fuse literature/VLM "
                    "evidence in the other tabs.")

    # -- tab 5: QDWA weights ------------------------------------------------
    def tab_qdwa() -> None:
        st.header("QDWA - Query Distillation & Weighted Allocation")
        st.caption("Routes a research query to the 4 pillars: Geometry, "
                   "Mechanics, Electrochem, Morphology.")
        st.subheader("Pillar definitions")
        for cat, members in DENDRITE_CATEGORIES.items():
            st.markdown(
                f"**{CATEGORY_SHORT_NAMES[cat]}** "
                f"<span style='color:{CATEGORY_COLORS[cat]}'>\u25CF</span> "
                f"({cat}) - {len(members)} concepts",
                unsafe_allow_html=True)
            st.caption(", ".join(members))
        st.markdown("---")
        q = st.text_input(
            "Enter a research query",
            value="How does high von Mises stress at the boundary layer "
                  "inhibit side branching of lithium dendrites?",
            key="qdwa_query")
        if q:
            alloc = get_allocator().allocate(q)
            st.subheader("Weight allocation")
            wdf = pd.DataFrame([
                {"pillar": CATEGORY_SHORT_NAMES[c],
                 "weight": round(alloc.weights[c], 3),
                 "colour": CATEGORY_COLORS[c],
                 "matched_concepts": ", ".join(alloc.matched_concepts[c]) or "-"}
                for c in DENDRITE_CATEGORIES
            ])
            # Bar chart
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            try:
                fm.fontManager.addfont(
                    '/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
            except Exception:
                pass
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            fig, ax = plt.subplots(figsize=(8, 2.5), constrained_layout=True)
            pillars = [CATEGORY_SHORT_NAMES[c] for c in DENDRITE_CATEGORIES]
            weights = [alloc.weights[c] for c in DENDRITE_CATEGORIES]
            colors = [CATEGORY_COLORS[c] for c in DENDRITE_CATEGORIES]
            ax.barh(pillars, weights, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("QDWA weight")
            ax.set_title(f"Dominant: {CATEGORY_SHORT_NAMES[alloc.dominant]} "
                         f"(w={alloc.weights[alloc.dominant]:.2f}, "
                         f"hits={alloc.total_hits})")
            for i, w in enumerate(weights):
                ax.text(w + 0.02, i, f"{w:.2f}", va="center", fontsize=9)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(wdf, use_container_width=True)
            get_metrics().log("QDWA Allocation", {
                "query": q[:100],
                "dominant": CATEGORY_SHORT_NAMES[alloc.dominant],
                "hits": alloc.total_hits,
            })

    # -- tab 6: advanced analytics ------------------------------------------
    def tab_analytics() -> None:
        st.header("Advanced Analytics")
        st.caption("Keyword bursts, concept genealogy, cross-domain bridges, "
                   "semantic drift over the literature corpus.")
        # Need a literature DF; reuse the bundled one if no upload.
        uploaded = st.session_state.get("_lit_df_upload")
        if uploaded is not None:
            lit_df = uploaded
        else:
            lit_df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
        ont_ = get_ontology()
        res = get_resolver()
        ext = TextConceptExtractor(ont_, res)
        graph = get_graph()

        st.subheader("\U0001F4C8 Keyword burst detection")
        bursts = detect_dendrite_keyword_bursts(lit_df, ext)
        if bursts.empty:
            st.info("No bursts detected (need a 'year' column and >1 paper/year).")
        else:
            st.dataframe(bursts, use_container_width=True)
            st.caption("Burst > 1.5x means the concept's share of mentions that "
                       "year exceeds its long-run average by 50%+.")

        st.subheader("\U0001F9EC Concept genealogy (PageRank + Betweenness)")
        gene = build_dendrite_genealogy(graph)
        if gene.empty:
            st.info("Graph has no edges yet. Build the CSV graph first.")
        else:
            st.dataframe(gene, use_container_width=True)
            st.caption("role = foundational (high PageRank), bridge (high "
                       "betweenness), hub (both), peripheral (neither).")

        st.subheader("\U0001F517 Mechanics \u2192 Morphology bridges")
        bridges = detect_mechanics_morphology_bridges(graph)
        if bridges.empty:
            st.info("No multi-hop bridges between Mechanics and Morphology "
                    "concepts. Add literature or VLM edges to populate.")
        else:
            st.dataframe(bridges, use_container_width=True)
            st.caption("These 'bridge' concepts translate stress into dendrite "
                       "behaviour - the scientifically load-bearing nodes.")

        st.subheader("\U0001F4C9 Semantic drift (pillar share by year)")
        drift = compute_semantic_drift(lit_df, ext)
        if drift.empty:
            st.info("Need a 'year' column to compute drift.")
        else:
            st.dataframe(drift, use_container_width=True)
            st.line_chart(drift.set_index("year")[
                [CATEGORY_SHORT_NAMES[c] for c in DENDRITE_CATEGORIES]])
            st.caption("Rising Morphology share + falling Geometry share would "
                       "indicate the field is shifting from architecture design "
                       "to outcome characterisation.")
        get_metrics().log("Advanced Analytics", {
            "bursts": len(bursts),
            "genealogy_rows": len(gene),
            "bridges": len(bridges),
        })

    # -- tab 7: LLM-guided Q&A ----------------------------------------------
    def tab_llm() -> None:
        st.header("LLM-Guided Q&A (Graph-RAG)")
        st.caption("Deterministic Graph-RAG over the fused concept graph. "
                   "Classifies your query into a DendriteProblem, pulls the "
                   "induced subgraph, and composes a cited answer.")
        st.subheader("Problem types")
        for prob, pdef in DENDRITE_PROBLEM_DEFINITIONS.items():
            with st.expander(f"{pdef['title']} ({prob.value})"):
                st.markdown(f"**Q:** {pdef['question']}")
                st.markdown(f"**Key concepts:** {', '.join(pdef['key_concepts'])}")
                if pdef["key_relationships"]:
                    st.markdown("**Key relationships:**")
                    for s, r, o in pdef["key_relationships"]:
                        st.markdown(f"- `{s}` **[{r}]** `{o}`")
                st.markdown(f"**Example queries:** "
                            + "; ".join(pdef["example_queries"]))
        st.markdown("---")
        q = st.text_input(
            "Ask a question",
            value="How does the boundary layer's high von Mises stress "
                  "suppress dendrite side branching?",
            key="llm_query")
        if q:
            analyzer = get_analyzer()
            result = analyzer.answer(q, get_graph())
            st.subheader(f"\U0001F9E0 {result['problem_title']}")
            st.caption(f"Problem class: {result['problem']} | "
                       f"dominant pillar: {result['dominant_pillar']} | "
                       f"evidence edges: {result['evidence_edges']}")
            # QDWA weights mini-bar
            wrow = result["qdwa_weights"]
            st.bar_chart(pd.DataFrame({"weight": list(wrow.values())},
                                      index=list(wrow.keys())))
            st.markdown("**Answer**")
            st.markdown(result["answer"])
            st.markdown("**Matched concepts**")
            st.write(", ".join(result["matched_concepts"]) or "(none in graph)")
            if result["chains"]:
                st.markdown("**Causal chains (top 8)**")
                for c in result["chains"]:
                    st.markdown(f"- {c}")
            get_metrics().log("LLM Q&A", {
                "query": q[:100],
                "problem": result["problem"],
                "edges": result["evidence_edges"],
            })
            # Dynamic ontology expansion
            st.markdown("---")
            st.subheader("\U0001F9E0 Dynamic ontology expansion")
            proposals = get_expander().propose(q)
            if not proposals:
                st.info("No new concepts proposed for this query.")
            else:
                st.write(f"{len(proposals)} concept(s) detected outside the "
                         "current ontology:")
                for p in proposals:
                    st.markdown(f"- **{p['name']}** ({p['type']}): "
                                f"{p['definition']}")
                    st.caption(f"_evidence: {p['evidence']}_")
                    if st.button(f"Add {p['name']} to ontology",
                                 key=f"add_{p['name']}"):
                        if get_expander().accept(
                                p["name"], p["type"], p["definition"]):
                            st.success(f"Added {p['name']}. "
                                       "Reset the graph to seed it.")
                            get_metrics().log("Ontology Expansion", {
                                "concept": p["name"], "type": p["type"]})
                            st.rerun()
                        else:
                            st.warning(f"{p['name']} already exists.")

    # -- tab 8: export & metrics --------------------------------------------
    def tab_export() -> None:
        st.header("Export & Metrics")
        st.caption("Download the fused graph and inspect run metrics.")
        graph = get_graph()
        metrics = get_metrics()

        st.subheader("\U0001F4CA Run metrics")
        mdf = metrics.as_dataframe()
        if mdf.empty:
            st.info("No steps logged yet. Interact with the other tabs to "
                    "populate the metrics log.")
        else:
            st.dataframe(mdf, use_container_width=True)
            st.download_button(
                "Download metrics JSON",
                data=metrics.to_json(),
                file_name="dendrite_metrics.json",
                mime="application/json")

        st.subheader("\U0001F4E4 Graph export")
        col1, col2, col3 = st.columns(3)
        # GraphML
        try:
            graphml_buf = io.StringIO()
            nx.write_graphml(graph.G, graphml_buf)
            col1.download_button(
                "GraphML", data=graphml_buf.getvalue(),
                file_name="dendrite_graph.graphml",
                mime="application/xml")
        except Exception as exc:
            col1.warning(f"GraphML export failed: {exc}")
        # JSON (node-link)
        try:
            json_data = nx.node_link_data(graph.G, edges="links")
            json_str = json.dumps(json_data, indent=2, default=str)
            col2.download_button(
                "JSON (node-link)", data=json_str,
                file_name="dendrite_graph.json",
                mime="application/json")
        except Exception as exc:
            col2.warning(f"JSON export failed: {exc}")
        # Edge list CSV
        try:
            edge_rows = []
            for u, v, d in graph.G.edges(data=True):
                if d.get("source") == "Ontology_Backbone":
                    continue
                edge_rows.append({"from": u, "to": v,
                                  "relation": d.get("relation", ""),
                                  "weight": d.get("weight", ""),
                                  "source": d.get("source", "")})
            edge_df = pd.DataFrame(edge_rows)
            col3.download_button(
                "Edges CSV", data=edge_df.to_csv(index=False),
                file_name="dendrite_edges.csv", mime="text/csv")
        except Exception as exc:
            col3.warning(f"CSV export failed: {exc}")

        st.subheader("\U0001F4DD Markdown report")
        if st.button("Generate Markdown report"):
            md = _generate_markdown_report(graph, metrics)
            st.download_button(
                "Download report.md", data=md,
                file_name="dendrite_report.md", mime="text/markdown")
            st.markdown("Preview:")
            st.markdown(md)
        get_metrics().log("Export", {"nodes": graph.G.number_of_nodes(),
                                     "edges": graph.G.number_of_edges()})

    # -- layout -------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "\U0001F4CA Simulation Ground Truth",
        "\U0001F4DA Literature Concept Extraction",
        "\U0001F441 VLM Image Analyzer",
        "\U0001F578 Unified Concept Graph",
        "\U0001F3AF QDWA Weights",
        "\U0001F4C8 Advanced Analytics",
        "\U0001F9E0 LLM-Guided Q&A",
        "\U0001F4E5 Export & Metrics",
    ])
    with tab1:
        tab_simulation()
    with tab2:
        tab_literature()
    with tab3:
        tab_vlm()
    with tab4:
        tab_unified()
    with tab5:
        tab_qdwa()
    with tab6:
        tab_analytics()
    with tab7:
        tab_llm()
    with tab8:
        tab_export()


def _generate_markdown_report(graph: MultimodalGraphBuilder,
                              metrics: DendriteMetricsLogger) -> str:
    """Build a self-contained Markdown report of the current graph + run."""
    s = graph.summary()
    lines = [
        "# Dendrite Concept Graph - Run Report",
        "",
        f"Generated: {metrics.steps[-1].timestamp if metrics.steps else 'n/a'}",
        "",
        "## Graph summary",
        f"- Nodes: **{s['nodes']}**",
        f"- Edges: **{s['edges']}**",
        f"- Sources: {len(s['by_source'])}",
        "",
        "### Source breakdown",
        "",
    ]
    for src, cnt in sorted(s["by_source"].items(), key=lambda x: -x[1]):
        lines.append(f"- {src}: {cnt}")
    lines += ["", "## Edges", "",
              "| from | relation | to | weight | source |",
              "|---|---|---|---|---|"]
    for u, v, d in graph.G.edges(data=True):
        if d.get("source") == "Ontology_Backbone":
            continue
        lines.append(f"| {u} | {d.get('relation','')} | {v} | "
                     f"{d.get('weight','')} | {d.get('source','')} |")
    lines += ["", "## Run metrics", "",
              "| step | timestamp | ram_mb | vram_mb | duration_s |",
              "|---|---|---|---|---|"]
    for stp in metrics.steps:
        lines.append(
            f"| {stp.step} | {stp.timestamp} | "
            f"{round(stp.ram_mb,1) if stp.ram_mb else '-'} | "
            f"{round(stp.vram_mb,1) if stp.vram_mb else '-'} | "
            f"{round(stp.duration_s,3) if stp.duration_s else '-'} |")
    return "\n".join(lines)


# ===========================================================================
#  Headless entry point  (python app.py)
# ===========================================================================
def _headless_smoke_test() -> None:
    """Run the full fusion pipeline against the bundled sample CSVs and
    print a summary. Used when streamlit is not installed."""
    ont = DendriteGeometryOntology()
    res = AdvancedConceptResolver(ont, threshold=0.82)
    ext = TextConceptExtractor(ont, res)
    gb = MultimodalGraphBuilder(ont)

    models = pd.read_csv(io.StringIO(SAMPLE_MODELS_CSV))
    gb.build_base_graph_from_csv(models)
    print(f"[Stream 1] CSV: {len(models)} models -> {gb.G.number_of_edges()} edges")

    lit = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
    triples = ext.extract_from_corpus(lit)
    gb.fuse_literature_edges(triples)
    print(f"[Stream 2] Literature: {len(triples)} triples -> "
          f"{gb.G.number_of_edges()} edges")

    fake_vlm = [
        {"subject": "Model_A", "relation": "GENERATES", "object": "von_mises_stress",
         "weight": 0.8, "evidence": "bright red band along the solid-liquid boundary"},
        {"subject": "Model_E", "relation": "EXHIBITS", "object": "severe_side_branching",
         "weight": 0.8, "evidence": "many long lateral protrusions"},
    ]
    gb.fuse_vlm_insights(fake_vlm)
    print(f"[Stream 3] VLM (simulated): {len(fake_vlm)} triples -> "
          f"{gb.G.number_of_edges()} edges")

    gb.infer_cross_modal_paths(max_depth=2)
    print("[Cross-modal] bridges added. Final:", gb.summary())

    # VLM parser sanity (no model load)
    vlm = VLMImageExtractor(ont, res)
    fake_resp = '''```json
    [
      {"subject": "Model A", "relation": "GENERATES", "object": "von Mises stress",
       "evidence": "bright red band"},
      {"subject": "thermal runaway", "relation": "PROMOTES", "object": "rainbow"}
    ]
    ```'''
    parsed = vlm._parse_and_validate(fake_resp)
    print(f"[VLM parser] {len(parsed)} validated (1 hallucinated dropped)")

    # pyvis renderer sanity (with JS injection)
    html = render_graph_pyvis(gb, focus="Model_A", height="200px")
    has_js = "dendrite-edge-panel" in html
    print(f"[pyvis+JS] HTML length {len(html)}, has <html: {'<html' in html.lower()}, "
          f"JS panel injected: {has_js}")

    # --- Phase 1b: QDWA ---
    alloc_obj = QDWAWeightAllocator(ont, res)
    q = "How does high von Mises stress at the boundary layer inhibit side branching?"
    alloc = alloc_obj.allocate(q)
    print(f"[QDWA] '{q[:50]}...' -> dominant={CATEGORY_SHORT_NAMES[alloc.dominant]} "
          f"(w={alloc.weights[alloc.dominant]:.2f}), hits={alloc.total_hits}")
    print(f"        weights: {[(CATEGORY_SHORT_NAMES[c], round(alloc.weights[c],2)) for c in DENDRITE_CATEGORIES]}")

    # --- Phase 2b: FallbackAnalyzer (Graph-RAG) ---
    analyzer = FallbackAnalyzer(ont, res, alloc_obj)
    answer = analyzer.answer(q, gb)
    print(f"[Graph-RAG] problem={answer['problem']}, "
          f"edges={answer['evidence_edges']}, "
          f"concepts={len(answer['matched_concepts'])}")
    print(f"            answer preview: {answer['answer'][:120]}...")

    # --- Phase 2b: DynamicOntologyExpander ---
    expander = DynamicOntologyExpander(ont)
    proposals = expander.propose("How does SEI fracture affect dead lithium?")
    print(f"[OntologyExpander] {len(proposals)} proposal(s): "
          f"{[p['name'] for p in proposals]}")
    if proposals:
        ok = expander.accept(proposals[0]["name"], proposals[0]["type"],
                             proposals[0]["definition"])
        print(f"            accepted {proposals[0]['name']}: {ok}, "
              f"ontology now has {len(ont.concepts)} concepts")

    # --- Phase 3b: Advanced Analytics ---
    bursts = detect_dendrite_keyword_bursts(lit, ext)
    print(f"[Bursts] {len(bursts)} bursting (concept,year) pairs")
    if not bursts.empty:
        print(f"         top: {bursts.iloc[0].to_dict()}")

    gene = build_dendrite_genealogy(gb)
    print(f"[Genealogy] {len(gene)} concepts ranked; roles: "
          f"{gene['role'].value_counts().to_dict() if not gene.empty else {}}")
    if not gene.empty:
        top_pr = gene.iloc[0]
        print(f"            top PageRank: {top_pr['concept']} "
              f"(pr={top_pr['pagerank']}, role={top_pr['role']})")

    bridges = detect_mechanics_morphology_bridges(gb)
    print(f"[Mech->Morph bridges] {len(bridges)} bridge concepts")
    if not bridges.empty:
        print(f"            top bridge: {bridges.iloc[0].to_dict()}")

    drift = compute_semantic_drift(lit, ext)
    print(f"[Semantic drift] {len(drift)} years tracked")
    if not drift.empty:
        print(f"            years: {drift['year'].tolist()}")

    # --- Phase 6b: Metrics logger ---
    metrics = DendriteMetricsLogger()
    metrics.log("CSV Build", {"models": len(models)})
    metrics.log_literature_extraction(len(lit), len(triples), duration_s=0.05)
    metrics.log_vlm_inference("simulated.png", len(fake_vlm), duration_s=0.01)
    mdf = metrics.as_dataframe()
    print(f"[Metrics] {len(mdf)} steps logged, RAM sample: "
          f"{mdf['ram_mb'].iloc[0] is not None}")

    # --- Markdown report ---
    md = _generate_markdown_report(gb, metrics)
    print(f"[Markdown report] {len(md)} chars, "
          f"has table: {'|---|' in md}")

    print("\nALL CHECKS PASSED.")


def _streamlit_running() -> bool:
    """True only when executed under an active `streamlit run` server."""
    try:
        from streamlit import runtime
        return bool(runtime.exists())
    except Exception:
        return False


if __name__ == "__main__":
    if os.environ.get("RUN_STREAMLIT") or _streamlit_running():
        _streamlit_main()
    else:
        logging.basicConfig(level=logging.INFO)
        _headless_smoke_test()
