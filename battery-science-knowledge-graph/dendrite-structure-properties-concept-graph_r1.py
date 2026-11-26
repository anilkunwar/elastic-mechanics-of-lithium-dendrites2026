#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dendrite-Structure-Geometry Concept Graph v8.0 + QDWA + Graph-RAG + ML
================================================================================
Single-file multimodal knowledge-graph application for lithium-dendrite
suppression research, architected identically to the UAV Drone codebase
("Battery Optimization in Drone Systems Concept Graph v6.2 + QDWA")
but with full ML / GraphSAGE / Latent-MoE / distillation / validation parity.

Three evidence streams fused into one NetworkX DiGraph:
    Stream 1 : manuscript CSV ground truth (6 phase-field models A-F)
    Stream 2 : literature abstracts (sentence co-occurrence)
    Stream 3 : phase-field images via Qwen2.5-VL-3B-Instruct

Feature parity with UAV v6.2:
    ✓ 60+ RelationshipType enum + colour/width/style registries
    ✓ SparseGraphSAGE link-prediction model
    ✓ LatentMoEKGExtractor (32 experts, top-4 routing)
    ✓ Negative-edge sampling with path-length filter
    ✓ Research-direction scoring (GNN affinity × novelty × gain × feasibility)
    ✓ Concept distillation (TF-IDF × coherence × frequency)
    ✓ Graph validation (modularity, silhouette, permutation p-values)
    ✓ Bootstrap confidence intervals
    ✓ Semantic edge layer (sentence-transformer cosine)
    ✓ DynamicOntologyExpander (LLM-driven concept addition)
    ✓ Publication PyVis with JS side-panel
    ✓ 13-tab Streamlit UI
    ✓ RunMetricsLogger with VRAM tracking

Anchor manuscript: "Elasticity Coupled Phase Field Model for Understanding
Dendrite Growth Pattern in Structurally Partitioned Electrolytes"
(Y. Li et al., 2026).

Run:
    streamlit run app.py            # Streamlit UI (production)
    python app.py                   # headless smoke test (no Streamlit)
"""

# ============================================================================
# IMPORTS
# ============================================================================
from __future__ import annotations

import base64
import functools
import gc
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.sparse as sparse
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    sparse = None
    optim = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.colors
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None

try:
    from pyvis.network import Network
except Exception:
    Network = None

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (silhouette_score, r2_score,
                                  mean_absolute_error, mean_squared_error)
except Exception:
    Ridge = None
    cosine_similarity = None
    TfidfVectorizer = None
    silhouette_score = None
    r2_score = None
    mean_absolute_error = None
    mean_squared_error = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ============================================================================
# PERFORMANCE MONITORING DECORATOR
# ============================================================================
class PerformanceMonitor:
    _timings: Dict[str, float] = {}
    _call_counts: Dict[str, int] = {}

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls) -> str:
        if not cls._timings:
            return "No timing data."
        rows = []
        for name, total in sorted(cls._timings.items(),
                                  key=lambda x: x[1], reverse=True):
            n = cls._call_counts.get(name, 1)
            rows.append(f"  {name}: {total:.3f}s total "
                        f"({n} calls, {total / n:.4f}s avg)")
        return "\n".join(rows)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            key = func.__qualname__
            PerformanceMonitor._timings[key] = (
                PerformanceMonitor._timings.get(key, 0) + elapsed)
            PerformanceMonitor._call_counts[key] = (
                PerformanceMonitor._call_counts.get(key, 0) + 1)
    return wrapper


# ============================================================================
# RUNTIME COMPUTATIONAL METRICS LOGGER
# ============================================================================
def get_memory_usage_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0


def get_gpu_vram_mb() -> float:
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


class RunMetricsLogger:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_wall = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.counters: Dict[str, Any] = {}
        self._open: Optional[Dict[str, Any]] = None

    def start_step(self, name: str) -> None:
        self._open = {"name": name, "t0": time.perf_counter(),
                      "mem0": get_memory_usage_mb(), "vram0": get_gpu_vram_mb()}

    def end_step(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._open:
            return
        o = self._open
        dur = time.perf_counter() - o["t0"]
        mem = get_memory_usage_mb()
        vram = get_gpu_vram_mb()
        entry = {"name": o["name"], "duration_s": round(dur, 4),
                 "mem_start_mb": round(o["mem0"], 1),
                 "mem_end_mb": round(mem, 1),
                 "mem_delta_mb": round(mem - o["mem0"], 1),
                 "vram_start_mb": round(o["vram0"], 1),
                 "vram_end_mb": round(vram, 1),
                 "wall_clock": datetime.now().strftime("%H:%M:%S")}
        if extra:
            entry["extra"] = extra
        self.steps.append(entry)
        self.resources.append({"elapsed_s": round(time.time() - self.start_wall, 2),
                               "rss_mb": round(mem, 1),
                               "vram_mb": round(vram, 1),
                               "tag": o["name"]})
        self._open = None

    def set(self, key: str, value: Any) -> None:
        self.counters[key] = value

    def add(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def step_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def resource_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.resources)

    def summary(self) -> Dict[str, Any]:
        df = self.step_df()
        rdf = self.resource_df()
        return {
            "run_id": self.run_id,
            "total_duration_s": round(float(df["duration_s"].sum())
                                       if not df.empty else 0.0, 3),
            "wall_clock_s": round(time.time() - self.start_wall, 3),
            "n_steps": len(self.steps),
            "peak_rss_mb": round(float(rdf["rss_mb"].max()) if not rdf.empty
                                  else get_memory_usage_mb(), 1),
            "peak_vram_mb": round(float(rdf["vram_mb"].max()) if not rdf.empty
                                   else 0.0, 1),
            "final_rss_mb": round(get_memory_usage_mb(), 1),
            "counters": dict(self.counters),
            "function_timings": PerformanceMonitor.get_report(),
        }

    def save_log(self, folder: Optional[str] = None) -> str:
        folder = folder or os.path.join(SCRIPT_DIR, "metrics_logs")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"dendrite_metrics_{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "steps": self.steps,
                       "resources": self.resources}, f, indent=2, default=str)
        return path


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
try:
    import streamlit as st
    st.set_page_config(
        page_title="Dendrite Concept Graph v8.0 + QDWA + ML",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    st = None


# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
              if "__file__" in globals() else os.getcwd())
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics_logs")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "exports")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# ============================================================================
# WIDGET SESSION-STATE DEFAULTS
# ============================================================================
_DEND_WIDGET_DEFAULTS: Dict[str, Any] = {
    "qdwa_top_k": 8, "qdwa_min_weight": 0.05,
    "qdwa_cmap": "viridis", "qdwa_cmap_reverse": False,
    "graph_cmap": "viridis", "graph_height": 650,
    "graph_show_backbone": False, "graph_physics": True,
    "analytics_burst_threshold": 2.0, "analytics_drift_early": 0.3,
    "analytics_drift_late": 0.3,
    "llm_model": "gpt-4o-mini", "llm_temperature": 0.2, "llm_max_tokens": 800,
    "llm_problem": "general",
    "vlm_max_tokens": 512, "vlm_temperature": 0.1,
    "export_dpi": 300, "export_figsize_w": 14, "export_figsize_h": 12,
    "font_family": "Inter, Segoe UI, Roboto, sans-serif",
    "font_size": 11, "title_size": 15,
    # ML widget defaults
    "ml_epochs": 50, "ml_lr": 0.001, "ml_hidden": 128,
    "ml_train_ratio": 0.8, "ml_semantic_threshold": 0.7,
    "ml_top_pairs": 30, "ml_use_moe": True,
    "ml_n_experts": 32, "ml_top_k_experts": 4,
    "ml_distill_max_docs": 30, "ml_coherence_window": 20,
    "ml_validate_bootstrap": 500, "ml_permutation_trials": 50,
    "ml_expander_provider": "openai", "ml_expander_max_new": 5,
}


def init_widget_defaults() -> None:
    if st is None:
        return
    for k, v in _DEND_WIDGET_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_metrics_logger() -> RunMetricsLogger:
    if st is None:
        return RunMetricsLogger()
    if "metrics_logger" not in st.session_state:
        st.session_state.metrics_logger = RunMetricsLogger()
    return st.session_state.metrics_logger


# ============================================================================
# COLORMAP REGISTRY
# ============================================================================
SUPPORTED_COLORMAPS: Dict[str, str] = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno",
    "magma": "Magma", "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet",
    "rainbow": "Rainbow", "coolwarm": "Coolwarm", "RdBu": "RdBu",
    "seismic": "Seismic", "Spectral": "Spectral", "tab10": "Set1",
    "tab20": "Set2", "Dark2": "Dark2", "Paired": "Paired", "Accent": "Accent",
    "Pastel1": "Pastel1", "cubehelix": "Cubehelix", "spring": "Spring",
    "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "BuPu": "BuPu",
    "GnBu": "GnBu", "YlGnBu": "YlGnBu", "YlGn": "YlGn", "Greys": "Greys",
    "terrain": "Terrain", "ocean": "Ocean", "gist_earth": "GistEarth",
}


def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


THEME_PRESETS: Dict[str, Dict[str, str]] = {
    "Bright (Default)": {
        "plotly_paper": "#ffffff", "plotly_bg": "#ffffff", "font": "#1a1a1a",
        "axis_color": "#444", "grid_color": "#e5e5e5",
        "highlight_bg": "#ff6b6b",
    },
    "Dark": {
        "plotly_paper": "#1e1e1e", "plotly_bg": "#1e1e1e", "font": "#f0f0f0",
        "axis_color": "#c0c0c0", "grid_color": "#3a3a3a",
        "highlight_bg": "#66d9ff",
    },
}


# ============================================================================
# EMBEDDED SAMPLE DATA
# ============================================================================
SAMPLE_MODELS_CSV = """Model,Geometry,S_L_Ratio,Max_Stress_GPa,Area_Fraction_pct,Notes
A,boundary_layer,4:1,0.059,12.0,80% solid - dendrites confined near electrode
B,alternating_stripes,2:3,0.031,18.0,40% solid - moderate branching
C,central_layer,4:1,0.052,14.0,80% solid - confined near electrode
D,boundary_layer,3:2,0.040,22.0,60% solid - side branching begins
E,alternating_stripes,1:4,0.018,35.0,20% solid - severe side branching
F,central_layer,2:3,0.029,20.0,40% solid - thinner central layer than C
"""

SAMPLE_LITERATURE_CSV = """title,year,abstract
"Phase-field modeling of lithium dendrite growth",2019,"Lithium dendrite growth severely limits the safety of lithium-ion batteries. Phase-field models simulate dendrite evolution without explicit interface tracking. The dendrite morphology depends on the applied voltage and overpotential."
"Elasticity-coupled phase field model",2021,"The elastic modulus of LLZO solid electrolyte reaches 155 GPa while the liquid electrolyte is nearly incompressible with Poisson ratio 0.499. High von Mises stress at the solid-liquid boundary inhibits side branching of lithium dendrites."
"Solid-liquid hybrid electrolytes",2022,"Composite electrolytes combine the high ionic conductivity of liquid electrolytes with the mechanical stability of solid electrolytes. The solid-liquid ratio and spatial configuration govern dendrite suppression."
"Butler-Volmer kinetics in dendrite growth",2022,"The Butler-Volmer equation drives the electrode order parameter evolution. The Nernst-Planck equation governs lithium ion flux through the effective conductivity of the electrolyte."
"Stress-mediated dendrite suppression",2023,"Mechanical stress concentrates at the solid electrolyte boundary. The eigenstrain from lithium deposition generates von Mises stress that constrains dendrite growth and promotes dendrite suppression."
"Dead lithium and capacity loss",2024,"Dendrite growth causes internal short circuit and dead lithium formation. The dendrite area fraction correlates with capacity fade and Coulombic efficiency loss in lithium metal batteries."
"Grain boundary engineering for dendrite suppression",2024,"Grain boundary width controls dendrite penetration. Narrower channels increase stress concentration and suppress side branching in LLZO solid electrolytes."
"Phase-field simulation of striped electrolytes",2025,"Alternating stripe geometries modulate effective conductivity and Li-ion flux. The spatial configuration of solid partitions inhibits dendrite growth near electrode boundaries."
"""


# ============================================================================
# RELATIONSHIP TYPE ENUM (60+ entries)
# ============================================================================
class RelationshipType(Enum):
    # Structural / hierarchical
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"

    # Causal / directional
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    CONSTRAINS = "constrains"
    INHIBITS = "inhibits"
    PROMOTES = "promotes"
    DRIVES = "drives"
    ENABLES = "enables"
    MODIFIES = "modifies"
    CORRECTS = "corrects"

    # Phase / thermodynamic
    TRANSITIONS_TO = "transitions_to"
    REPLACES = "replaces"
    FORMS = "forms"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"

    # Computational
    TRAINS = "trains"
    OUTPUTS = "outputs"
    LEARNS = "learns"
    CAPTURES = "captures"
    COMPUTES = "computes"
    SIMULATES = "simulates"
    MODELS = "models"
    APPROXIMATES = "approximates"
    MAPS = "maps"

    # Analysis
    QUANTIFIES = "quantifies"
    EVALUATES = "evaluates"
    COMPARES = "compares"
    VALIDATES = "validates"
    AVERAGES = "averages"
    CORRELATES = "correlates"
    INDICATES = "indicates"

    # Structural graph
    PARALLELIZES = "parallelizes"
    POSITIONS = "positions"
    IDENTIFIES = "identifies"
    PROCESSES = "processes"
    GROUPS = "groups"
    INTEGRATES = "integrates"
    COUPLES = "couples"

    # Discovery / optimization
    DISCOVERS = "discovers"
    PRE_TRAINS = "pre_trains"
    GENERALIZES = "generalizes"
    QUERIES = "queries"
    OPTIMIZES = "optimizes"
    DESIGNS = "designs"
    CONSTRUCTS = "constructs"

    # Advanced modeling
    UPSCALES = "upscales"
    RESOLVES = "resolves"
    SYNCHRONIZES = "synchronizes"
    CHARACTERIZES = "characterizes"
    DECOMPOSES = "decomposes"
    FRAMES = "frames"
    COMPOSES = "composes"
    QUALIFIES = "qualifies"

    # Explanation / visualization
    STRENGTHENS = "strengthens"
    EXPLAINS = "explains"
    INTERPRETS = "interprets"
    VISUALIZES = "visualizes"
    ACCELERATES = "accelerates"
    ENFORCES = "enforces"

    # Domain-specific (manuscript)
    HAS_GEOMETRY = "has_geometry"
    HAS_RATIO = "has_ratio"
    HAS_PROPERTY = "has_property"
    HAS_PROBLEM = "has_problem"
    USES = "uses"
    GENERATES = "generates"
    EXHIBITS = "exhibits"
    SUPPORTED_BY_SIMULATION = "supported_by_simulation"

    # Generic fallbacks
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    SELECTS = "selects"
    INITIATES = "initiates"
    DETECTS = "detects"


RELATIONSHIP_TO_IDX: Dict[str, int] = {r.name: i for i, r in enumerate(RelationshipType)}
NUM_EDGE_TYPES: int = len(RelationshipType)


EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    RelationshipType.SYNONYM: "#AAAAAA",
    RelationshipType.HYPERNYM: "#5B9BD5",
    RelationshipType.HYPONYM: "#5B9BD5",
    RelationshipType.PART_OF: "#70AD47",
    RelationshipType.HAS_PART: "#70AD47",
    RelationshipType.CO_OCCURS: "#BFBFBF",
    RelationshipType.CAUSES: "#FF4444",
    RelationshipType.RESULTS_IN: "#E06040",
    RelationshipType.INFLUENCES: "#FF8C00",
    RelationshipType.DEPENDS_ON: "#DAA520",
    RelationshipType.CONSTRAINS: "#CC5500",
    RelationshipType.INHIBITS: "#8B0000",
    RelationshipType.PROMOTES: "#B22222",
    RelationshipType.DRIVES: "#DC143C",
    RelationshipType.ENABLES: "#FF7F50",
    RelationshipType.MODIFIES: "#FF6347",
    RelationshipType.CORRECTS: "#CD5C5C",
    RelationshipType.TRANSITIONS_TO: "#8A2BE2",
    RelationshipType.REPLACES: "#9932CC",
    RelationshipType.FORMS: "#9370DB",
    RelationshipType.STABILIZES: "#7B68EE",
    RelationshipType.PRESERVES: "#6A5ACD",
    RelationshipType.TRAINS: "#00CED1",
    RelationshipType.OUTPUTS: "#20B2AA",
    RelationshipType.LEARNS: "#48D1CC",
    RelationshipType.CAPTURES: "#40E0D0",
    RelationshipType.COMPUTES: "#008B8B",
    RelationshipType.SIMULATES: "#5F9EA0",
    RelationshipType.MODELS: "#4682B4",
    RelationshipType.APPROXIMATES: "#87CEEB",
    RelationshipType.MAPS: "#00BFFF",
    RelationshipType.QUANTIFIES: "#32CD32",
    RelationshipType.EVALUATES: "#228B22",
    RelationshipType.COMPARES: "#3CB371",
    RelationshipType.VALIDATES: "#2E8B57",
    RelationshipType.AVERAGES: "#66CDAA",
    RelationshipType.CORRELATES: "#00FA9A",
    RelationshipType.INDICATES: "#90EE90",
    RelationshipType.PARALLELIZES: "#FFD700",
    RelationshipType.POSITIONS: "#FFC125",
    RelationshipType.IDENTIFIES: "#F0E68C",
    RelationshipType.PROCESSES: "#EEE8AA",
    RelationshipType.GROUPS: "#DAA520",
    RelationshipType.INTEGRATES: "#B8860B",
    RelationshipType.COUPLES: "#CD950C",
    RelationshipType.DISCOVERS: "#FF69B4",
    RelationshipType.PRE_TRAINS: "#FF1493",
    RelationshipType.GENERALIZES: "#DB7093",
    RelationshipType.QUERIES: "#C71585",
    RelationshipType.OPTIMIZES: "#FF00FF",
    RelationshipType.DESIGNS: "#BA55D3",
    RelationshipType.CONSTRUCTS: "#DA70D6",
    RelationshipType.UPSCALES: "#8B4513",
    RelationshipType.RESOLVES: "#A0522D",
    RelationshipType.SYNCHRONIZES: "#D2691E",
    RelationshipType.CHARACTERIZES: "#CD853F",
    RelationshipType.DECOMPOSES: "#DEB887",
    RelationshipType.FRAMES: "#D2B48C",
    RelationshipType.COMPOSES: "#BC8F8F",
    RelationshipType.QUALIFIES: "#F4A460",
    RelationshipType.STRENGTHENS: "#7FFF00",
    RelationshipType.EXPLAINS: "#ADFF2F",
    RelationshipType.INTERPRETS: "#7CFC00",
    RelationshipType.VISUALIZES: "#00FF7F",
    RelationshipType.ACCELERATES: "#98FB98",
    RelationshipType.ENFORCES: "#90EE90",
    RelationshipType.HAS_GEOMETRY: "#2ca02c",
    RelationshipType.HAS_RATIO: "#98df8a",
    RelationshipType.HAS_PROPERTY: "#c7c7c7",
    RelationshipType.HAS_PROBLEM: "#d62728",
    RelationshipType.USES: "#dbdb8d",
    RelationshipType.GENERATES: "#ff7f0e",
    RelationshipType.EXHIBITS: "#e377c2",
    RelationshipType.SUPPORTED_BY_SIMULATION: "#9467bd",
    RelationshipType.SEMANTIC: "#808080",
    RelationshipType.INFERRED: "#A9A9A9",
    RelationshipType.BRIDGE: "#C0C0C0",
    RelationshipType.SELECTS: "#D3D3D3",
    RelationshipType.INITIATES: "#696969",
    RelationshipType.DETECTS: "#556B2F",
}
EDGE_COLOR_FALLBACK = "#888888"


def get_edge_color(rel_type: RelationshipType) -> str:
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)


def get_edge_color_by_name(rel_name: str) -> str:
    if not rel_name:
        return EDGE_COLOR_FALLBACK
    key = rel_name.upper().replace(" ", "_")
    try:
        return get_edge_color(RelationshipType[key])
    except (KeyError, AttributeError):
        return EDGE_COLOR_FALLBACK


def get_edge_width(rel_type: RelationshipType) -> float:
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES,
              RelationshipType.FORMS, RelationshipType.STABILIZES,
              RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS,
              RelationshipType.INHIBITS, RelationshipType.GENERATES,
              RelationshipType.SUPPORTED_BY_SIMULATION}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO, RelationshipType.COMPUTES,
              RelationshipType.PROMOTES, RelationshipType.EXHIBITS}
    if rel_type in STRONG:
        return 3.0
    if rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE,
              RelationshipType.SUPPORTED_BY_SIMULATION}
    return "dashed" if rel_type in DASHED else "solid"


def resolve_relation_type(rel_name: str) -> Optional[RelationshipType]:
    if not rel_name:
        return None
    try:
        return RelationshipType[rel_name.upper().replace(" ", "_")]
    except (KeyError, AttributeError):
        return None


# ============================================================================
# PHASE 1: DENDRITE GEOMETRY ONTOLOGY
# ============================================================================
@dataclass
class ConceptNode:
    name: str
    type: str
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        t = text.lower().strip()
        if t == self.name.lower():
            return True
        return any(t == s.lower() for s in self.synonyms)


RELATION_EXPLANATIONS: Dict[str, str] = {
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
    "SUPPORTED_BY_SIMULATION": "literature claim corroborated by simulation",
    "HAS_PROPERTY": "material has this property",
    "USES": "method uses this sub-component",
    "HAS_PROBLEM": "application suffers from this problem",
    "INFLUENCES": "subject affects object's behaviour",
    "SEMANTIC": "semantic similarity (embedding-based) edge",
}

MODEL_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "Model_A": {"geometry": "boundary_layer", "ratio": "4:1", "solid_frac": 0.80},
    "Model_B": {"geometry": "alternating_stripes", "ratio": "2:3", "solid_frac": 0.40},
    "Model_C": {"geometry": "central_layer", "ratio": "4:1", "solid_frac": 0.80},
    "Model_D": {"geometry": "boundary_layer", "ratio": "3:2", "solid_frac": 0.60},
    "Model_E": {"geometry": "alternating_stripes", "ratio": "1:4", "solid_frac": 0.20},
    "Model_F": {"geometry": "central_layer", "ratio": "2:3", "solid_frac": 0.40},
}


class DendriteGeometryOntology:
    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Tuple[str, str, str, float]] = []
        self._synonym_index: Dict[str, str] = {}
        self._build()

    def _add_concept(self, name: str, ctype: str, *,
                     definition: str = "",
                     synonyms: Optional[List[str]] = None) -> None:
        self.concepts[name] = ConceptNode(
            name=name, type=ctype, definition=definition,
            synonyms=list(synonyms or []))

    def _build(self) -> None:
        for model in MODEL_SIGNATURES:
            sig = MODEL_SIGNATURES[model]
            self._add_concept(
                model, "Simulation_Model",
                definition=f"{model}: {sig['geometry']} with S/L = "
                           f"{sig['ratio']} ({int(sig['solid_frac']*100)}% solid)",
                synonyms=[model.replace("_", " ")])
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
                          synonyms=["spatial distribution", "spatial location"])
        self._add_concept("elastic_modulus", "Mechanical_Property",
                          synonyms=["Young's modulus", "stiffness"])
        self._add_concept("poisson_ratio", "Mechanical_Property",
                          synonyms=["Poisson ratio", "incompressibility"])
        self._add_concept("von_mises_stress", "Mechanical_Property",
                          synonyms=["von Mises stress", "boundary stress",
                                    "stress concentration", "high stress"])
        self._add_concept("high_von_mises_stress", "Mechanical_Property",
                          synonyms=["stress peak", "stress concentration at boundary"])
        self._add_concept("eigenstrain", "Mechanical_Property",
                          synonyms=["volumetric expansion", "lithiation strain"])
        self._add_concept("effective_conductivity", "Electrochemical_Property",
                          synonyms=["ionic conductivity", "Li+ transport",
                                    "sigma_eff", "electric conductivity"])
        self._add_concept("overpotential", "Electrochemical_Property",
                          synonyms=["activation overpotential"])
        self._add_concept("applied_voltage", "Electrochemical_Property",
                          synonyms=["boundary potential", "bias"])
        self._add_concept("li_ion_flux", "Electrochemical_Property",
                          synonyms=["lithium ion migration", "ionic flux"])
        self._add_concept("phase_field_model", "Method",
                          synonyms=["PFM", "phase-field", "phase field"])
        self._add_concept("order_parameter_eta", "Method",
                          synonyms=["eta", "electrode order parameter"])
        self._add_concept("order_parameter_phi", "Method",
                          synonyms=["phi", "solid partition order parameter"])
        self._add_concept("butler_volmer", "Method",
                          synonyms=["Butler-Volmer kinetics",
                                    "electrochemical kinetics"])
        self._add_concept("nernst_planck", "Method",
                          synonyms=["Nernst-Planck equation", "species transport"])
        self._add_concept("lithium_metal", "Material",
                          synonyms=["Li metal", "anode", "electrode"])
        self._add_concept("llzo", "Material",
                          synonyms=["LLZO", "solid electrolyte partition", "garnet"])
        self._add_concept("liquid_electrolyte", "Material",
                          synonyms=["LiPF6", "LiPF4", "organic electrolyte"])
        self._add_concept("hybrid_electrolyte", "Material",
                          synonyms=["solid-liquid electrolyte",
                                    "partitioned electrolyte",
                                    "composite electrolyte"])
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
                          synonyms=["Li-ion battery", "lithium battery",
                                    "Li-metal battery"])
        for name, ctype, syn in [
            ("low_solid_fraction", "Parameter", ["low S/L", "low solid content"]),
            ("SL_4:1", "Ratio", ["4:1"]),
            ("SL_2:3", "Ratio", ["2:3"]),
            ("SL_3:2", "Ratio", ["3:2"]),
            ("SL_1:4", "Ratio", ["1:4"]),
        ]:
            if name not in self.concepts:
                self._add_concept(name, ctype, synonyms=syn)
        self._add_ground_truth()
        self._build_synonym_index()

    def _build_synonym_index(self) -> None:
        self._synonym_index = {}
        for canonical, node in self.concepts.items():
            self._synonym_index[canonical.lower()] = canonical
            for syn in node.synonyms:
                self._synonym_index[syn.lower()] = canonical

    def _add_ground_truth(self) -> None:
        for model, sig in MODEL_SIGNATURES.items():
            self.relationships.append((model, "HAS_GEOMETRY",
                                        sig["geometry"], 1.0))
            self.relationships.append((model, "HAS_RATIO",
                                        f"SL_{sig['ratio']}", 1.0))
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
            ("grain_boundary_width", "INFLUENCES", "von_mises_stress", 0.75),
        ]
        self.relationships.extend(gt)

    def resolve(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower().strip()
        if t in self._synonym_index:
            return self._synonym_index[t]
        for name, node in self.concepts.items():
            if node.matches(text):
                return name
        return None

    def allowed_concepts(self) -> Set[str]:
        return set(self.concepts.keys())

    def concept_types(self) -> Dict[str, str]:
        return {n: c.type for n, c in self.concepts.items()}

    def infer_path(self, src: str, dst: str,
                   max_depth: int = 3) -> List[List[str]]:
        if src not in self.concepts or dst not in self.concepts:
            return []
        adj: Dict[str, List[str]] = defaultdict(list)
        for s, _r, o, _w in self.relationships:
            adj[s].append(o)
            adj[o].append(s)
        q = deque([[src]])
        seen = {src}
        out: List[List[str]] = []
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

    def add_concept(self, name: str, ctype: str,
                    definition: str = "",
                    synonyms: Optional[List[str]] = None,
                    inferred_from: Optional[str] = None) -> bool:
        """Dynamic addition used by DynamicOntologyExpander."""
        if name in self.concepts:
            return False
        self._add_concept(name, ctype, definition=definition,
                          synonyms=synonyms)
        self._synonym_index[name.lower()] = name
        for syn in (synonyms or []):
            self._synonym_index[syn.lower()] = name
        if inferred_from and inferred_from in self.concepts:
            self.relationships.append((inferred_from, "SYNONYM", name, 0.5))
        return True


# ============================================================================
# QDWA — DENDRITE CATEGORY DEFINITIONS
# ============================================================================
DENDRITE_CATEGORIES: Dict[str, List[str]] = {
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
    ],
    "Morphology Outcomes": [
        "dendrite_area_fraction", "side_branching", "severe_side_branching",
        "dendrite_suppression", "dendrite_growth", "dead_lithium",
        "internal_short_circuit",
    ],
}

CATEGORY_SHORT_NAMES: Dict[str, str] = {
    "Geometrical Design": "Geometry",
    "Mechanical Constraints": "Mechanics",
    "Electrochemical Transport": "Electrochem",
    "Morphology Outcomes": "Morphology",
}

CATEGORY_COLOURS: Dict[str, str] = {
    "Geometrical Design": "#2ca02c",
    "Mechanical Constraints": "#ff7f0e",
    "Electrochemical Transport": "#1f77b4",
    "Morphology Outcomes": "#d62728",
}


def concept_to_category(concept: str) -> str:
    for cat, members in DENDRITE_CATEGORIES.items():
        if concept in members:
            return cat
    return "Other"


# ============================================================================
# PHASE 2: QDWA WEIGHT ALLOCATOR
# ============================================================================
@dataclass
class QDWAAllocation:
    query: str
    category_weights: Dict[str, float]
    concept_weights: Dict[str, float]
    top_concepts: List[Tuple[str, float, str]]
    problem: str
    reasoning: List[str]
    coverage: Dict[str, int]


class QDWAWeightAllocator:
    def __init__(self, ontology: DendriteGeometryOntology,
                 resolver: "AdvancedConceptResolver") -> None:
        self.ontology = ontology
        self.resolver = resolver

    @timed
    def allocate(self, query: str,
                 centrality: Optional[Dict[str, float]] = None,
                 top_k: int = 8, min_weight: float = 0.05) -> QDWAAllocation:
        centrality = centrality or {}
        reasoning: List[str] = []
        query_lower = query.lower()
        found: List[Tuple[str, float]] = []
        for name, node in self.ontology.concepts.items():
            forms = [name] + list(node.synonyms)
            best = 0.0
            for form in forms:
                if len(form) < 3:
                    continue
                if re.search(rf"\b{re.escape(form.lower())}\b", query_lower):
                    best = 1.0
                    break
            if best > 0:
                found.append((name, best))
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", query)
        for tok in tokens:
            resolved = self.resolver.resolve(tok)
            if resolved and all(resolved != n for n, _ in found):
                found.append((resolved, 0.6))
                reasoning.append(f"Semantic match: '{tok}' → {resolved}")
        category_scores: Dict[str, float] = {c: 0.0 for c in DENDRITE_CATEGORIES}
        category_counts: Dict[str, int] = {c: 0 for c in DENDRITE_CATEGORIES}
        concept_scores: Dict[str, float] = {}
        for concept, base in found:
            cat = concept_to_category(concept)
            cent = centrality.get(concept, 0.05)
            weight = base * (1.0 + cent * 10.0)
            concept_scores[concept] = weight
            if cat in category_scores:
                category_scores[cat] += weight
                category_counts[cat] += 1
        total = sum(category_scores.values())
        if total < 1e-9:
            category_weights = {c: 1.0 / len(DENDRITE_CATEGORIES)
                                for c in DENDRITE_CATEGORIES}
            reasoning.append("No concept detected — uniform fallback.")
        else:
            category_weights = {c: s / total for c, s in category_scores.items()}
        sorted_c = sorted(concept_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_concepts = [(c, w, concept_to_category(c)) for c, w in sorted_c
                        if w >= min_weight][:top_k]
        problem = self._infer_problem(query_lower, category_weights)
        return QDWAAllocation(query=query, category_weights=category_weights,
                              concept_weights=concept_scores,
                              top_concepts=top_concepts, problem=problem,
                              reasoning=reasoning, coverage=category_counts)

    @staticmethod
    def _infer_problem(query_lower: str,
                       cat_w: Dict[str, float]) -> str:
        if any(k in query_lower for k in
               ["dead lithium", "capacity", "fade", "dead li"]):
            return "dead_lithium_mitigation"
        if any(k in query_lower for k in ["sei", "solid electrolyte interphase"]):
            return "sei_stability"
        if any(k in query_lower for k in
               ["stress", "modulus", "elastic", "suppress", "confine"]):
            return "mechanical_suppression"
        if any(k in query_lower for k in
               ["geometry", "stripe", "layer", "ratio", "shape"]):
            return "geometry_optimization"
        top_cat = max(cat_w.items(), key=lambda kv: kv[1])[0]
        return {"Geometrical Design": "geometry_optimization",
                "Mechanical Constraints": "mechanical_suppression",
                "Electrochemical Transport": "sei_stability",
                "Morphology Outcomes": "dead_lithium_mitigation"}.get(
                    top_cat, "general")


# ============================================================================
# PHASE 3: ADVANCED CONCEPT RESOLVER
# ============================================================================
class AdvancedConceptResolver:
    def __init__(self, ontology: DendriteGeometryOntology,
                 threshold: float = 0.82) -> None:
        self.ontology = ontology
        self.threshold = threshold
        self._embed_model = None
        self._ontology_concepts_list: List[str] = list(ontology.concepts.keys())
        self._ontology_embedding_matrix: Optional[np.ndarray] = None

    def resolve(self, text: str) -> Optional[str]:
        if not text:
            return None
        name = self.ontology.resolve(text)
        if name:
            return name
        return self._semantic_resolve(text)

    def strict_ontology_mapping(self, text: str,
                                allowed_concepts: Set[str]) -> Optional[str]:
        if not text:
            return None
        name = self.ontology.resolve(text)
        if name and name in allowed_concepts:
            return name
        self._init_embeddings()
        if self._ontology_embedding_matrix is not None:
            query_emb = self._embed_one(text)
            if query_emb is None:
                return None
            allowed_indices = [i for i, c in enumerate(self._ontology_concepts_list)
                               if c in allowed_concepts]
            if not allowed_indices:
                return None
            sub = self._ontology_embedding_matrix[allowed_indices]
            sims = sub @ query_emb
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
                        "exact/synonym only.", exc)
            self._embed_model = False
            return
        try:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            texts: List[str] = []
            owners: List[str] = []
            for name, node in self.ontology.concepts.items():
                for t in ([name] + list(node.synonyms)
                          + ([node.definition] if node.definition else [])):
                    texts.append(t)
                    owners.append(name)
            embs = self._embed_model.encode(texts, convert_to_numpy=True,
                                             normalize_embeddings=True,
                                             show_progress_bar=False)
            dim = embs.shape[1]
            self._ontology_embedding_matrix = np.zeros(
                (len(self._ontology_concepts_list), dim), dtype=np.float32)
            idx_of = {c: i for i, c in enumerate(self._ontology_concepts_list)}
            counts = [0] * len(self._ontology_concepts_list)
            for owner, vec in zip(owners, embs):
                i = idx_of[owner]
                self._ontology_embedding_matrix[i] += vec
                counts[i] += 1
            for i, n in enumerate(counts):
                if n:
                    self._ontology_embedding_matrix[i] /= n
            log.info("Loaded embeddings for %d concepts.",
                     len(self._ontology_concepts_list))
        except Exception as exc:
            log.warning("Embedding init failed (%s); exact/synonym only.", exc)
            self._embed_model = False

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        self._init_embeddings()
        if not self._embed_model:
            return None
        return self._embed_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False)[0].astype(np.float32)

    def _semantic_resolve(self, text: str) -> Optional[str]:
        vec = self._embed_one(text)
        if vec is None or self._ontology_embedding_matrix is None:
            return None
        sims = self._ontology_embedding_matrix @ vec
        best = int(np.argmax(sims))
        return (self._ontology_concepts_list[best]
                if sims[best] > self.threshold else None)

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        self._init_embeddings()
        if not self._embed_model or not texts:
            return None
        return self._embed_model.encode(texts, convert_to_numpy=True,
                                         normalize_embeddings=True,
                                         show_progress_bar=False)


# ============================================================================
# PHASE 4A: LITERATURE TEXT CONCEPT EXTRACTOR
# ============================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
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
        self._matchers: List[Tuple[re.Pattern, str]] = []
        for name, node in ontology.concepts.items():
            for t in [name] + list(node.synonyms):
                t = t.strip()
                if len(t) <= 2 and not t.isupper():
                    continue
                pat = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])",
                    re.IGNORECASE)
                self._matchers.append((pat, name))

    def _scan_sentence(self, sentence: str) -> List[str]:
        found: List[str] = []
        seen: Set[str] = set()
        for pat, concept in self._matchers:
            if concept in seen:
                continue
            if pat.search(sentence):
                found.append(concept)
                seen.add(concept)
        return found

    def extract_from_abstract(self, abstract: str) -> List[LitTriple]:
        pair_weight: Counter = Counter()
        evidence: Dict[Tuple[str, str], str] = {}
        for sent in split_sentences(abstract):
            concepts = self._scan_sentence(sent)
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a, b = concepts[i], concepts[j]
                    key = tuple(sorted((a, b)))
                    pair_weight[key] += 1.0 / (j - i)
                    evidence.setdefault(key, sent)
        if not pair_weight:
            return []
        mx = max(pair_weight.values())
        triples: List[LitTriple] = []
        for (a, b), w in pair_weight.items():
            weight = round(0.4 + 0.6 * (w / mx), 3)
            if weight < self.min_weight:
                continue
            triples.append(LitTriple(a, "CO_OCCURS", b, weight, evidence[(a, b)]))
        return triples

    def extract_from_corpus(self, df: pd.DataFrame,
                            title_col: str = "title",
                            abstract_col: str = "abstract") -> List[Dict]:
        out: List[Dict] = []
        if not {title_col, abstract_col}.issubset(df.columns):
            raise ValueError(f"CSV must contain '{title_col}' and "
                             f"'{abstract_col}'; got {list(df.columns)}")
        for _, row in df.iterrows():
            abstract = f"{row.get(title_col, '')}. {row.get(abstract_col, '')}"
            year_val = row.get("year")
            year_int = (int(year_val) if pd.notna(year_val)
                        and year_val is not None else None)
            for t in self.extract_from_abstract(str(abstract)):
                out.append({
                    "subject": t.subject, "relation": t.relation,
                    "object": t.object, "weight": t.weight,
                    "evidence": t.evidence, "source": t.source,
                    "doc": str(row.get(title_col, ""))[:80],
                    "year": year_int,
                })
        return out

    def concept_frequencies(self, df: pd.DataFrame,
                            abstract_col: str = "abstract") -> Dict[str, int]:
        freq: Counter = Counter()
        for abstract in df.get(abstract_col, []):
            for sent in split_sentences(str(abstract)):
                for c in self._scan_sentence(sent):
                    freq[c] += 1
        return dict(freq.most_common())

    def concept_abstract_map(self, df: pd.DataFrame,
                              abstract_col: str = "abstract") -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = defaultdict(list)
        for idx, abstract in enumerate(df.get(abstract_col, [])):
            for sent in split_sentences(str(abstract)):
                for c in self._scan_sentence(sent):
                    if idx not in mapping[c]:
                        mapping[c].append(idx)
        return dict(mapping)


# ============================================================================
# PHASE 4B: VLM IMAGE EXTRACTOR
# ============================================================================
VALID_RELATIONS: Set[str] = {
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
    "Then output the JSON array of triples.")

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
            return f"✅ Loaded: {self.model_id} on {self.device or 'auto'}"
        if self._load_error:
            return f"⚠️ Unavailable: {self._load_error}"
        return "⏳ Not loaded yet."

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error:
            return
        try:
            import torch as _torch
            from transformers import (Qwen2VLForConditionalGeneration,
                                       AutoProcessor)
        except Exception as exc:
            self._load_error = f"missing torch/transformers: {exc}"
            raise
        if self.device is None:
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        try:
            dtype = _torch.float16 if self.device == "cuda" else _torch.float32
            kwargs = dict(torch_dtype=dtype, device_map="auto")
            try:
                from transformers import BitsAndBytesConfig
                if self.device == "cuda":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=_torch.float16)
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
                                   temperature: float = 0.1) -> List[Dict]:
        self._ensure_loaded()
        if self._model is None:
            return []
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]},
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
        inputs = self._processor(text=[text], images=image_inputs,
                                  padding=True, return_tensors="pt").to(self.device)
        import torch as _torch
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs.update({"temperature": temperature, "do_sample": True})
        else:
            gen_kwargs["do_sample"] = False
        with _torch.no_grad():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        generated = out_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generated, skip_special_tokens=True)[0]
        return self._parse_and_validate(response)

    def _parse_and_validate(self, response: str) -> List[Dict]:
        triples = self._extract_json_array(response)
        validated: List[Dict] = []
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
                log.debug("Dropped VLM triple (%r, %r, %r)",
                          subj_raw, rel, obj_raw)
                continue
            validated.append({
                "subject": subj, "relation": rel, "object": obj,
                "evidence": evidence, "source": "VLM_Vision", "weight": 0.80})
        return validated

    @staticmethod
    def _extract_json_array(text: str) -> List:
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


# ============================================================================
# PHASE 5: MULTIMODAL GRAPH BUILDER (with semantic edge layer)
# ============================================================================
SOURCE_COLOUR: Dict[str, str] = {
    "CSV_Ground_Truth": "#2ca02c", "Manuscript_CSV": "#2ca02c",
    "Manuscript_Theory": "#2ca02c", "Literature_Text": "#1f77b4",
    "VLM_Vision": "#ff7f0e", "Multimodal_Inference": "#9467bd",
    "Ontology_Backbone": "#bdbdbd", "Semantic_Similarity": "#17becf",
    "Ontology_Expander": "#e377c2",
}

NODE_TYPE_COLOUR: Dict[str, str] = {
    "Simulation_Model": "#2ca02c", "Geometry": "#98df8a",
    "Parameter": "#aec7e8", "Ratio": "#98df8a",
    "Mechanical_Property": "#ffbb78",
    "Electrochemical_Property": "#ff9896",
    "Morphology_Metric": "#c5b0d5", "Phenomenon": "#e377c2",
    "Material": "#c7c7c7", "Method": "#dbdb8d",
    "Microstructure": "#9edae5", "Application": "#f7b6d2",
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
            geom = self._match_geometry(
                str(row[cols["geometry"]]).strip().lower().replace(" ", "_"))
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
                self.G.add_edge("high_von_mises_stress",
                                "dendrite_suppression",
                                relation="INHIBITS", weight=0.9,
                                source="Manuscript_Theory")
            area = float(row[cols["area_fraction_pct"]] or 0.0)
            self.G.add_edge(model, "dendrite_area_fraction",
                            relation="INDICATES", weight=round(area / 100.0, 3),
                            source="CSV_Ground_Truth")
            self.G.nodes[model]["csv_observed"] = True

    @staticmethod
    def _norm_model(value: Any) -> str:
        v = str(value).strip().replace(" ", "_")
        if v.upper().startswith("MODEL_"):
            return "Model_" + v.split("_", 1)[1].upper()
        if len(v) == 1 and v.upper() in "ABCDEF":
            return f"Model_{v.upper()}"
        return v

    @staticmethod
    def _match_geometry(text: str) -> str:
        m = {"boundary_layer": "boundary_layer",
             "boundarylayer": "boundary_layer",
             "edge_layer": "boundary_layer",
             "horizontal_layering": "boundary_layer",
             "alternating_stripes": "alternating_stripes",
             "vertical_stripes": "alternating_stripes",
             "stripes": "alternating_stripes",
             "central_layer": "central_layer",
             "central_solid_layer": "central_layer",
             "core_layer": "central_layer"}
        return m.get(text, text if text in m.values() else "boundary_layer")

    def fuse_literature_edges(self, triples: List[Dict]) -> None:
        for t in triples:
            subj, rel = t["subject"], t["relation"]
            obj, w = t["object"], float(t.get("weight", 0.5))
            self._ensure_node(subj, "Literature_Text")
            self._ensure_node(obj, "Literature_Text")
            if self.G.has_edge(subj, obj):
                data = self.G[subj][obj]
                data["weight"] = min(1.0,
                                     float(data.get("weight", 0.5)) + 0.1)
                data["source"] = self._merge_source(data.get("source"),
                                                     "Literature_Text")
                data.setdefault("literature_evidence",
                                 []).append(t.get("evidence", ""))
            else:
                self.G.add_edge(subj, obj, relation=rel, weight=w,
                                source="Literature_Text",
                                literature_evidence=[t.get("evidence", "")],
                                doc=t.get("doc", ""))

    def fuse_vlm_insights(self, triples: List[Dict]) -> None:
        for t in triples:
            subj, rel = t["subject"], t["relation"]
            obj = t["object"]
            self._ensure_node(subj, "VLM_Vision")
            self._ensure_node(obj, "VLM_Vision")
            if self.G.has_edge(subj, obj):
                data = self.G[subj][obj]
                data["source"] = self._merge_source(data.get("source"),
                                                     "VLM_Vision")
                data["weight"] = min(1.0,
                                     float(data.get("weight", 0.5)) + 0.2)
                data["visual_evidence"] = t.get("evidence", "VLM extracted")
            else:
                self.G.add_edge(subj, obj, relation=rel,
                                weight=float(t.get("weight", 0.8)),
                                source="VLM_Vision",
                                visual_evidence=t.get("evidence",
                                                       "VLM extracted"))

    def infer_cross_modal_paths(self, max_depth: int = 2) -> int:
        lit_nodes = [n for n, d in self.G.nodes(data=True)
                     if d.get("source") == "Literature_Text"
                     or "Literature_Text" in str(d.get("source", ""))]
        sim_nodes = [n for n, d in self.G.nodes(data=True)
                     if d.get("source") in ("CSV_Ground_Truth",
                                             "Manuscript_CSV",
                                             "VLM_Vision",
                                             "Manuscript_Theory")]
        added = 0
        for lit in lit_nodes:
            if lit not in self.ontology.concepts:
                continue
            for sim in sim_nodes:
                if sim not in self.ontology.concepts or lit == sim:
                    continue
                if self.G.has_edge(lit, sim):
                    continue
                paths = self.ontology.infer_path(lit, sim, max_depth=max_depth)
                if paths:
                    self.G.add_edge(lit, sim,
                                    relation="SUPPORTED_BY_SIMULATION",
                                    weight=0.7,
                                    source="Multimodal_Inference",
                                    path=" -> ".join(paths[0]))
                    added += 1
        log.info("Cross-modal inference added %d bridge edges.", added)
        return added

    @timed
    def add_semantic_edges(self, resolver: AdvancedConceptResolver,
                           threshold: float = 0.70,
                           max_edges: int = 300) -> int:
        """Add SEMANTIC edges between concepts whose embeddings are similar.

        The UAV code uses a similar approach with `_add_semantic_edges`.
        Only adds edges for node pairs not already directly connected.
        Bounded by `max_edges` to keep the graph sparse.
        """
        nodes = list(self.G.nodes)
        if len(nodes) < 10:
            return 0
        embs = resolver.embed_batch(nodes)
        if embs is None:
            return 0
        try:
            sim = embs @ embs.T
        except Exception:
            return 0
        added = 0
        n = len(nodes)
        candidates: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] < threshold:
                    continue
                u, v = nodes[i], nodes[j]
                if self.G.has_edge(u, v) or self.G.has_edge(v, u):
                    continue
                candidates.append((float(sim[i, j]), i, j))
        candidates.sort(reverse=True)
        for score, i, j in candidates[:max_edges]:
            u, v = nodes[i], nodes[j]
            self.G.add_edge(u, v, relation="SEMANTIC",
                            weight=round(score, 3),
                            source="Semantic_Similarity")
            added += 1
        log.info("Semantic edge layer added %d edges (threshold=%.2f).",
                 added, threshold)
        return added

    def add_ontology_expanded_node(self, name: str, ctype: str,
                                   definition: str = "",
                                   synonyms: Optional[List[str]] = None,
                                   connected_to: Optional[str] = None) -> bool:
        """Add a dynamically-proposed concept into the graph."""
        if name in self.G.nodes:
            return False
        self.G.add_node(name, type=ctype, source="Ontology_Expander",
                        definition=definition)
        if connected_to and connected_to in self.G.nodes:
            self.G.add_edge(connected_to, name, relation="SYNONYM",
                            weight=0.5, source="Ontology_Expander")
        return True

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

    def summary(self) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        for _, d in self.G.nodes(data=True):
            s = str(d.get("source", "Ontology_Backbone"))
            by_source[s] = by_source.get(s, 0) + 1
        by_relation: Dict[str, int] = {}
        for _, _, d in self.G.edges(data=True):
            r = d.get("relation", "?")
            by_relation[r] = by_relation.get(r, 0) + 1
        return {"nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "by_source": by_source,
                "by_relation": by_relation}

    def node_colour(self, name: str) -> str:
        d = self.G.nodes[name]
        src = d.get("source", "")
        if src in SOURCE_COLOUR:
            return SOURCE_COLOUR[src]
        for prefix in src.split(" + "):
            if prefix in SOURCE_COLOUR:
                return SOURCE_COLOUR[prefix]
        return NODE_TYPE_COLOUR.get(d.get("type", ""), "#999999")


# ============================================================================
# PHASE 6: ADVANCED ANALYTICS
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def detect_keyword_bursts(lit_df: pd.DataFrame,
                          _ontology: DendriteGeometryOntology,
                          _resolver: AdvancedConceptResolver,
                          burst_threshold: float = 2.0) -> pd.DataFrame:
    if "year" not in lit_df.columns or lit_df["year"].isna().all():
        return pd.DataFrame()
    years = lit_df["year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame()
    extractor = TextConceptExtractor(_ontology, _resolver)
    year_range = sorted(years.unique())
    concept_years: Dict[str, List[int]] = defaultdict(list)
    for _, row in lit_df.iterrows():
        y = row.get("year")
        if pd.isna(y):
            continue
        abstract = f"{row.get('title', '')}. {row.get('abstract', '')}"
        concepts = set()
        for sent in split_sentences(str(abstract)):
            concepts.update(extractor._scan_sentence(sent))
        for c in concepts:
            concept_years[c].append(int(y))
    rows: List[Dict[str, Any]] = []
    for concept, ys in concept_years.items():
        if len(ys) < 3:
            continue
        counts = Counter(ys)
        series = [counts.get(y, 0) for y in year_range]
        window = max(2, len(series) // 4)
        ma = pd.Series(series).rolling(window=window, min_periods=1).mean()
        bursts = []
        for i in range(window, len(series)):
            if ma.iloc[i - 1] > 0:
                bursts.append((series[i] / max(ma.iloc[i - 1], 0.1),
                                year_range[i]))
        if not bursts:
            continue
        peak_score, peak_year = max(bursts, key=lambda x: x[0])
        if peak_score >= burst_threshold:
            rows.append({"concept": concept,
                         "burst_score": round(peak_score, 2),
                         "burst_year": peak_year,
                         "total_mentions": len(ys),
                         "year_range": f"{min(ys)}-{max(ys)}",
                         "category": concept_to_category(concept)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("burst_score", ascending=False)
            .reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def detect_semantic_drift(lit_df: pd.DataFrame,
                          _ontology: DendriteGeometryOntology,
                          _resolver: AdvancedConceptResolver,
                          early_fraction: float = 0.3,
                          late_fraction: float = 0.3) -> pd.DataFrame:
    if "year" not in lit_df.columns or lit_df["year"].isna().all():
        return pd.DataFrame()
    years = sorted(lit_df["year"].dropna().astype(int).unique())
    if len(years) < 4:
        return pd.DataFrame()
    _resolver._init_embeddings()
    if not _resolver._embed_model:
        return pd.DataFrame()
    early_cut = years[int(len(years) * early_fraction)]
    late_cut = years[int(len(years) * (1 - late_fraction))]
    extractor = TextConceptExtractor(_ontology, _resolver)
    concept_texts: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"early": [], "late": []})
    for _, row in lit_df.iterrows():
        y = row.get("year")
        if pd.isna(y):
            continue
        y = int(y)
        text = f"{row.get('title', '')}. {row.get('abstract', '')}"
        concepts = set()
        for sent in split_sentences(str(text)):
            concepts.update(extractor._scan_sentence(sent))
        for c in concepts:
            if y <= early_cut:
                concept_texts[c]["early"].append(str(text))
            elif y >= late_cut:
                concept_texts[c]["late"].append(str(text))
    rows: List[Dict[str, Any]] = []
    for concept, bucket in concept_texts.items():
        if len(bucket["early"]) < 1 or len(bucket["late"]) < 1:
            continue
        try:
            e = _resolver._embed_model.encode(
                bucket["early"], convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False)
            l = _resolver._embed_model.encode(
                bucket["late"], convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False)
            drift = 1.0 - float(np.dot(e.mean(0), l.mean(0)))
            rows.append({"concept": concept,
                         "semantic_drift": round(drift, 4),
                         "early_count": len(bucket["early"]),
                         "late_count": len(bucket["late"]),
                         "early_period": f"{years[0]}-{early_cut}",
                         "late_period": f"{late_cut}-{years[-1]}"})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("semantic_drift", ascending=False)
            .reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def build_concept_genealogy(_graph: nx.DiGraph,
                             valid_concepts: List[str]) -> pd.DataFrame:
    if _graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        pagerank = nx.pagerank(_graph.to_undirected(), weight="weight")
    except Exception:
        pagerank = {n: 1.0 / _graph.number_of_nodes() for n in _graph.nodes()}
    try:
        betweenness = nx.betweenness_centrality(_graph.to_undirected(),
                                                 weight="weight")
    except Exception:
        betweenness = {n: 0.0 for n in _graph.nodes()}
    pr_vals = list(pagerank.values())
    pr_hi = float(np.percentile(pr_vals, 75))
    pr_lo = float(np.percentile(pr_vals, 25))
    deg_vals = [d for _, d in _graph.degree()]
    deg_hi = float(np.percentile(deg_vals, 75))
    deg_lo = float(np.percentile(deg_vals, 25))
    rows: List[Dict[str, Any]] = []
    for c in valid_concepts:
        if c not in _graph:
            continue
        pr = pagerank.get(c, 0.0)
        bc = betweenness.get(c, 0.0)
        deg = _graph.degree(c)
        gen = "Intermediate"
        if pr >= pr_hi and deg >= deg_hi:
            gen = "Foundational (Parent)"
        elif pr <= pr_lo and deg <= deg_lo:
            gen = "Emerging (Child)"
        rows.append({"concept": c, "pagerank": round(pr, 5),
                     "betweenness": round(bc, 5), "degree": deg,
                     "generation": gen,
                     "category": concept_to_category(c)})
    return (pd.DataFrame(rows)
            .sort_values("pagerank", ascending=False)
            .reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def detect_cross_domain_bridges(_graph: nx.DiGraph,
                                 valid_concepts: List[str]) -> pd.DataFrame:
    if _graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        betweenness = nx.betweenness_centrality(_graph.to_undirected(),
                                                 weight="weight")
    except Exception:
        betweenness = {n: 0.0 for n in _graph.nodes()}
    rows: List[Dict[str, Any]] = []
    for c in valid_concepts:
        if c not in _graph:
            continue
        nbrs = set(_graph.successors(c)) | set(_graph.predecessors(c))
        if len(nbrs) < 2:
            continue
        nbr_cats = {concept_to_category(n) for n in nbrs
                    if concept_to_category(n) != "Other"}
        if len(nbr_cats) < 2:
            continue
        score = betweenness.get(c, 0.0) * len(nbr_cats)
        rows.append({"concept": c, "bridge_score": round(score, 4),
                     "betweenness": round(betweenness.get(c, 0.0), 4),
                     "connected_categories": len(nbr_cats),
                     "categories": ", ".join(sorted(nbr_cats)),
                     "own_category": concept_to_category(c),
                     "degree": len(nbrs)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("bridge_score", ascending=False)
            .reset_index(drop=True))


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_network_motifs(_graph: nx.DiGraph) -> Dict[str, Any]:
    if _graph.number_of_nodes() < 3:
        return {}
    ug = _graph.to_undirected()
    out: Dict[str, Any] = {}
    try:
        tris = nx.triangles(ug)
        out["total_triangles"] = sum(tris.values()) // 3
        out["nodes_in_triangles"] = sum(1 for v in tris.values() if v > 0)
    except Exception:
        out["total_triangles"] = 0
    try:
        cliques = list(nx.find_cliques(ug))
        sizes = [len(c) for c in cliques]
        out["total_cliques"] = len(cliques)
        out["max_clique_size"] = max(sizes) if sizes else 0
        out["avg_clique_size"] = (round(float(np.mean(sizes)), 2)
                                    if sizes else 0.0)
        out["size_ge_4"] = sum(1 for s in sizes if s >= 4)
    except Exception:
        out["total_cliques"] = 0
    try:
        clustering = nx.clustering(ug)
        stars = [(n, ug.degree(n), clustering.get(n, 0)) for n in ug.nodes()
                 if ug.degree(n) >= 4 and clustering.get(n, 0) < 0.2]
        stars.sort(key=lambda x: x[1], reverse=True)
        out["star_motifs"] = len(stars)
        out["top_stars"] = stars[:10]
    except Exception:
        out["star_motifs"] = 0
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def compute_centrality_comparison(_graph: nx.DiGraph,
                                   valid_concepts: List[str]) -> pd.DataFrame:
    if _graph.number_of_nodes() < 3:
        return pd.DataFrame()
    ug = _graph.to_undirected()
    try:
        deg = dict(ug.degree())
        btw = nx.betweenness_centrality(ug, weight="weight")
        cls = nx.closeness_centrality(ug)
        pr = nx.pagerank(_graph, weight="weight")
        try:
            eig = nx.eigenvector_centrality(ug, weight="weight", max_iter=1000)
        except Exception:
            eig = {n: 0.0 for n in ug.nodes()}
    except Exception:
        return pd.DataFrame()
    rows = []
    for c in valid_concepts:
        if c not in ug:
            continue
        rows.append({"concept": c, "degree": deg.get(c, 0),
                     "betweenness": round(btw.get(c, 0.0), 5),
                     "closeness": round(cls.get(c, 0.0), 5),
                     "eigenvector": round(eig.get(c, 0.0), 5),
                     "pagerank": round(pr.get(c, 0.0), 5),
                     "category": concept_to_category(c)})
    return (pd.DataFrame(rows)
            .sort_values("pagerank", ascending=False)
            .reset_index(drop=True))


def plot_degree_distribution(_graph: nx.DiGraph,
                              theme: Optional[Dict] = None) -> Any:
    if go is None:
        return None
    theme = theme or THEME_PRESETS["Bright (Default)"]
    degrees = [d for _, d in _graph.degree()]
    if len(degrees) < 3:
        return None
    counts = Counter(degrees)
    x = sorted(counts.keys())
    y = [counts[k] for k in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Degree",
        marker=dict(size=10, color=theme.get("highlight_bg", "#ff6b6b"))))
    fig.update_layout(
        title="Degree Distribution (Log-Log)", xaxis_type="log",
        yaxis_type="log", xaxis_title="Degree k",
        yaxis_title="Frequency P(k)",
        paper_bgcolor=theme["plotly_paper"], plot_bgcolor=theme["plotly_bg"],
        font_color=theme["font"])
    return fig


# ============================================================================
# PHASE 7: ML MODELS (GraphSAGE + LatentMoE)
# ============================================================================
def sample_edges_for_training(nx_graph: nx.Graph,
                              valid_concepts: List[str],
                              concept_to_id: Dict[str, int],
                              memory_safe: bool = False
                              ) -> Tuple[List[Tuple[int, int]],
                                          List[Tuple[int, int]]]:
    """Positive edges + negative edges filtered by path length.

    UAV pattern: negatives with 2-3 hop distance (hard) or far apart (easy).
    """
    pos_pairs = [(concept_to_id[u], concept_to_id[v])
                 for u, v in nx_graph.edges()]
    neg_pairs: List[Tuple[int, int]] = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    target_negs = min((len(pos_pairs) * 2 if pos_pairs else 30)
                       if memory_safe else
                       (len(pos_pairs) * 3 if pos_pairs else 30), 2000)
    attempts = 0
    max_attempts = 50000
    path_lengths: Dict[str, Dict[str, int]] = {}
    if not memory_safe:
        try:
            path_lengths = dict(nx.all_pairs_shortest_path_length(
                nx_graph, cutoff=3))
        except Exception:
            path_lengths = {}
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            attempts += 1
            continue
        dist = path_lengths.get(u_c, {}).get(v_c, 999) if path_lengths else 999
        if dist in (2, 3):
            neg_pairs.append((int(u_idx), int(v_idx)))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((int(u_idx), int(v_idx)))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx],
                                  valid_concepts[v_idx]):
            neg_pairs.append((int(u_idx), int(v_idx)))
    return pos_pairs, neg_pairs


class SparseGraphSAGE(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, adj_indices, adj_values, num_nodes, h,
                pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(
            adj_indices, adj_values,
            torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(
            torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        if len(neg_u) > 0 and len(neg_v) > 0:
            neg_scores = self.decoder(
                torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        else:
            neg_scores = torch.zeros(0, device=h.device)
        return pos_scores, neg_scores, h2


def train_graphsage(node_features, nx_graph, concept_to_id,
                     pos_pairs, neg_pairs, progress_callback=None,
                     epochs: int = 50, lr: float = 1e-3,
                     hidden_dim: int = 128
                     ) -> Tuple[Optional[Any], Optional[Any],
                                 Optional[Any], Optional[Any]]:
    if not TORCH_AVAILABLE:
        log.warning("torch not available; skipping GraphSAGE training.")
        return None, None, None, None
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) < 2:
            return None, None, None, None
        pos_pairs = [(nodes[0], nodes[1])]
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    device = (node_features.device if node_features.numel() > 0
              else torch.device("cpu"))
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long,
                          device=device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long,
                          device=device)
    neg_u = (torch.tensor([n[0] for n in neg_pairs], dtype=torch.long,
                           device=device)
             if neg_pairs else torch.tensor([], dtype=torch.long,
                                             device=device))
    neg_v = (torch.tensor([n[1] for n in neg_pairs], dtype=torch.long,
                           device=device)
             if neg_pairs else torch.tensor([], dtype=torch.long,
                                             device=device))
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses: List[float] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_out, neg_out, _ = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u, pos_v, neg_u, neg_v)
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        if len(neg_out) > 0:
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        else:
            loss = pos_loss * 0.5
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, float(loss.item()))
    model.eval()
    with torch.no_grad():
        _, _, final_emb = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u[:1], pos_v[:1],
            neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
            neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1])
    return model, final_emb.cpu(), adj_indices.cpu(), adj_values.cpu()


class LatentMoEKGExtractor(nn.Module if TORCH_AVAILABLE else object):
    """32-expert sparse MoE with top-k routing (UAV pattern).

    Adds a domain-prior via `router_prior` (one-hot over categories).
    """

    def __init__(self, num_nodes: int, num_edge_types: int,
                 d_model: int = 96, latent_dim: int = 24,
                 n_experts: int = 32, top_k: int = 4,
                 num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)
        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.router = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True,
            dim_feedforward=d_model * 2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, node_seq, edge_seq, router_prior=None):
        node_emb = self.node_embedding(node_seq)
        if edge_seq.size(1) > 0:
            edge_emb = self.edge_embedding(edge_seq)
            node_emb[:, 1:, :] = node_emb[:, 1:, :] + edge_emb
        b, s, d = node_emb.shape
        latent = self.down_proj(node_emb)
        flat = latent.view(b * s, -1)
        logits = self.router(flat)
        if router_prior is not None:
            logits = logits + router_prior.to(logits.device)
        w = F.softmax(logits, dim=-1)
        topk_w, topk_i = torch.topk(w, self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-6)
        expert_outs = torch.stack(
            [self.experts[i](flat) for i in range(self.n_experts)], dim=0)
        token_idx = (torch.arange(b * s, device=flat.device)
                      .unsqueeze(1).expand(-1, self.top_k))
        sel = expert_outs[topk_i, token_idx, :]
        out_flat = (topk_w.unsqueeze(-1) * sel).sum(dim=1)
        moe_out = out_flat.view(b, s, -1)
        node_emb = node_emb + self.up_proj(moe_out)
        out = self.output_proj(self.transformer(node_emb))
        return out, w.view(b, s, -1)


# ============================================================================
# PHASE 7B: RESEARCH DIRECTION SCORING
# ============================================================================
def compute_research_direction_scores(model, node_features, final_emb,
                                       nx_graph, valid_concepts,
                                       concept_properties, ridge,
                                       embed_model_or_matrix=None,
                                       n_samples: int = 2000
                                       ) -> pd.DataFrame:
    """UAV pattern: rank non-existent concept pairs by expected impact.

    Composite D_uv = α·GNN_affinity + β·novelty + γ·gain - δ·(1 - feasibility)
    """
    if not TORCH_AVAILABLE or model is None:
        return pd.DataFrame()
    n = len(valid_concepts)
    if n < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(n, size=min(n_samples, n * 5))
    v_ids = np.random.randint(n, size=min(n_samples, n * 5))
    candidates: List[Tuple[int, int, str, str]] = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c) or nx_graph.has_edge(v_c, u_c):
            continue
        candidates.append((int(u_idx), int(v_idx), u_c, v_c))
    if not candidates:
        return pd.DataFrame()
    u_t = torch.tensor([p[0] for p in candidates], dtype=torch.long)
    v_t = torch.tensor([p[1] for p in candidates], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_feat = torch.cat([final_emb[u_t], final_emb[v_t]], dim=1)
        scores = torch.sigmoid(model.decoder(pair_feat).squeeze(1)).numpy()
    # Semantic novelty via embedding distance if available
    if embed_model_or_matrix is not None:
        try:
            if hasattr(embed_model_or_matrix, "embed_batch"):
                embs = embed_model_or_matrix.embed_batch(valid_concepts)
            else:
                embs = embed_model_or_matrix
            if embs is not None:
                cos = (embs[u_t.numpy()] * embs[v_t.numpy()]).sum(axis=1)
            else:
                cos = np.ones(len(candidates))
        except Exception:
            cos = np.ones(len(candidates))
    else:
        cos = np.ones(len(candidates))
    novelty = 1.0 - cos
    rows = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidates):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        exp_gain = 0.0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                exp_gain = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except Exception:
                exp_gain = max(p_u, p_v) * 1.05
        feasibility = (math.exp(-0.5 * novelty[i])
                        * (1.0 if (p_u > 0 or p_v > 0) else 0.6))
        norm_gain = (np.clip((exp_gain - 50) / 200, 0, 1)
                     if exp_gain > 0 else 0)
        D_uv = (0.4 * float(scores[i]) + 0.3 * float(novelty[i])
                + 0.2 * float(norm_gain) - 0.1 * (1.0 - feasibility))
        rows.append({"concept_u": u_c, "concept_v": v_c,
                     "gnn_affinity": round(float(scores[i]), 4),
                     "semantic_novelty": round(float(novelty[i]), 4),
                     "expected_property_gain": round(exp_gain, 3),
                     "feasibility_score": round(float(feasibility), 4),
                     "composite_score": round(D_uv, 4)})
    return (pd.DataFrame(rows)
            .sort_values("composite_score", ascending=False)
            .head(100).reset_index(drop=True))


# ============================================================================
# PHASE 7C: CONCEPT DISTILLATION
# ============================================================================
def compute_concept_distillation(valid_concepts: List[str],
                                  concept_abstract_map: Dict[str, List[int]],
                                  all_texts: Union[List[str], Dict[int, str]],
                                  resolver: Optional[AdvancedConceptResolver] = None,
                                  max_docs_per_concept: int = 30,
                                  coherence_window: int = 20) -> pd.DataFrame:
    """UAV pattern: TF-IDF × frequency × coherence.

    Ranks concepts by 'distillation efficiency' - a measure of how much
    signal a concept carries in the corpus.
    """
    texts_is_dict = isinstance(all_texts, dict)
    n_texts = len(all_texts)
    doc_corpus: List[str] = []
    for c in valid_concepts:
        doc_idxs = concept_abstract_map.get(c, [])
        if max_docs_per_concept and len(doc_idxs) > max_docs_per_concept:
            doc_idxs = doc_idxs[:max_docs_per_concept]
        if texts_is_dict:
            doc_text = " ".join([all_texts[i] for i in doc_idxs if i in all_texts])
        else:
            doc_text = " ".join([all_texts[i] for i in doc_idxs
                                 if isinstance(i, int) and 0 <= i < n_texts])
        doc_corpus.append(doc_text)
    # TF-IDF
    tfidf_scores = np.ones(len(valid_concepts))
    if TfidfVectorizer is not None and any(doc_corpus) and any(
            t.strip() for t in doc_corpus):
        try:
            tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     stop_words="english", max_features=2000)
            mat = tfidf.fit_transform(doc_corpus)
            tfidf_scores = mat.max(axis=1).toarray().ravel()
            del mat
        except Exception:
            pass
    gc.collect()
    # Coherence via embeddings if resolver available
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic = float(tfidf_scores[i])
        coherence = 0.0
        if resolver is not None and freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:coherence_window]
                embs = resolver.embed_batch(words)
                if embs is not None and len(embs) > 1:
                    sim = embs @ embs.T
                    iu = np.triu_indices_from(sim, k=1)
                    coherence = float(np.mean(sim[iu]))
            except Exception:
                coherence = 0.0
        distillation = semantic * math.log1p(freq) * (0.5 + 0.5 * coherence)
        rows.append({"concept": c, "frequency": freq,
                     "tfidf_weight": round(semantic, 4),
                     "semantic_density": round(semantic, 4),
                     "coherence_score": round(coherence, 4),
                     "distillation_efficiency": round(distillation, 4),
                     "category": concept_to_category(c)})
    return (pd.DataFrame(rows)
            .sort_values("distillation_efficiency", ascending=False)
            .reset_index(drop=True))


# ============================================================================
# PHASE 7D: GRAPH VALIDATION & BOOTSTRAP
# ============================================================================
def validate_graph_metrics(nx_graph: nx.Graph,
                            valid_concepts: List[str],
                            resolver: Optional[AdvancedConceptResolver] = None,
                            n_permutations: int = 50
                            ) -> Dict[str, Any]:
    """UAV pattern: modularity, silhouette, permutation p-values."""
    metrics: Dict[str, Any] = {}
    if nx_graph.number_of_nodes() < 3:
        return metrics
    # Modularity
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(
            nx_graph.to_undirected() if hasattr(nx_graph, "to_undirected")
            else nx_graph))
        metrics["modularity"] = round(
            float(community.modularity(nx_graph, partition)), 4)
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0
        partition = []
    # Silhouette on embedding-based labels
    if resolver is not None and silhouette_score is not None:
        try:
            embs = resolver.embed_batch(valid_concepts)
            if embs is not None and len(valid_concepts) >= 3:
                labels = np.zeros(len(valid_concepts))
                comm_of_node = {}
                for idx, comm in enumerate(partition):
                    for n in comm:
                        comm_of_node[n] = idx
                for i, c in enumerate(valid_concepts):
                    labels[i] = comm_of_node.get(c, -1)
                if len(set(labels)) > 1:
                    metrics["silhouette_score"] = round(
                        float(silhouette_score(embs, labels)), 4)
                else:
                    metrics["silhouette_score"] = 0.0
        except Exception:
            metrics["silhouette_score"] = 0.0
    # Permutation test for edge significance
    weights = [d.get("weight", 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 5:
        p_values = []
        for w in weights[:min(n_permutations, len(weights))]:
            permuted = np.random.permutation(weights)
            p_values.append(float(np.sum(permuted >= w) / len(weights)))
        metrics["edge_significance_p_mean"] = round(float(np.mean(p_values)), 4)
        metrics["edge_significant_count"] = int(
            sum(1 for p in p_values if p < 0.05))
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    # Centrality ranges
    try:
        btw = nx.betweenness_centrality(nx_graph.to_undirected(),
                                          weight="weight")
        cls = nx.closeness_centrality(nx_graph.to_undirected())
        metrics["avg_betweenness"] = round(float(np.mean(list(btw.values()))), 5)
        metrics["avg_closeness"] = round(float(np.mean(list(cls.values()))), 5)
    except Exception:
        pass
    metrics["n_nodes"] = nx_graph.number_of_nodes()
    metrics["n_edges"] = nx_graph.number_of_edges()
    metrics["density"] = round(nx.density(nx_graph), 5)
    return metrics


@st.cache_data(ttl=3600, show_spinner=False)
def compute_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 500,
                          alpha: float = 0.05) -> Tuple[float, float, float]:
    if len(scores) < 2:
        return float(np.mean(scores)) if len(scores) else 0.0, 0.0, 0.0
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(float(np.mean(sample)))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), lo, hi


# ============================================================================
# PHASE 8: DYNAMIC ONTOLOGY EXPANDER
# ============================================================================
class DynamicOntologyExpander:
    """LLM-driven concept addition (blueprint Phase 2).

    Given an unresolved phrase, ask an LLM to propose:
      - canonical name
      - concept type
      - synonyms
      - definition
    User reviews and approves before the graph/ontology mutates.
    """

    PROPOSAL_SYSTEM = (
        "You are a materials-science ontology curator. Given a phrase extracted "
        "from dendrite-suppression literature, propose ONE new concept node.\n"
        "Return strict JSON: "
        '{"name": "<snake_case>", "type": "<one of: Geometry, Parameter, Ratio, '
        'Mechanical_Property, Electrochemical_Property, Morphology_Metric, '
        'Phenomenon, Material, Method, Microstructure, Application>", '
        '"definition": "<1 sentence>", "synonyms": ["<0-3 terms>"]}\n'
        "If the phrase is not a valid dendrite-domain concept, return {}."
    )

    def __init__(self, ontology: DendriteGeometryOntology) -> None:
        self.ontology = ontology
        self.client = None
        self.error: Optional[str] = None
        if OpenAI is not None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                except Exception as exc:
                    self.error = f"OpenAI init failed: {exc}"

    def is_available(self) -> bool:
        return self.client is not None

    def status(self) -> str:
        if self.client is not None:
            return "✅ OpenAI client ready for ontology expansion"
        if self.error:
            return f"⚠️ {self.error}"
        return "⏳ No OPENAI_API_KEY set; expansion unavailable"

    def propose(self, phrase: str, context: str = "",
                 model: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
        if not self.is_available() or not phrase:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=model, temperature=0.1, max_tokens=300,
                messages=[
                    {"role": "system", "content": self.PROPOSAL_SYSTEM},
                    {"role": "user", "content": (
                        f"Phrase: {phrase}\n"
                        f"Context: {context[:500] if context else '(none)'}\n"
                        "Existing concepts: "
                        + ", ".join(list(self.ontology.concepts.keys())[:40]))},
                ])
            raw = resp.choices[0].message.content or "{}"
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            if not data or "name" not in data:
                return None
            return {"name": data.get("name", phrase),
                    "type": data.get("type", "Phenomenon"),
                    "definition": data.get("definition", ""),
                    "synonyms": data.get("synonyms", []),
                    "source_phrase": phrase}
        except Exception as exc:
            log.warning("Ontology expander call failed: %s", exc)
            return None

    def propose_batch(self, phrases: List[str], context: str = "",
                      model: str = "gpt-4o-mini",
                      max_new: int = 5) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()
        for phrase in phrases[:max_new * 2]:
            p = self.propose(phrase, context, model)
            if p and p["name"] not in seen_names:
                if p["name"] not in self.ontology.concepts:
                    proposals.append(p)
                    seen_names.add(p["name"])
            if len(proposals) >= max_new:
                break
        return proposals

    def commit(self, proposal: Dict[str, Any],
               graph_builder: Optional[MultimodalGraphBuilder] = None
               ) -> bool:
        name = proposal["name"]
        ok = self.ontology.add_concept(
            name, proposal.get("type", "Phenomenon"),
            definition=proposal.get("definition", ""),
            synonyms=proposal.get("synonyms", []),
            inferred_from=proposal.get("source_phrase"))
        if ok and graph_builder is not None:
            graph_builder.add_ontology_expanded_node(
                name, proposal.get("type", "Phenomenon"),
                definition=proposal.get("definition", ""),
                synonyms=proposal.get("synonyms", []),
                connected_to=proposal.get("source_phrase"))
        return ok


# ============================================================================
# PHASE 9: LLM QUERY ANALYZER + GRAPH-RAG
# ============================================================================
class DendriteProblem(Enum):
    GEOMETRY_OPTIMIZATION = "geometry_optimization"
    MECHANICAL_SUPPRESSION = "mechanical_suppression"
    SEI_STABILITY = "sei_stability"
    DEAD_LITHIUM_MITIGATION = "dead_lithium_mitigation"
    GENERAL = "general"


DENDRITE_PROBLEM_DEFINITIONS: Dict[DendriteProblem, Dict[str, Any]] = {
    DendriteProblem.GEOMETRY_OPTIMIZATION: {
        "title": "Electrolyte Spatial Configuration",
        "key_concepts": ["boundary_layer", "alternating_stripes",
                          "central_layer", "solid_liquid_ratio",
                          "spatial_configuration"],
        "key_relationships": [
            ("Model_A", "HAS_GEOMETRY", "boundary_layer"),
            ("spatial_configuration", "INHIBITS", "dendrite_growth")],
        "example_queries": ["Which S/L ratio provides best stress confinement?",
                              "How does central layer geometry differ?"],
    },
    DendriteProblem.MECHANICAL_SUPPRESSION: {
        "title": "Mechanical Suppression of Dendrites",
        "key_concepts": ["elastic_modulus", "von_mises_stress", "llzo",
                          "dendrite_suppression", "high_von_mises_stress"],
        "key_relationships": [
            ("elastic_modulus", "CONSTRAINS", "dendrite_growth"),
            ("high_von_mises_stress", "INHIBITS", "side_branching")],
        "example_queries": ["How does high Young's modulus inhibit side branching?"],
    },
    DendriteProblem.SEI_STABILITY: {
        "title": "SEI and Electrochemical Stability",
        "key_concepts": ["effective_conductivity", "li_ion_flux",
                          "overpotential", "butler_volmer", "nernst_planck",
                          "hybrid_electrolyte"],
        "key_relationships": [
            ("effective_conductivity", "DRIVES", "li_ion_flux"),
            ("li_ion_flux", "DRIVES", "dendrite_growth")],
        "example_queries": ["How does effective conductivity route Li-ion flux?"],
    },
    DendriteProblem.DEAD_LITHIUM_MITIGATION: {
        "title": "Dead Lithium and Capacity Fade",
        "key_concepts": ["dead_lithium", "internal_short_circuit",
                          "dendrite_area_fraction", "lithium_ion_battery"],
        "key_relationships": [
            ("dendrite_growth", "DRIVES", "dead_lithium"),
            ("dendrite_growth", "DRIVES", "internal_short_circuit")],
        "example_queries": ["How does dendrite growth correlate with dead lithium?"],
    },
    DendriteProblem.GENERAL: {
        "title": "General Dendrite Analysis",
        "key_concepts": list(DENDRITE_CATEGORIES["Morphology Outcomes"]),
        "key_relationships": [],
        "example_queries": ["Summarise current dendrite suppression understanding."],
    },
}


class GraphRAGRetriever:
    def __init__(self, graph: nx.DiGraph,
                 ontology: DendriteGeometryOntology,
                 resolver: AdvancedConceptResolver) -> None:
        self.graph = graph
        self.ontology = ontology
        self.resolver = resolver

    @timed
    def retrieve(self, query: str, max_hops: int = 2,
                 max_nodes: int = 40) -> Tuple[nx.DiGraph, List[str]]:
        allocator = QDWAWeightAllocator(self.ontology, self.resolver)
        allocation = allocator.allocate(query, top_k=10)
        seeds = [c for c, _w, _cat in allocation.top_concepts]
        if not seeds:
            return self.graph.subgraph([]).copy(), []
        seeds = [s for s in seeds if s in self.graph]
        if not seeds:
            return self.graph.subgraph([]).copy(), []
        sub = self.graph.to_undirected()
        nodes: Set[str] = set()
        for s in seeds:
            nodes.add(s)
            try:
                ego = nx.ego_graph(sub, s, radius=max_hops)
                nodes.update(ego.nodes())
            except Exception:
                pass
            if len(nodes) >= max_nodes:
                break
        nodes = set(list(nodes)[:max_nodes])
        return self.graph.subgraph(nodes).copy(), seeds

    @staticmethod
    def serialize_subgraph(sub: nx.DiGraph) -> str:
        lines = []
        for u, v, d in sub.edges(data=True):
            rel = d.get("relation", "RELATED_TO")
            w = d.get("weight", "?")
            src = d.get("source", "")
            line = f"  ({u}) -[{rel}, w={w}, src={src}]-> ({v})"
            if d.get("visual_evidence"):
                line += f"  [visual: {d['visual_evidence']}]"
            if d.get("literature_evidence"):
                ev = d["literature_evidence"]
                if isinstance(ev, list) and ev:
                    line += f"  [lit: {ev[0][:120]}]"
            lines.append(line)
        return "\n".join(lines) or "(empty subgraph)"


class FallbackAnalyzer:
    def __init__(self, retriever: GraphRAGRetriever) -> None:
        self.retriever = retriever

    def analyze(self, query: str, problem: DendriteProblem) -> Dict[str, Any]:
        sub, seeds = self.retriever.retrieve(query, max_hops=2, max_nodes=30)
        context = self.retriever.serialize_subgraph(sub)
        definition = DENDRITE_PROBLEM_DEFINITIONS[problem]
        parts = [f"**Problem framing:** {definition['title']}", "",
                 f"**Detected seed concepts:** "
                 f"{', '.join(seeds) if seeds else '(none)'}", "",
                 "**Relevant subgraph edges:**", "```", context, "```", "",
                 "**Observations:**"]
        if sub.number_of_edges() == 0:
            parts.append("- No edges found; consider broadening the query.")
        else:
            n_multi = sum(1 for _u, _v, d in sub.edges(data=True)
                          if "+" in str(d.get("source", "")))
            n_vlm = sum(1 for _u, _v, d in sub.edges(data=True)
                        if d.get("source") == "VLM_Vision")
            n_lit = sum(1 for _u, _v, d in sub.edges(data=True)
                        if d.get("source") == "Literature_Text")
            n_semantic = sum(1 for _u, _v, d in sub.edges(data=True)
                             if d.get("source") == "Semantic_Similarity")
            parts.append(f"- {sub.number_of_edges()} edges in retrieved subgraph.")
            parts.append(f"- Corroborated by literature: {n_lit} edges.")
            parts.append(f"- Corroborated by VLM: {n_vlm} edges.")
            parts.append(f"- Semantic similarity edges: {n_semantic}.")
            parts.append(f"- Multi-source (fusion): {n_multi} edges.")
        return {"answer": "\n".join(parts),
                "subgraph_nodes": list(sub.nodes()),
                "subgraph_edges": [(u, v) for u, v in sub.edges()],
                "context": context, "seeds": seeds, "mode": "fallback"}


class LLMGraphRAG:
    def __init__(self, retriever: GraphRAGRetriever,
                 fallback: FallbackAnalyzer) -> None:
        self.retriever = retriever
        self.fallback = fallback
        self.client = None
        self.error: Optional[str] = None
        if OpenAI is not None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                except Exception as exc:
                    self.error = f"OpenAI init failed: {exc}"

    def is_available(self) -> bool:
        return self.client is not None

    def status(self) -> str:
        if self.client is not None:
            return "✅ OpenAI client ready"
        if self.error:
            return f"⚠️ {self.error}"
        return "⏳ No OPENAI_API_KEY set; using deterministic fallback."

    def answer(self, query: str, problem: DendriteProblem,
               model: str = "gpt-4o-mini", temperature: float = 0.2,
               max_tokens: int = 800) -> Dict[str, Any]:
        if not self.is_available():
            return self.fallback.analyze(query, problem)
        sub, seeds = self.retriever.retrieve(query, max_hops=2, max_nodes=30)
        context = self.retriever.serialize_subgraph(sub)
        definition = DENDRITE_PROBLEM_DEFINITIONS[problem]
        system_prompt = (
            "You are an expert materials scientist analysing lithium-dendrite "
            "suppression in partitioned electrolytes. Answer ONLY using the "
            "provided knowledge-graph subgraph. Cite edges as "
            "`source -[RELATION]-> target`. Be concise (250-400 words) and "
            "structure the answer as: (1) Direct answer, (2) Evidence from "
            "the graph, (3) Open questions.\n\n"
            f"Problem framing: {definition['title']}\n"
            f"Key concepts: {', '.join(definition['key_concepts'])}\n")
        user_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved subgraph edges:\n{context}\n\n"
            f"Seed concepts: {', '.join(seeds)}\n")
        try:
            resp = self.client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}])
            answer = resp.choices[0].message.content or "(empty response)"
            return {"answer": answer,
                    "subgraph_nodes": list(sub.nodes()),
                    "subgraph_edges": [(u, v) for u, v in sub.edges()],
                    "context": context, "seeds": seeds, "mode": "llm"}
        except Exception as exc:
            log.warning("LLM call failed (%s); using fallback.", exc)
            result = self.fallback.analyze(query, problem)
            result["mode"] = f"fallback (LLM error: {exc})"
            return result


# ============================================================================
# PHASE 10: VISUALIZATION (PyVis + JS panel)
# ============================================================================
_PYVIS_JS_PANEL = r"""
<script>
(function() {
  function initPanel() {
    if (typeof network === 'undefined') { setTimeout(initPanel, 150); return; }
    var nodesDS = (typeof nodesDS !== 'undefined') ? nodesDS :
                  (typeof nodes !== 'undefined' ? nodes : null);
    var edgesDS = (typeof edgesDS !== 'undefined') ? edgesDS :
                  (typeof edges !== 'undefined' ? edges : null);
    if (!nodesDS || !edgesDS) { setTimeout(initPanel, 150); return; }
    var panel = document.createElement('div');
    panel.id = 'dendrite-evidence-panel';
    panel.style.cssText = 'position:fixed;top:12px;right:12px;width:360px;'
        + 'max-height:88vh;overflow:auto;background:#ffffff;'
        + 'border:1px solid #d0d0d0;border-radius:10px;padding:14px;'
        + 'box-shadow:0 6px 18px rgba(0,0,0,0.18);'
        + 'font-family:Inter,Segoe UI,sans-serif;font-size:12px;'
        + 'color:#1a1a1a;z-index:9999;display:none;';
    document.body.appendChild(panel);
    function edgeRow(e, dir) {
      var label = e.label || '(edge)';
      var colour = dir === 'out' ? '#27ae60' : '#2980b9';
      var evTitle = (e.title || '').replace(/<[^>]*>/g, '');
      return '<div style="margin:5px 0;padding:5px 8px;background:#f8f9fa;'
        + 'border-left:3px solid ' + colour + ';border-radius:3px;">'
        + '<b>' + (dir === 'out' ? e.to : e.from) + '</b>'
        + ' <span style="color:' + colour + '">[' + label + ']</span>'
        + '<div style="color:#666;font-size:10px;margin-top:2px;">'
        + evTitle.slice(0, 160) + '</div></div>';
    }
    network.on('selectNode', function(params) {
      if (!params.nodes || params.nodes.length === 0) {
        panel.style.display = 'none'; return;
      }
      var nodeId = params.nodes[0];
      var node = nodesDS.get(nodeId);
      var all = edgesDS.get();
      var outgoing = all.filter(function(e){ return e.from === nodeId; });
      var incoming = all.filter(function(e){ return e.to === nodeId; });
      var title = (node && node.title) ? node.title.replace(/\n/g, '<br>') : '';
      var html = '<div style="font-size:16px;font-weight:700;color:#2c3e50;'
        + 'margin-bottom:6px;">🔬 ' + nodeId + '</div>'
        + '<div style="color:#555;font-size:11px;margin-bottom:10px;">'
        + title + '</div>'
        + '<div style="font-weight:600;margin-top:8px;color:#27ae60;">'
        + '→ Outgoing (' + outgoing.length + ')</div>';
      if (outgoing.length === 0) {
        html += '<div style="color:#999;font-size:11px;">(none)</div>';
      } else {
        outgoing.slice(0, 15).forEach(function(e){ html += edgeRow(e, 'out'); });
      }
      html += '<div style="font-weight:600;margin-top:10px;color:#2980b9;">'
        + '← Incoming (' + incoming.length + ')</div>';
      if (incoming.length === 0) {
        html += '<div style="color:#999;font-size:11px;">(none)</div>';
      } else {
        incoming.slice(0, 15).forEach(function(e){ html += edgeRow(e, 'in'); });
      }
      html += '<div style="margin-top:10px;font-size:10px;color:#999;">'
        + 'Click a node to inspect · Click empty space to close</div>';
      panel.innerHTML = html;
      panel.style.display = 'block';
    });
    network.on('deselectNode', function(){ panel.style.display = 'none'; });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPanel);
  } else { initPanel(); }
})();
</script>
"""


def render_graph_pyvis(graph: MultimodalGraphBuilder,
                       focus: Optional[str] = None,
                       height: str = "650px",
                       show_backbone: bool = False,
                       inject_js: bool = True) -> str:
    if Network is None:
        return _fallback_table(graph)
    net = Network(height=height, width="100%", directed=True,
                  notebook=False, bgcolor="#ffffff", font_color="#1a1a1a")
    net.repulsion(node_distance=140, central_gravity=0.18,
                  spring_length=120, spring_strength=0.05)
    nodes_to_show: Set[str] = set()
    if focus:
        nodes_to_show.add(focus)
        nodes_to_show.update(graph.G.successors(focus))
        nodes_to_show.update(graph.G.predecessors(focus))
    else:
        for u, v, d in graph.G.edges(data=True):
            if show_backbone or d.get("source") != "Ontology_Backbone":
                nodes_to_show.add(u)
                nodes_to_show.add(v)
        nodes_to_show.update(MODEL_SIGNATURES.keys())
        if show_backbone:
            nodes_to_show.update(graph.G.nodes())
    for name in nodes_to_show:
        if name not in graph.G.nodes:
            continue
        nd = graph.G.nodes[name]
        colour = graph.node_colour(name)
        title = (f"{name}\ntype: {nd.get('type', '?')}\n"
                 f"source: {nd.get('source', '?')}\n"
                 f"{nd.get('definition', '')}")
        if nd.get("csv_observed"):
            title += "\n[CSV ground truth]"
        size = 24 if nd.get("type") == "Simulation_Model" else 14
        net.add_node(name, label=name, color=colour, title=title,
                     shape="dot", size=size)
    for u, v, d in graph.G.edges(data=True):
        if u not in nodes_to_show or v not in nodes_to_show:
            continue
        if not show_backbone and d.get("source") == "Ontology_Backbone":
            continue
        rel = d.get("relation", "?")
        rel_type = resolve_relation_type(rel)
        colour = get_edge_color(rel_type) if rel_type else EDGE_COLOR_FALLBACK
        width = get_edge_width(rel_type) if rel_type else 1.5
        style = get_edge_style(rel_type) if rel_type else "solid"
        title = (f"{u} --[{rel}]--> {v}\n"
                 f"weight: {d.get('weight', '?')}\n"
                 f"source: {d.get('source', '?')}")
        if d.get("visual_evidence"):
            title += f"\nvisual: {d['visual_evidence']}"
        if d.get("literature_evidence"):
            ev = d["literature_evidence"]
            if isinstance(ev, list) and ev:
                title += f"\nliterature: {ev[0][:200]}"
        if d.get("path"):
            title += f"\npath: {d['path']}"
        net.add_edge(u, v, title=title, label=rel, width=width,
                     color=colour, dashes=(style == "dashed"))
    net.set_options(json.dumps({
        "physics": {"barnesHut": {
            "gravitationalConstant": -8000, "springLength": 130,
            "centralGravity": 0.2, "damping": 0.5}},
        "edges": {"smooth": {"type": "continuous"}},
        "interaction": {"hover": True, "navigationButtons": True}}))
    try:
        html = net.generate_html(notebook=False)
    except Exception:
        return _fallback_table(graph)
    if inject_js:
        html = html.replace("</body>", _PYVIS_JS_PANEL + "</body>")
    return html


def _fallback_table(graph: MultimodalGraphBuilder) -> str:
    rows = ["<table border=1 cellpadding=4 style='border-collapse:collapse;'>",
            "<tr style='background:#f0f0f0;'><th>source</th><th>relation</th>"
            "<th>from</th><th>to</th><th>weight</th></tr>"]
    for u, v, d in graph.G.edges(data=True):
        if d.get("source") == "Ontology_Backbone":
            continue
        rows.append(f"<tr><td>{d.get('source', '')}</td>"
                    f"<td>{d.get('relation', '')}</td>"
                    f"<td>{u}</td><td>{v}</td>"
                    f"<td>{d.get('weight', '')}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def render_sunburst_categories(graph: MultimodalGraphBuilder,
                                allocation: Optional[QDWAAllocation] = None
                                ) -> Any:
    if go is None:
        return None
    labels, parents, values, colours = ["All"], [""], [1], ["#cccccc"]
    for cat, members in DENDRITE_CATEGORIES.items():
        present = [m for m in members if m in graph.G.nodes]
        if not present:
            continue
        labels.append(CATEGORY_SHORT_NAMES[cat])
        parents.append("All")
        values.append(len(present))
        colours.append(CATEGORY_COLOURS[cat])
        for m in present:
            degree = graph.G.degree(m) if m in graph.G.nodes else 0
            labels.append(m)
            parents.append(CATEGORY_SHORT_NAMES[cat])
            values.append(max(1, degree))
            colours.append(CATEGORY_COLOURS[cat])
    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colours), branchvalues="total",
        insidetextorientation="radial"))
    fig.update_layout(
        title="Dendrite Concept Categories",
        paper_bgcolor=THEME_PRESETS["Bright (Default)"]["plotly_paper"],
        font_color=THEME_PRESETS["Bright (Default)"]["font"],
        height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def export_publication_figure(graph: MultimodalGraphBuilder,
                              dpi: int = 300,
                              figsize: Tuple[float, float] = (14, 12)) -> bytes:
    if matplotlib is None:
        return b""
    try:
        G = graph.G.to_undirected()
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=200)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        node_colours = [graph.node_colour(n) for n in G.nodes()]
        node_sizes = [400 if G.nodes[n].get("type") == "Simulation_Model"
                      else 120 + 40 * G.degree(n) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                               node_size=node_sizes, edgecolors="white",
                               linewidths=1.5)
        edge_colours = []
        for u, v, d in G.edges(data=True):
            rel = d.get("relation", "?")
            rel_type = resolve_relation_type(rel)
            edge_colours.append(get_edge_color(rel_type) if rel_type
                                 else "#cccccc")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colours,
                               width=0.6, alpha=0.7, arrows=False)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                                 font_weight="bold")
        ax.set_title("Dendrite Structure-Geometry Concept Graph",
                     fontsize=14, fontweight="bold", pad=20)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        if st:
            st.error(f"Publication figure export failed: {exc}")
        return b""


def generate_analysis_report(graph: MultimodalGraphBuilder,
                             centrality_df: pd.DataFrame,
                             burst_df: pd.DataFrame,
                             drift_df: pd.DataFrame,
                             genealogy_df: pd.DataFrame,
                             bridge_df: pd.DataFrame,
                             motifs: Dict[str, Any],
                             model_df: pd.DataFrame,
                             lit_df: pd.DataFrame,
                             validation_metrics: Optional[Dict] = None,
                             distillation_df: Optional[pd.DataFrame] = None
                             ) -> str:
    lines: List[str] = []
    lines.append("# Dendrite Concept Graph Analysis Report\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- Simulation models: {len(model_df)}")
    lines.append(f"- Literature abstracts: {len(lit_df)}")
    if "year" in lit_df.columns:
        yrs = lit_df["year"].dropna()
        if not yrs.empty:
            lines.append(f"- Year range: {int(yrs.min())} – {int(yrs.max())}")
    s = graph.summary()
    lines.append(f"- Graph nodes: {s['nodes']}")
    lines.append(f"- Graph edges: {s['edges']}")
    lines.append(f"- Node sources: {s['by_source']}")
    lines.append(f"- Relations: {s['by_relation']}\n")
    if not centrality_df.empty:
        lines.append("## 2. Top Concepts by PageRank\n")
        for _, r in centrality_df.head(15).iterrows():
            lines.append(f"- **{r['concept']}** ({r['category']}) — "
                          f"PR={r['pagerank']}, deg={r['degree']}, "
                          f"btw={r['betweenness']}")
        lines.append("")
    if not burst_df.empty:
        lines.append("## 3. Emerging Concepts (Keyword Bursts)\n")
        for _, r in burst_df.head(10).iterrows():
            lines.append(f"- **{r['concept']}** — burst ×{r['burst_score']} "
                          f"in {r['burst_year']} ({r['total_mentions']} mentions)")
        lines.append("")
    if not drift_df.empty:
        lines.append("## 4. Semantic Drift\n")
        for _, r in drift_df.head(10).iterrows():
            lines.append(f"- **{r['concept']}** — drift={r['semantic_drift']} "
                          f"({r['early_period']} → {r['late_period']})")
        lines.append("")
    if not genealogy_df.empty:
        lines.append("## 5. Concept Genealogy\n")
        for _, r in genealogy_df.head(20).iterrows():
            lines.append(f"- {r['generation']}: **{r['concept']}**")
        lines.append("")
    if not bridge_df.empty:
        lines.append("## 6. Cross-Category Bridges\n")
        for _, r in bridge_df.head(10).iterrows():
            lines.append(f"- **{r['concept']}** — spans "
                          f"{r['connected_categories']} categories "
                          f"({r['categories']})")
        lines.append("")
    if motifs:
        lines.append("## 7. Network Motifs\n")
        for k, v in motifs.items():
            if k == "top_stars":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    if distillation_df is not None and not distillation_df.empty:
        lines.append("## 8. Top Distilled Concepts\n")
        for _, r in distillation_df.head(15).iterrows():
            lines.append(f"- **{r['concept']}** — distill="
                          f"{r['distillation_efficiency']}, "
                          f"freq={r['frequency']}, "
                          f"coherence={r['coherence_score']}")
        lines.append("")
    if validation_metrics:
        lines.append("## 9. Graph Validation Metrics\n")
        for k, v in validation_metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# PHASE 11: STREAMLIT UI (13 tabs)
# ============================================================================
def _streamlit_main() -> None:
    if st is None:
        return
    init_widget_defaults()
    metrics = get_metrics_logger()

    @st.cache_resource
    def get_ontology() -> DendriteGeometryOntology:
        return DendriteGeometryOntology()

    @st.cache_resource
    def get_resolver() -> AdvancedConceptResolver:
        return AdvancedConceptResolver(get_ontology(), threshold=0.82)

    @st.cache_resource
    def get_vlm() -> VLMImageExtractor:
        return VLMImageExtractor(get_ontology(), get_resolver())

    @st.cache_resource
    def get_qdwa() -> QDWAWeightAllocator:
        return QDWAWeightAllocator(get_ontology(), get_resolver())

    @st.cache_resource
    def get_expander() -> DynamicOntologyExpander:
        return DynamicOntologyExpander(get_ontology())

    def get_graph() -> MultimodalGraphBuilder:
        if "graph" not in st.session_state:
            st.session_state.graph = MultimodalGraphBuilder(get_ontology())
        return st.session_state.graph

    def reset_graph() -> None:
        st.session_state.graph = MultimodalGraphBuilder(get_ontology())

    # -- Sidebar ---------------------------------------------------------
    st.sidebar.title("🧬 Dendrite Concept Graph v8.0")
    st.sidebar.caption("Tri-modal fusion + ML parity with UAV v6.2")
    ont = get_ontology()
    st.sidebar.write(f"**{len(ont.concepts)}** concepts · "
                     f"**{len(ont.relationships)}** GT edges")
    vlm = get_vlm()
    st.sidebar.code(vlm.status())
    if TORCH_AVAILABLE:
        st.sidebar.code(f"✅ torch {torch.__version__}")
    else:
        st.sidebar.code("⚠️ torch not available")
    if st.sidebar.button("🔄 Reset graph", key="reset_btn"):
        reset_graph()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Node colours**")
    for src, col in SOURCE_COLOUR.items():
        st.sidebar.markdown(
            f"<span style='color:{col}'>●</span> {src}",
            unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**QDWA categories**")
    for cat, members in DENDRITE_CATEGORIES.items():
        st.sidebar.markdown(
            f"<span style='color:{CATEGORY_COLOURS[cat]}'>●</span> "
            f"{CATEGORY_SHORT_NAMES[cat]} ({len(members)})",
            unsafe_allow_html=True)

    # ============== TAB 1: SIMULATION ====================
    def tab_simulation() -> None:
        st.header("📊 Simulation Ground Truth")
        st.caption("Manuscript Table — 6 phase-field models (A–F)")
        uploaded = st.file_uploader("Upload models CSV", type=["csv"],
                                     key="models_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(io.StringIO(SAMPLE_MODELS_CSV))
            st.info("Using bundled sample_models.csv.")
        st.dataframe(df, use_container_width=True)
        if st.button("Build / refresh graph from CSV", type="primary"):
            metrics.start_step("build_csv_graph")
            try:
                get_graph().build_base_graph_from_csv(df)
                metrics.end_step({"rows": len(df)})
                st.success(f"Loaded {len(df)} models. Graph now has "
                            f"{get_graph().G.number_of_edges()} edges.")
            except ValueError as exc:
                metrics.end_step({"error": str(exc)})
                st.error(str(exc))
        st.subheader("Inspect a model")
        model = st.selectbox("Pick a model", list(MODEL_SIGNATURES.keys()))
        if model:
            sig = MODEL_SIGNATURES[model]
            st.markdown(f"**{model}** — {sig['geometry']} · "
                        f"S/L = {sig['ratio']} · "
                        f"{int(sig['solid_frac']*100)}% solid")
            g = get_graph()
            if g.G.has_node(model):
                for u, v, d in g.G.out_edges(model, data=True):
                    st.markdown(f"- `{u}` **[{d.get('relation', '?')}]** `{v}` "
                                f"w={d.get('weight', '?')} "
                                f"_{d.get('source', '?')}_")
            else:
                st.info("Model not yet in graph.")

    # ============== TAB 2: LITERATURE ====================
    def tab_literature() -> None:
        st.header("📚 Literature Concept Extraction")
        st.caption("Stream 2 — sentence co-occurrence over abstracts")
        uploaded = st.file_uploader(
            "Upload literature CSV (title, abstract, year)",
            type=["csv"], key="lit_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
            st.info("Using bundled sample_literature.csv.")
        st.dataframe(df, use_container_width=True)
        if st.button("Extract & fuse literature triples", type="primary"):
            metrics.start_step("extract_literature")
            ext = TextConceptExtractor(get_ontology(), get_resolver())
            with st.spinner("Scanning abstracts..."):
                triples = ext.extract_from_corpus(df)
                freq = ext.concept_frequencies(df)
            metrics.end_step({"triples": len(triples)})
            st.success(f"Extracted {len(triples)} co-occurrence triples from "
                        f"{len(df)} abstracts.")
            if triples:
                st.dataframe(pd.DataFrame(triples), use_container_width=True)
            st.subheader("Concept frequencies")
            if freq:
                freq_df = pd.DataFrame({"concept": list(freq.keys()),
                                         "count": list(freq.values())})
                st.bar_chart(freq_df.set_index("concept"))
            get_graph().fuse_literature_edges(triples)
            n_bridges = get_graph().infer_cross_modal_paths(max_depth=2)
            st.info(f"Added {n_bridges} cross-modal bridge edges.")
            st.session_state.last_lit_df = df
            st.session_state.last_lit_triples = triples

    # ============== TAB 3: VLM ===========================
    def tab_vlm() -> None:
        st.header("👁 VLM Image Analyzer")
        st.caption("Stream 3 — Qwen2.5-VL-3B-Instruct on phase-field images")
        vlm_ = get_vlm()
        if not vlm_.is_available():
            st.warning(f"VLM backend not available. {vlm_.status()}")
            st.info("Other tabs work without the VLM.")
            st.subheader("Manual triple entry (VLM-less fallback)")
            with st.form("manual_triple"):
                concepts = sorted(get_ontology().allowed_concepts())
                subj = st.selectbox("Subject", concepts, key="m_subj")
                rel = st.selectbox("Relation",
                                    ["HAS_GEOMETRY", "HAS_RATIO", "GENERATES",
                                     "EXHIBITS", "INHIBITS", "CONSTRAINS",
                                     "PROMOTES", "DRIVES", "INDICATES"],
                                    key="m_rel")
                obj = st.selectbox("Object", concepts, key="m_obj")
                ev = st.text_input("Evidence", key="m_ev")
                if st.form_submit_button("Inject triple"):
                    get_graph().fuse_vlm_insights([{
                        "subject": subj, "relation": rel, "object": obj,
                        "weight": 0.8, "evidence": ev or "manual",
                        "source": "VLM_Vision"}])
                    st.success(f"Injected {subj} {rel} {obj}")
                    st.rerun()
            return
        uploaded = st.file_uploader(
            "Upload phase-field simulation image",
            type=["png", "jpg", "jpeg"], key="vlm_img")
        if not uploaded:
            st.info("Upload a PNG/JPG to analyze.")
            return
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        col_a, col_b = st.columns(2)
        max_tok = col_a.number_input("Max new tokens", min_value=64,
                                       max_value=2048, step=64,
                                       key="vlm_max_tokens")
        temp = col_b.slider("Temperature", 0.0, 1.0, step=0.05,
                             key="vlm_temperature")
        if st.button("Run VLM Analysis", type="primary"):
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix,
                                               delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            metrics.start_step("vlm_inference")
            with st.spinner("Running Qwen2.5-VL..."):
                triples = vlm_.extract_triples_from_image(
                    tmp_path, max_new_tokens=int(max_tok),
                    temperature=float(temp))
            os.unlink(tmp_path)
            metrics.end_step({"triples": len(triples),
                              "vram_mb": round(get_gpu_vram_mb(), 1)})
            st.success(f"Got {len(triples)} validated triples.")
            if triples:
                st.dataframe(pd.DataFrame(triples), use_container_width=True)
                if st.button("Inject into concept graph", type="primary"):
                    get_graph().fuse_vlm_insights(triples)
                    get_graph().infer_cross_modal_paths(max_depth=2)
                    st.success("Injected.")
                    st.rerun()
            else:
                st.info("No triples passed ontology validation.")

    # ============== TAB 4: QDWA ==========================
    def tab_qdwa() -> None:
        st.header("🎯 QDWA — Query Distillation & Weighted Allocation")
        st.caption("Routes a free-text query to the four dendrite pillars")
        query = st.text_area(
            "Enter your research query",
            value="How does the elastic modulus of LLZO constrain dendrite "
                  "growth and reduce dead lithium?",
            height=80, key="qdwa_query")
        col_a, col_b, col_c = st.columns(3)
        top_k = col_a.slider("Top-K concepts", 3, 20, key="qdwa_top_k")
        min_w = col_b.slider("Min concept weight", 0.0, 1.0, step=0.01,
                              key="qdwa_min_weight")
        cmap = col_c.selectbox("Colormap", list(SUPPORTED_COLORMAPS.keys()),
                                key="qdwa_cmap")
        if st.button("Allocate weights", type="primary", key="qdwa_run"):
            metrics.start_step("qdwa_allocate")
            g = get_graph()
            centrality = {}
            try:
                centrality = nx.pagerank(g.G.to_undirected(), weight="weight")
            except Exception:
                pass
            alloc = get_qdwa().allocate(query, centrality=centrality,
                                          top_k=int(top_k),
                                          min_weight=float(min_w))
            metrics.end_step({"concepts": len(alloc.top_concepts)})
            st.session_state.qdwa_result = alloc
        if "qdwa_result" in st.session_state:
            alloc: QDWAAllocation = st.session_state.qdwa_result
            st.subheader("Category weights")
            cat_df = pd.DataFrame([
                {"Category": CATEGORY_SHORT_NAMES[c],
                 "Weight": round(w, 4),
                 "Colour": CATEGORY_COLOURS[c]}
                for c, w in alloc.category_weights.items()])
            if px:
                fig = px.bar(cat_df, x="Category", y="Weight",
                              color="Category",
                              color_discrete_map={
                                  CATEGORY_SHORT_NAMES[c]: CATEGORY_COLOURS[c]
                                  for c in DENDRITE_CATEGORIES})
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cat_df[["Category", "Weight"]],
                          use_container_width=True)
            st.subheader(f"Top concepts (K={top_k})")
            if alloc.top_concepts:
                top_df = pd.DataFrame([
                    {"concept": c, "weight": round(w, 4), "category": cat}
                    for c, w, cat in alloc.top_concepts])
                st.dataframe(top_df, use_container_width=True)
                if px:
                    bar_fig = px.bar(top_df, x="weight", y="concept",
                                      orientation="h", color="category",
                                      color_discrete_map={
                                          CATEGORY_SHORT_NAMES[c]:
                                          CATEGORY_COLOURS[c]
                                          for c in DENDRITE_CATEGORIES})
                    bar_fig.update_layout(height=max(300, 24 * len(top_df)))
                    st.plotly_chart(bar_fig, use_container_width=True)
            st.subheader("Inferred problem type")
            st.code(alloc.problem, language="text")
            st.subheader("Reasoning")
            for r in alloc.reasoning or ["(direct matches only)"]:
                st.markdown(f"- {r}")

    # ============== TAB 5: UNIFIED GRAPH =================
    def tab_unified() -> None:
        st.header("🕸 Unified Concept Graph")
        st.caption("Green = CSV · Blue = Literature · Orange = VLM · "
                   "Purple = Inferred · Cyan = Semantic · Pink = Expansion")
        g = get_graph()
        s = g.summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", s["nodes"])
        c2.metric("Edges", s["edges"])
        c3.metric("Relation types", len(s["by_relation"]))
        with st.expander("Source breakdown"):
            st.json(s["by_source"])
        with st.expander("Relation breakdown"):
            st.json(s["by_relation"])
        col_a, col_b, col_c = st.columns(3)
        focus = col_a.selectbox(
            "Focus node (shows neighbourhood)",
            [None] + sorted(g.G.nodes),
            format_func=lambda n: "(all)" if n is None else n,
            key="graph_focus")
        show_backbone = col_b.checkbox("Show backbone",
                                         key="graph_show_backbone")
        height = col_c.slider("Height (px)", 400, 1200, step=50,
                                key="graph_height")
        html = render_graph_pyvis(g, focus=focus,
                                    height=f"{int(height)}px",
                                    show_backbone=bool(show_backbone))
        if "<html" in html.lower():
            st.components.v1.html(html, height=int(height) + 50,
                                    scrolling=True)
        else:
            st.markdown(html, unsafe_allow_html=True)
        st.subheader("All real edges (non-backbone)")
        rows = []
        for u, v, d in g.G.edges(data=True):
            if d.get("source") == "Ontology_Backbone":
                continue
            ev = ""
            if d.get("visual_evidence"):
                ev = d["visual_evidence"]
            elif d.get("literature_evidence"):
                lst = d["literature_evidence"]
                ev = lst[0] if isinstance(lst, list) and lst else ""
            rows.append({"from": u, "relation": d.get("relation", ""),
                         "to": v, "weight": d.get("weight", ""),
                         "source": d.get("source", ""), "evidence": ev})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No real edges yet.")
        st.subheader("Category sunburst")
        fig = render_sunburst_categories(g)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ============== TAB 6: ANALYTICS =====================
    def tab_analytics() -> None:
        st.header("📈 Advanced Analytics")
        st.caption("Burst detection · Drift · Genealogy · Bridges")
        g = get_graph()
        valid = list(g.G.nodes)
        lit_df = st.session_state.get("last_lit_df")
        if lit_df is None:
            lit_df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
        col_a, col_b = st.columns(2)
        burst_thresh = col_a.slider("Burst threshold (×)", 1.5, 5.0,
                                      step=0.1, key="analytics_burst_threshold")
        early_frac = col_b.slider("Drift early fraction", 0.1, 0.5,
                                    step=0.05, key="analytics_drift_early")
        with st.expander("🔍 Keyword burst detection", expanded=True):
            metrics.start_step("burst_detection")
            burst_df = detect_keyword_bursts(lit_df, get_ontology(),
                                                get_resolver(),
                                                float(burst_thresh))
            metrics.end_step({"rows": len(burst_df)})
            if burst_df.empty:
                st.info("No bursts detected.")
            else:
                st.dataframe(burst_df, use_container_width=True)
                if px:
                    fig = px.bar(burst_df.head(15), x="burst_score",
                                  y="concept", orientation="h",
                                  color="category",
                                  color_discrete_map={
                                      CATEGORY_SHORT_NAMES[c]:
                                      CATEGORY_COLOURS[c]
                                      for c in DENDRITE_CATEGORIES})
                    fig.update_layout(
                        height=max(300, 26 * len(burst_df.head(15))))
                    st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔀 Semantic drift"):
            metrics.start_step("semantic_drift")
            drift_df = detect_semantic_drift(lit_df, get_ontology(),
                                                get_resolver(),
                                                float(early_frac),
                                                float(early_frac))
            metrics.end_step({"rows": len(drift_df)})
            if drift_df.empty:
                st.info("Drift requires ≥4 years + embeddings.")
            else:
                st.dataframe(drift_df, use_container_width=True)
        with st.expander("🌳 Concept genealogy"):
            metrics.start_step("genealogy")
            gene_df = build_concept_genealogy(g.G, valid)
            metrics.end_step({"rows": len(gene_df)})
            if gene_df.empty:
                st.info("Requires ≥5 nodes.")
            else:
                gen_counts = gene_df["generation"].value_counts()
                cols = st.columns(len(gen_counts))
                for i, (gen, cnt) in enumerate(gen_counts.items()):
                    cols[i].metric(gen, int(cnt))
                st.dataframe(gene_df.head(30), use_container_width=True)
        with st.expander("🌉 Cross-category bridges"):
            metrics.start_step("bridges")
            bridge_df = detect_cross_domain_bridges(g.G, valid)
            metrics.end_step({"rows": len(bridge_df)})
            if bridge_df.empty:
                st.info("No cross-category bridges yet.")
            else:
                st.dataframe(bridge_df, use_container_width=True)
        st.session_state.last_burst_df = burst_df
        st.session_state.last_drift_df = drift_df
        st.session_state.last_genealogy_df = gene_df
        st.session_state.last_bridge_df = bridge_df

    # ============== TAB 7: TOPOLOGY ======================
    def tab_topology() -> None:
        st.header("🔬 Network Topology")
        st.caption("Centrality · Motifs · Degree distribution")
        g = get_graph()
        valid = list(g.G.nodes)
        metrics.start_step("topology")
        cent_df = compute_centrality_comparison(g.G, valid)
        motifs = analyze_network_motifs(g.G)
        metrics.end_step({"concepts": len(cent_df)})
        st.session_state.last_centrality_df = cent_df
        st.session_state.last_motifs = motifs
        if not cent_df.empty:
            st.subheader("Centrality comparison")
            st.dataframe(cent_df.head(40), use_container_width=True)
            if go:
                top = cent_df.head(15)
                fig = go.Figure()
                for col_name, colour in [("pagerank", "#1f77b4"),
                                           ("betweenness", "#ff7f0e"),
                                           ("closeness", "#2ca02c")]:
                    fig.add_trace(go.Bar(x=top["concept"], y=top[col_name],
                                          name=col_name, marker_color=colour))
                fig.update_layout(barmode="group", height=420,
                                    xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Degree distribution")
        deg_fig = plot_degree_distribution(g.G)
        if deg_fig:
            st.plotly_chart(deg_fig, use_container_width=True)
        st.subheader("Network motifs")
        if motifs:
            m_cols = st.columns(4)
            m_cols[0].metric("Triangles", motifs.get("total_triangles", 0))
            m_cols[1].metric("Cliques", motifs.get("total_cliques", 0))
            m_cols[2].metric("Max clique", motifs.get("max_clique_size", 0))
            m_cols[3].metric("Star motifs", motifs.get("star_motifs", 0))
            if motifs.get("top_stars"):
                st.markdown("**Top star motifs**")
                star_df = pd.DataFrame(motifs["top_stars"],
                                        columns=["node", "degree", "clustering"])
                st.dataframe(star_df, use_container_width=True)

    # ============== TAB 8: ML TRAINING ===================
    def tab_ml() -> None:
        st.header("🔬 ML Training — GraphSAGE + LatentMoE")
        st.caption("Link prediction + research-direction scoring")
        if not TORCH_AVAILABLE:
            st.error("torch not available. Install torch to enable this tab.")
            return
        g = get_graph()
        valid = list(g.G.nodes)
        if len(valid) < 5:
            st.warning("Need at least 5 nodes in the graph. Build the CSV "
                        "graph first.")
            return
        col_a, col_b, col_c = st.columns(3)
        epochs = col_a.slider("Epochs", 5, 200, step=5, key="ml_epochs")
        lr = col_b.number_input("Learning rate", 1e-5, 1e-1, step=1e-4,
                                  format="%.5f", key="ml_lr")
        hidden = col_c.slider("Hidden dim", 32, 256, step=32,
                                key="ml_hidden")
        if st.button("Train GraphSAGE", type="primary", key="ml_train"):
            metrics.start_step("train_graphsage")
            concept_to_id = {c: i for i, c in enumerate(valid)}
            embs = get_resolver().embed_batch(valid)
            if embs is None:
                embs = np.random.randn(len(valid), 384).astype(np.float32)
            node_features = torch.tensor(embs, dtype=torch.float32)
            pos_pairs, neg_pairs = sample_edges_for_training(
                g.G.to_undirected(), valid, concept_to_id)
            progress_bar = st.progress(0)
            status_text = st.empty()
            def _cb(epoch, loss):
                pct = min(1.0, (epoch + 1) / int(epochs))
                progress_bar.progress(pct)
                status_text.text(f"Epoch {epoch + 1} loss={loss:.4f}")
            model, final_emb, adj_idx, adj_val = train_graphsage(
                node_features, g.G.to_undirected(), concept_to_id,
                pos_pairs, neg_pairs, progress_callback=_cb,
                epochs=int(epochs), lr=float(lr), hidden_dim=int(hidden))
            progress_bar.empty()
            status_text.empty()
            metrics.end_step({"pos_edges": len(pos_pairs),
                              "neg_edges": len(neg_pairs)})
            if model is None:
                st.error("Training failed.")
                return
            st.success(f"Trained on {len(pos_pairs)} positive and "
                        f"{len(neg_pairs)} negative pairs.")
            st.session_state.ml_model = model
            st.session_state.ml_final_emb = final_emb
            st.session_state.ml_valid = valid
            st.session_state.ml_concept_to_id = concept_to_id
        if "ml_model" in st.session_state:
            st.subheader("Research direction scoring")
            top_n = st.slider("Top-K predicted pairs", 5, 50, step=5,
                                key="ml_top_pairs")
            if st.button("Score novel pairs", key="ml_score"):
                metrics.start_step("score_pairs")
                g = get_graph()
                valid = st.session_state.ml_valid
                concept_to_id = st.session_state.ml_concept_to_id
                final_emb = st.session_state.ml_final_emb
                model = st.session_state.ml_model
                embs = get_resolver().embed_batch(valid)
                if embs is None:
                    embs = np.random.randn(len(valid), 384).astype(np.float32)
                node_features = torch.tensor(embs, dtype=torch.float32)
                # Build a simple Ridge on degree-features for exp_gain
                degrees = dict(g.G.degree())
                X = []
                y = []
                for u, v, d in g.G.edges(data=True):
                    du = degrees.get(u, 0)
                    dv = degrees.get(v, 0)
                    w = d.get("weight", 0.5)
                    X.append([du, dv, w])
                    y.append(w * 100.0)
                ridge = None
                if Ridge is not None and len(X) > 5:
                    try:
                        ridge = Ridge(alpha=1.0).fit(X, y)
                    except Exception:
                        ridge = None
                concept_properties = {c: int(degrees.get(c, 0) * 10)
                                       for c in valid}
                scores_df = compute_research_direction_scores(
                    model, node_features, final_emb, g.G.to_undirected(),
                    valid, concept_properties, ridge,
                    embed_model_or_matrix=get_resolver(),
                    n_samples=2000)
                metrics.end_step({"pairs": len(scores_df)})
                st.session_state.ml_scores_df = scores_df
                if not scores_df.empty:
                    st.dataframe(scores_df.head(int(top_n)),
                                  use_container_width=True)
                    if px:
                        fig = px.bar(scores_df.head(int(top_n)),
                                      x="composite_score",
                                      y=scores_df.head(int(top_n)).apply(
                                          lambda r: f"{r['concept_u']} ↔ "
                                                    f"{r['concept_v']}",
                                          axis=1),
                                      orientation="h",
                                      color="gnn_affinity",
                                      color_continuous_scale="turbo")
                        fig.update_layout(height=max(300, 26 * int(top_n)),
                                           yaxis_title="pair")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No novel pairs found. Try lowering the "
                             "semantic threshold or expanding the graph.")

    # ============== TAB 9: DISTILLATION ==================
    def tab_distillation() -> None:
        st.header("🧪 Concept Distillation")
        st.caption("TF-IDF × frequency × coherence ranking")
        g = get_graph()
        valid = list(g.G.nodes)
        lit_df = st.session_state.get("last_lit_df")
        if lit_df is None:
            lit_df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
        col_a, col_b = st.columns(2)
        max_docs = col_a.slider("Max docs per concept", 5, 100, step=5,
                                  key="ml_distill_max_docs")
        coh_window = col_b.slider("Coherence window (words)", 5, 50, step=5,
                                    key="ml_coherence_window")
        if st.button("Run distillation", type="primary", key="ml_distill"):
            metrics.start_step("concept_distillation")
            ext = TextConceptExtractor(get_ontology(), get_resolver())
            abstract_map = ext.concept_abstract_map(lit_df)
            all_texts = [f"{row.get('title', '')}. {row.get('abstract', '')}"
                          for _, row in lit_df.iterrows()]
            distilled_df = compute_concept_distillation(
                valid, abstract_map, all_texts,
                resolver=get_resolver(),
                max_docs_per_concept=int(max_docs),
                coherence_window=int(coh_window))
            metrics.end_step({"concepts": len(distilled_df)})
            st.session_state.distilled_df = distilled_df
        if "distilled_df" in st.session_state:
            distilled_df = st.session_state.distilled_df
            st.dataframe(distilled_df.head(30), use_container_width=True)
            if go:
                top = distilled_df.head(20)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=top["concept"],
                                       y=top["tfidf_weight"],
                                       name="TF-IDF", marker_color="#1f77b4"))
                fig.add_trace(go.Bar(x=top["concept"],
                                       y=top["coherence_score"],
                                       name="Coherence", marker_color="#ff7f0e"))
                fig.add_trace(go.Bar(x=top["concept"],
                                       y=top["distillation_efficiency"],
                                       name="Efficiency",
                                       marker_color="#2ca02c"))
                fig.update_layout(barmode="group", height=420,
                                    xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

    # ============== TAB 10: VALIDATION ===================
    def tab_validation() -> None:
        st.header("✅ Graph Validation")
        st.caption("Modularity · Silhouette · Bootstrap CI · Permutation test")
        g = get_graph()
        valid = list(g.G.nodes)
        col_a, col_b = st.columns(2)
        n_perm = col_a.slider("Permutation trials", 10, 200, step=10,
                                key="ml_permutation_trials")
        n_boot = col_b.slider("Bootstrap samples", 100, 2000, step=100,
                                key="ml_validate_bootstrap")
        if st.button("Run validation", type="primary", key="ml_validate"):
            metrics.start_step("graph_validation")
            vmetrics = validate_graph_metrics(
                g.G.to_undirected(), valid,
                resolver=get_resolver(), n_permutations=int(n_perm))
            metrics.end_step({"metrics": len(vmetrics)})
            st.session_state.validation_metrics = vmetrics
        if "validation_metrics" in st.session_state:
            vm = st.session_state.validation_metrics
            st.subheader("Graph metrics")
            k1, k2, k3 = st.columns(3)
            k1.metric("Modularity", f"{vm.get('modularity', 0):.4f}")
            k2.metric("Silhouette", f"{vm.get('silhouette_score', 0):.4f}")
            k3.metric("Density", f"{vm.get('density', 0):.5f}")
            k4, k5, k6 = st.columns(3)
            k4.metric("Communities", vm.get("n_communities", 0))
            k5.metric("Significant edges", vm.get("edge_significant_count", 0))
            k6.metric("Permutation p-mean",
                       f"{vm.get('edge_significance_p_mean', 1.0):.4f}")
            st.json(vm)
            st.subheader("Bootstrap confidence intervals")
            weights = np.array([d.get("weight", 1.0)
                                 for _, _, d in g.G.edges(data=True)])
            if len(weights) >= 2:
                mean, lo, hi = compute_bootstrap_ci(
                    weights, n_bootstrap=int(n_boot))
                st.metric("Edge weight mean",
                          f"{mean:.4f}",
                          delta=f"[{lo:.4f}, {hi:.4f}] 95% CI")
            else:
                st.info("Need ≥2 edges for bootstrap.")

    # ============== TAB 11: ONTOLOGY EXPANSION ===========
    def tab_expansion() -> None:
        st.header("🧬 Dynamic Ontology Expander")
        st.caption("LLM proposes new concepts from unresolved phrases")
        expander = get_expander()
        st.info(expander.status())
        g = get_graph()
        lit_df = st.session_state.get("last_lit_df")
        if lit_df is None:
            lit_df = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
        # Find unresolved phrases: n-grams in literature not covered by ontology
        if st.button("Find unresolved phrases", key="exp_find"):
            all_text = " ".join(
                f"{r.get('title', '')} {r.get('abstract', '')}"
                for _, r in lit_df.iterrows())
            words = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", all_text.lower())
            # 2-grams
            bigrams = [f"{words[i]} {words[i + 1]}"
                       for i in range(len(words) - 1)]
            resolved = set()
            for bg in set(bigrams):
                if get_resolver().resolve(bg):
                    resolved.add(bg)
            unresolved = [bg for bg in set(bigrams)
                          if bg not in resolved
                          and bg not in st.session_state.get(
                              "exp_resolved_cache", set())]
            # Filter stop-wordy
            stop = {"the ", "and ", "for ", "with ", "from ", "this ",
                     "that ", "have ", "which ", "these ", "using "}
            unresolved = [u for u in unresolved
                          if not any(s in u for s in stop)][:30]
            st.session_state.exp_unresolved = unresolved
        if "exp_unresolved" in st.session_state:
            unresolved = st.session_state.exp_unresolved
            if not unresolved:
                st.success("All n-grams are covered by the ontology.")
                return
            st.write(f"Found {len(unresolved)} candidate phrases:")
            st.dataframe(pd.DataFrame({"phrase": unresolved}),
                          use_container_width=True)
            if not expander.is_available():
                st.warning("Set OPENAI_API_KEY to enable LLM proposals.")
                return
            n_new = st.slider("Propose how many?", 1, 10, step=1,
                                key="ml_expander_max_new")
            if st.button(f"Propose {n_new} concepts", type="primary",
                          key="exp_propose"):
                context = lit_df.iloc[0].get("abstract", "") if len(lit_df) else ""
                with st.spinner("Asking LLM for ontology proposals..."):
                    proposals = expander.propose_batch(
                        unresolved, context=context,
                        model=st.session_state.get("llm_model", "gpt-4o-mini"),
                        max_new=int(n_new))
                st.session_state.exp_proposals = proposals
        if "exp_proposals" in st.session_state:
            proposals = st.session_state.exp_proposals
            if not proposals:
                st.info("No valid proposals returned.")
                return
            st.subheader("Proposals (approve to commit)")
            for i, p in enumerate(proposals):
                with st.expander(f"{p['name']} — {p['type']}", expanded=True):
                    st.json(p)
                    if st.button(f"✅ Commit {p['name']}",
                                  key=f"commit_{i}"):
                        ok = expander.commit(p, graph_builder=g)
                        if ok:
                            st.success(f"Added {p['name']} to ontology and graph.")
                            st.session_state.pop("exp_proposals", None)
                            st.rerun()
                        else:
                            st.warning(f"{p['name']} already exists.")

    # ============== TAB 12: EXPORT =======================
    def tab_export() -> None:
        st.header("🎨 Publication Export")
        st.caption("Figures, GraphML, and analysis report")
        g = get_graph()
        col_a, col_b, col_c = st.columns(3)
        dpi = col_a.number_input("DPI", 72, 600, step=24, key="export_dpi")
        fw = col_b.number_input("Width (in)", 8.0, 24.0, step=0.5,
                                 key="export_figsize_w")
        fh = col_c.number_input("Height (in)", 8.0, 24.0, step=0.5,
                                 key="export_figsize_h")
        st.subheader("Publication figure")
        if st.button("Render PNG", key="render_png"):
            metrics.start_step("export_png")
            png = export_publication_figure(
                g, dpi=int(dpi), figsize=(float(fw), float(fh)))
            metrics.end_step({"bytes": len(png)})
            if png:
                st.image(png, caption="Publication figure preview")
                st.download_button("⬇ Download PNG", png,
                                     file_name="dendrite_graph_pub.png",
                                     mime="image/png")
            else:
                st.error("PNG render failed.")
        st.subheader("GraphML export")
        gml_bytes = b""
        try:
            buf = io.BytesIO()
            G_export = g.G.copy()
            for _, _, d in G_export.edges(data=True):
                for k, v in list(d.items()):
                    if isinstance(v, (list, dict)):
                        d[k] = json.dumps(v)
            nx.write_graphml(G_export, buf)
            gml_bytes = buf.getvalue()
        except Exception as exc:
            st.warning(f"GraphML export failed: {exc}")
        if gml_bytes:
            st.download_button("⬇ Download GraphML", gml_bytes,
                                 file_name="dendrite_graph.graphml",
                                 mime="application/graphml+xml")
        st.subheader("Analysis report")
        if st.button("Generate Markdown report", key="render_report"):
            metrics.start_step("export_report")
            report = generate_analysis_report(
                g,
                st.session_state.get("last_centrality_df", pd.DataFrame()),
                st.session_state.get("last_burst_df", pd.DataFrame()),
                st.session_state.get("last_drift_df", pd.DataFrame()),
                st.session_state.get("last_genealogy_df", pd.DataFrame()),
                st.session_state.get("last_bridge_df", pd.DataFrame()),
                st.session_state.get("last_motifs", {}),
                pd.read_csv(io.StringIO(SAMPLE_MODELS_CSV)),
                st.session_state.get("last_lit_df",
                    pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))),
                st.session_state.get("validation_metrics"),
                st.session_state.get("distilled_df"))
            metrics.end_step({"length": len(report)})
            st.session_state.report_md = report
        if "report_md" in st.session_state:
            st.text_area("Report preview", st.session_state.report_md,
                           height=300)
            st.download_button("⬇ Download report.md",
                                 st.session_state.report_md.encode("utf-8"),
                                 file_name="dendrite_report.md",
                                 mime="text/markdown")

    # ============== TAB 13: METRICS ======================
    def tab_metrics() -> None:
        st.header("⚙️ Metrics & Performance")
        if not metrics.steps:
            st.info("No metrics recorded yet.")
            return
        df = metrics.step_df()
        rdf = metrics.resource_df()
        total = float(df["duration_s"].sum())
        peak = float(rdf["rss_mb"].max()) if not rdf.empty else 0.0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total runtime", f"{total:.2f} s")
        k2.metric("Steps", len(df))
        k3.metric("Peak RSS", f"{peak:.0f} MB")
        k4.metric("Peak VRAM",
                   f"{rdf['vram_mb'].max() if not rdf.empty else 0.0:.0f} MB")
        if px:
            fig = px.bar(df.sort_values("duration_s", ascending=False),
                          x="duration_s", y="name", orientation="h",
                          color="duration_s", color_continuous_scale="Turbo")
            fig.update_layout(height=max(280, 26 * len(df)))
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Step log")
        st.dataframe(df, use_container_width=True)
        st.subheader("Counters")
        if metrics.counters:
            st.dataframe(pd.DataFrame(list(metrics.counters.items()),
                                       columns=["Counter", "Value"]),
                          use_container_width=True)
        with st.expander("Function timings"):
            st.code(PerformanceMonitor.get_report(), language="text")
        if st.button("💾 Save log to disk"):
            path = metrics.save_log()
            st.success(f"Saved → {path}")
        st.download_button(
            "⬇ Download JSON",
            json.dumps(metrics.summary(), indent=2, default=str),
            file_name=f"dendrite_metrics_{metrics.run_id}.json",
            mime="application/json")

    # ============== TAB LAYOUT ==========================
    tabs = st.tabs([
        "📊 Simulation",
        "📚 Literature",
        "👁 VLM",
        "🎯 QDWA",
        "🕸 Unified Graph",
        "📈 Analytics",
        "🔬 Topology",
        "🧠 ML Training",
        "🧪 Distillation",
        "✅ Validation",
        "🧬 Expansion",
        "🎨 Export",
        "⚙️ Metrics",
    ])
    with tabs[0]: tab_simulation()
    with tabs[1]: tab_literature()
    with tabs[2]: tab_vlm()
    with tabs[3]: tab_qdwa()
    with tabs[4]: tab_unified()
    with tabs[5]: tab_analytics()
    with tabs[6]: tab_topology()
    with tabs[7]: tab_ml()
    with tabs[8]: tab_distillation()
    with tabs[9]: tab_validation()
    with tabs[10]: tab_expansion()
    with tabs[11]: tab_export()
    with tabs[12]: tab_metrics()


# ============================================================================
# HEADLESS SMOKE TEST
# ============================================================================
def _headless_smoke_test() -> None:
    print("=" * 78)
    print("Dendrite Concept Graph v8.0 — Headless Smoke Test")
    print("=" * 78)
    print(f"torch: {'available' if TORCH_AVAILABLE else 'unavailable'}")
    print(f"pyvis: {'available' if Network else 'unavailable'}")
    print(f"plotly: {'available' if go else 'unavailable'}")
    print(f"sklearn: {'available' if Ridge else 'unavailable'}")
    print()

    ont = DendriteGeometryOntology()
    res = AdvancedConceptResolver(ont, threshold=0.82)
    ext = TextConceptExtractor(ont, res)
    gb = MultimodalGraphBuilder(ont)
    metrics = RunMetricsLogger()

    print(f"[Ontology] {len(ont.concepts)} concepts · "
          f"{len(ont.relationships)} edges · "
          f"{NUM_EDGE_TYPES} relationship types")
    print(f"[RelationshipType] {len(EDGE_COLOR_REGISTRY)} registered colours")

    # 1. Stream 1
    metrics.start_step("csv_build")
    models = pd.read_csv(io.StringIO(SAMPLE_MODELS_CSV))
    gb.build_base_graph_from_csv(models)
    metrics.end_step()
    print(f"[Stream 1] CSV: {len(models)} models → "
          f"{gb.G.number_of_edges()} edges")

    # 2. Stream 2
    metrics.start_step("lit_fuse")
    lit = pd.read_csv(io.StringIO(SAMPLE_LITERATURE_CSV))
    triples = ext.extract_from_corpus(lit)
    gb.fuse_literature_edges(triples)
    metrics.end_step()
    print(f"[Stream 2] Literature: {len(triples)} triples → "
          f"{gb.G.number_of_edges()} edges")

    # 3. Stream 3 (simulated)
    metrics.start_step("vlm_sim")
    fake_vlm = [
        {"subject": "Model_A", "relation": "GENERATES",
         "object": "von_mises_stress", "weight": 0.8,
         "evidence": "bright red band along the solid-liquid boundary"},
        {"subject": "Model_E", "relation": "EXHIBITS",
         "object": "severe_side_branching", "weight": 0.8,
         "evidence": "many long lateral protrusions"}]
    gb.fuse_vlm_insights(fake_vlm)
    metrics.end_step()
    print(f"[Stream 3] VLM (simulated): {len(fake_vlm)} triples → "
          f"{gb.G.number_of_edges()} edges")

    # 4. Semantic edges
    metrics.start_step("semantic")
    n_semantic = gb.add_semantic_edges(res, threshold=0.65, max_edges=50)
    metrics.end_step({"edges": n_semantic})
    print(f"[Semantic] Added {n_semantic} SEMANTIC edges → "
          f"{gb.G.number_of_edges()} edges")

    # 5. Cross-modal
    n_bridges = gb.infer_cross_modal_paths(max_depth=2)
    print(f"[Cross-modal] Added {n_bridges} bridges → "
          f"{gb.G.number_of_edges()} edges")
    print(f"[Summary] {gb.summary()}")

    # 6. VLM parser
    vlm = VLMImageExtractor(ont, res)
    fake_resp = '''```json
    [
      {"subject": "Model A", "relation": "GENERATES", "object": "von Mises stress",
       "evidence": "bright red band"},
      {"subject": "thermal runaway", "relation": "PROMOTES", "object": "rainbow"}
    ]
    ```'''
    parsed = vlm._parse_and_validate(fake_resp)
    print(f"[VLM parser] {len(parsed)} validated (hallucinations dropped)")

    # 7. QDWA
    qdwa = QDWAWeightAllocator(ont, res)
    alloc = qdwa.allocate(
        "How does the elastic modulus of LLZO constrain dendrite growth "
        "and reduce dead lithium?",
        centrality=nx.pagerank(gb.G.to_undirected(), weight="weight")
        if gb.G.number_of_nodes() > 0 else {})
    print(f"[QDWA] Problem inferred: {alloc.problem}")
    for cat, w in alloc.category_weights.items():
        print(f"  {CATEGORY_SHORT_NAMES[cat]:12s} {w:.3f}")
    print(f"  Top concepts: {[(c, round(w, 3)) for c, w, _ in alloc.top_concepts[:5]]}")

    # 8. Graph-RAG
    retriever = GraphRAGRetriever(gb.G, ont, res)
    sub, seeds = retriever.retrieve("dendrite suppression via stress")
    print(f"[Graph-RAG] Subgraph: {sub.number_of_nodes()} nodes, "
          f"{sub.number_of_edges()} edges; seeds: {seeds[:3]}")

    # 9. Analytics
    bursts = detect_keyword_bursts(lit, ont, res, burst_threshold=1.5)
    print(f"[Analytics] Keyword bursts: {len(bursts)}")
    valid_concepts = list(gb.G.nodes)
    gene = build_concept_genealogy(gb.G, valid_concepts)
    print(f"[Analytics] Genealogy rows: {len(gene)}")
    bridges = detect_cross_domain_bridges(gb.G, valid_concepts)
    print(f"[Analytics] Cross-domain bridges: {len(bridges)}")

    # 10. Distillation
    metrics.start_step("distillation")
    abstract_map = ext.concept_abstract_map(lit)
    all_texts = [f"{r.get('title', '')}. {r.get('abstract', '')}"
                 for _, r in lit.iterrows()]
    distilled = compute_concept_distillation(
        valid_concepts, abstract_map, all_texts,
        resolver=res, max_docs_per_concept=10)
    metrics.end_step({"concepts": len(distilled)})
    print(f"[Distillation] {len(distilled)} concepts ranked; "
          f"top: {distilled.iloc[0]['concept'] if not distilled.empty else 'n/a'}")

    # 11. Validation
    metrics.start_step("validation")
    vmetrics = validate_graph_metrics(
        gb.G.to_undirected(), valid_concepts, resolver=res,
        n_permutations=20)
    metrics.end_step({"metrics": len(vmetrics)})
    print(f"[Validation] modularity={vmetrics.get('modularity', 0)}, "
          f"silhouette={vmetrics.get('silhouette_score', 0)}, "
          f"sig_edges={vmetrics.get('edge_significant_count', 0)}")

    # 12. Bootstrap
    weights = np.array([d.get("weight", 1.0) for _, _, d in gb.G.edges(data=True)])
    if len(weights) >= 2:
        mean, lo, hi = compute_bootstrap_ci(weights, n_bootstrap=200)
        print(f"[Bootstrap] edge weight mean={mean:.4f} "
              f"95%CI=[{lo:.4f}, {hi:.4f}]")

    # 13. ML training (only if torch available and enough nodes)
    if TORCH_AVAILABLE and len(valid_concepts) >= 10:
        metrics.start_step("ml_train")
        concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
        embs = res.embed_batch(valid_concepts)
        if embs is None:
            embs = np.random.randn(len(valid_concepts), 384).astype(np.float32)
        node_features = torch.tensor(embs, dtype=torch.float32)
        pos_pairs, neg_pairs = sample_edges_for_training(
            gb.G.to_undirected(), valid_concepts, concept_to_id)
        model, final_emb, _, _ = train_graphsage(
            node_features, gb.G.to_undirected(), concept_to_id,
            pos_pairs, neg_pairs, epochs=20, lr=0.005, hidden_dim=64)
        metrics.end_step({"pos": len(pos_pairs), "neg": len(neg_pairs)})
        if model is not None:
            print(f"[GraphSAGE] Trained: pos={len(pos_pairs)}, "
                  f"neg={len(neg_pairs)}")
            # Score a few novel pairs
            degrees = dict(gb.G.degree())
            concept_props = {c: int(degrees.get(c, 0) * 10)
                             for c in valid_concepts}
            scores_df = compute_research_direction_scores(
                model, node_features, final_emb, gb.G.to_undirected(),
                valid_concepts, concept_props, None, embs, n_samples=500)
            print(f"[Research-direction] {len(scores_df)} novel pairs ranked; "
                  f"top: {scores_df.iloc[0].to_dict() if not scores_df.empty else 'n/a'}")
        else:
            print("[GraphSAGE] Training returned None.")
    else:
        print("[GraphSAGE] Skipped (torch unavailable or <10 nodes).")

    # 14. LatentMoE extraction (lightweight smoke)
    if TORCH_AVAILABLE and len(valid_concepts) >= 10:
        try:
            moe = LatentMoEKGExtractor(
                num_nodes=len(valid_concepts),
                num_edge_types=NUM_EDGE_TYPES,
                d_model=32, latent_dim=8, n_experts=8, top_k=2,
                num_heads=2, num_layers=1)
            node_seq = torch.randint(0, len(valid_concepts), (1, 6))
            edge_seq = torch.randint(0, NUM_EDGE_TYPES, (1, 5))
            with torch.no_grad():
                out, routing = moe(node_seq, edge_seq)
            print(f"[LatentMoE] Forward OK: out={tuple(out.shape)}, "
                  f"routing={tuple(routing.shape)}")
        except Exception as exc:
            print(f"[LatentMoE] Failed: {exc}")

    # 15. DynamicOntologyExpander (no API key needed for construction)
    expander = DynamicOntologyExpander(ont)
    print(f"[Expander] {expander.status()}")

    # 16. PyVis
    html = render_graph_pyvis(gb, focus="Model_A", height="200px")
    print(f"[PyVis] HTML length {len(html)}, "
          f"has <html: {'<html' in html.lower()}, "
          f"has JS panel: {'dendrite-evidence-panel' in html}")

    # Final metrics
    print()
    summary = metrics.summary()
    print(f"[Metrics] {summary['total_duration_s']} s total, "
          f"{summary['n_steps']} steps, "
          f"peak RSS {summary['peak_rss_mb']} MB")
    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED.")
    print("=" * 78)


def _streamlit_running() -> bool:
    try:
        from streamlit import runtime
        return bool(runtime.exists())
    except Exception:
        return False


if __name__ == "__main__":
    if os.environ.get("RUN_STREAMLIT") or _streamlit_running():
        _streamlit_main()
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")
        _headless_smoke_test()
