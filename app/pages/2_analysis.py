"""2 · Analiz — estimand-first + spec menüsü.

SPRINT-2: estimand/H0-H1 elicitation (Socratic/TAC) → dondur → savunulabilir spec
menüsü (yargı LLM) → aydınlık yol (4 fiil) → dondur (hash) → multiverse başlat.
Menü dondurma ve faktöriyel genişletme çekirdeği (menu.py) şimdiden hazır.
"""

from __future__ import annotations

import streamlit as st

st.title("2 · Analiz")
st.info(
    "Estimand elicitation ve LLM spec-menü üretimi Sprint-2. Altyapı hazır: "
    "`pareto.analysis.hypothesis.Estimand` (donmuş), `pareto.analysis.menu.FrozenSpecMenu` "
    "(hash'li dondurma) + `expand_to_specs` (faktöriyel, sert tavan 24)."
)
