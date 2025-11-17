"""
Interface web Streamlit pour le système RAG hybride.

Application web localhost permettant de:
- Poser des questions mathématiques
- Voir les réponses avec sources
- Visualiser le workflow en temps réel
- Monitorer métriques et coûts

Usage:
    streamlit run src/interface/app.py

    Puis ouvrir: http://localhost:8501
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Streamlit
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Assistant Mathématiques RAG",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS pour améliorer l'UI/UX
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════
       CHARTE GRAPHIQUE - Style ChatGPT/Notion

       PALETTE DE COULEURS:
       - Fond principal: #f7f7f8 (gris très clair)
       - Fond cartes: #ffffff (blanc pur)
       - Texte principal: #2d333a (gris foncé)
       - Texte secondaire: #6e6e80 (gris moyen)
       - Bordures: #ececf1 (gris très clair)
       - Accent: #10a37f (vert ChatGPT)
       - Accent hover: #0d8c6d (vert foncé)

       ESPACEMENTS (grille 8px):
       - xs: 0.25rem (4px)
       - sm: 0.5rem (8px)
       - md: 1rem (16px)
       - lg: 1.5rem (24px)
       - xl: 2rem (32px)

       BORDER RADIUS:
       - Petits éléments: 6px
       - Cartes: 8px
       - Grands containers: 10px
    ═══════════════════════════════════════════════════════════ */

    /* Reset et base - FORCER partout */
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Forcer html et body */
    html, body, #root {
        background-color: #f7f7f8 !important;
    }

    /* FORCER tout le texte à être sombre PARTOUT */
    *, *::before, *::after,
    p, span, div, label, input, textarea, select, option,
    h1, h2, h3, h4, h5, h6,
    [class*="st"] {
        color: #2d333a !important;
    }

    /* Sauf le texte blanc sur boutons et éléments spécifiques */
    .stButton > button,
    .stButton > button *,
    button[kind="primary"],
    button[kind="primary"] * {
        color: white !important;
    }

    /* Background principal - Forcer avec !important */
    .main, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f7f7f8 !important;
    }

    .main {
        padding: 2rem 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Forcer tous les backgrounds à ne pas être noirs */
    section[data-testid="stAppViewContainer"] > div:first-child {
        background-color: #f7f7f8 !important;
    }

    /* Header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Bottom container - FORCER fond clair partout */
    [data-testid="stBottom"],
    .stChatFloatingInputContainer,
    section[data-testid="stBottom"],
    div[data-testid="stChatInputContainer"],
    .stChatInput,
    footer {
        background-color: #f7f7f8 !important;
    }

    /* Forcer TOUS les divs à ne pas avoir de fond noir */
    div[class*="st-"] {
        background-color: inherit;
    }

    /* Input container parent */
    section > div > div {
        background-color: transparent !important;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* SIDEBAR - Style cohérent */
    /* ─────────────────────────────────────────────────────────── */

    .css-1d391kg, [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #ececf1;
        padding: 1.5rem 1rem;
    }

    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2d333a !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .css-1d391kg p, .css-1d391kg label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #2d333a !important;
        font-size: 0.875rem !important;
    }

    /* Dividers dans sidebar */
    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border-color: #ececf1;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* TYPOGRAPHIE */
    /* ─────────────────────────────────────────────────────────── */

    .main-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2d333a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }

    .subtitle {
        font-size: 0.875rem;
        color: #6e6e80;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* BOUTONS - Style ChatGPT */
    /* ─────────────────────────────────────────────────────────── */

    .stButton > button {
        width: 100%;
        border-radius: 6px;
        background: #10a37f;
        color: white;
        border: none;
        padding: 0.625rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .stButton > button:hover {
        background: #0d8c6d;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    /* ─────────────────────────────────────────────────────────── */
    /* INPUTS - Selectbox et Chat Input */
    /* ─────────────────────────────────────────────────────────── */

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #ececf1;
        background: #ffffff;
        font-size: 0.875rem;
        transition: border-color 0.2s ease;
        color: #2d333a !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #d1d5db;
    }

    /* Forcer texte sombre dans selectbox */
    .stSelectbox select,
    .stSelectbox input,
    .stSelectbox div[data-baseweb="select"] > div {
        color: #2d333a !important;
    }

    /* Dropdown menu - FORCER fond blanc */
    [data-baseweb="popover"],
    [role="listbox"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }

    /* Options du dropdown */
    [role="option"],
    li[role="option"],
    [data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #2d333a !important;
    }

    /* Hover sur options */
    [role="option"]:hover,
    li[role="option"]:hover {
        background-color: #f7f7f8 !important;
        color: #2d333a !important;
    }

    /* Chat Input Container */
    .stChatInputContainer {
        border: 1px solid #ececf1;
        border-radius: 8px;
        background: #f7f7f8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Chat Input - FORCER texte sombre */
    .stChatInput textarea,
    .stChatInput input,
    textarea[data-testid="stChatInput"],
    input[type="text"] {
        color: #2d333a !important;
        background: #f7f7f8 !important;
    }

    /* Placeholder text */
    .stChatInput textarea::placeholder,
    textarea[data-testid="stChatInput"]::placeholder {
        color: #8e8ea0 !important;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* MESSAGES CHAT - Style ChatGPT */
    /* ─────────────────────────────────────────────────────────── */

    .stChatMessage {
        background: #ffffff;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        border: 1px solid #ececf1;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stChatMessageContent"] {
        color: #2d333a;
        font-size: 0.9375rem;
        line-height: 1.6;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* MÉTRIQUES - Cards cohérentes */
    /* ─────────────────────────────────────────────────────────── */

    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d333a;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: #6e6e80;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ececf1;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* EXPANDERS - Style minimaliste */
    /* ─────────────────────────────────────────────────────────── */

    .streamlit-expanderHeader,
    [data-testid="stExpander"] > summary,
    details > summary {
        background: #f7f7f8 !important;
        border: 1px solid #ececf1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        color: #2d333a !important;
        font-size: 0.875rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] > summary:hover,
    details > summary:hover {
        background: #ececf1 !important;
        border-color: #d1d5db !important;
    }

    .streamlit-expanderContent,
    [data-testid="stExpander"] > div,
    details > div {
        border: 1px solid #ececf1 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        background: #ffffff !important;
        padding: 1rem !important;
    }

    /* Forcer TOUT dans les expanders à avoir un fond clair */
    [data-testid="stExpander"],
    details[open] {
        background-color: transparent !important;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* SOURCES - Cards distinctives */
    /* ─────────────────────────────────────────────────────────── */

    .source-item {
        background: #f7f7f8;
        padding: 0.75rem 1rem;
        border-left: 3px solid #10a37f;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.875rem;
        color: #2d333a;
        transition: all 0.2s ease;
    }

    .source-item:hover {
        background: #ececf1;
        border-left-color: #0d8c6d;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* PROGRESS BAR */
    /* ─────────────────────────────────────────────────────────── */

    .stProgress > div > div {
        background: #10a37f;
        border-radius: 4px;
    }

    .stProgress > div {
        background: #ececf1;
        border-radius: 4px;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* STATUS MESSAGES */
    /* ─────────────────────────────────────────────────────────── */

    .stInfo {
        background: #f0f9ff;
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        padding: 1rem;
        color: #1e40af;
    }

    .stSuccess {
        background: #f0fdf4;
        border-left: 3px solid #10a37f;
        border-radius: 0 6px 6px 0;
        padding: 1rem;
        color: #065f46;
    }

    .stError {
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        border-radius: 0 6px 6px 0;
        padding: 1rem;
        color: #991b1b;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* CAPTIONS */
    /* ─────────────────────────────────────────────────────────── */

    .stCaption {
        color: #8e8ea0;
        font-size: 0.8125rem;
        line-height: 1.5;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* FOOTER */
    /* ─────────────────────────────────────────────────────────── */

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #8e8ea0;
        font-size: 0.8125rem;
        border-top: 1px solid #ececf1;
        margin-top: 3rem;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* LATEX RENDERING - Améliorer lisibilité */
    /* ─────────────────────────────────────────────────────────── */

    .katex {
        font-size: 1.05em;
    }

    .katex-display {
        margin: 1.25rem 0;
        padding: 1rem;
        background: #f7f7f8;
        border-radius: 6px;
        overflow-x: auto;
    }

    /* ─────────────────────────────────────────────────────────── */
    /* SUGGESTIONS - Boutons cliquables */
    /* ─────────────────────────────────────────────────────────── */

    .suggestions-title {
        font-size: 0.875rem;
        color: #6e6e80;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Conteneur suggestions avec espacement */
    .suggestion-button-container {
        margin: 0.5rem 0;
    }

    /* Style minimal pour les boutons suggestions - pas d'override avec !important */
    button[data-testid^="baseButton-secondary"] {
        min-height: 60px;
        white-space: normal;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions - LaTeX Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_text_with_latex(text: str):
    """
    Rend le texte avec formules LaTeX VISUELLEMENT (pas en code).

    Utilise st.markdown avec KaTeX pour afficher les formules.
    - Formules display: $$...$$ → centrées
    - Formules inline: $...$ → dans le texte

    Args:
        text: Texte contenant des formules LaTeX
    """
    import re

    # Remplacer \[...\] par $$...$$ pour uniformiser
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    # Séparer le texte en parties (texte normal vs formules)
    parts = []
    last_end = 0

    # Pattern pour détecter les formules display $$...$$
    display_pattern = r'\$\$(.*?)\$\$'
    for match in re.finditer(display_pattern, text, re.DOTALL):
        # Ajouter texte avant la formule
        if match.start() > last_end:
            parts.append(('text', text[last_end:match.start()]))

        # Ajouter formule display
        parts.append(('display', match.group(1)))
        last_end = match.end()

    # Ajouter le reste du texte
    if last_end < len(text):
        parts.append(('text', text[last_end:]))

    # Afficher chaque partie
    for part_type, content in parts:
        if part_type == 'display':
            # Formule display: utiliser st.latex()
            st.latex(content)
        else:
            # Texte normal (peut contenir formules inline): utiliser st.markdown
            st.markdown(content, unsafe_allow_html=True)


def display_suggestions(suggestions: list, message_idx: int):
    """
    Affiche les suggestions de questions de suivi comme boutons cliquables.

    Args:
        suggestions: Liste de suggestions (max 3)
        message_idx: Index du message dans l'historique (pour clés uniques)
    """
    if not suggestions or len(suggestions) == 0:
        return

    # Limiter à 3 suggestions
    suggestions = suggestions[:3]

    st.markdown('<div class="suggestions-title">Pour aller plus loin :</div>', unsafe_allow_html=True)

    # Afficher en colonnes
    cols = st.columns(len(suggestions))

    for idx, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            # Créer un bouton pour chaque suggestion
            button_key = f"suggest_{message_idx}_{idx}"
            if st.button(
                suggestion,
                key=button_key,
                type="secondary",
                use_container_width=True
            ):
                # Stocker la suggestion cliquée dans session_state
                st.session_state.clicked_suggestion = suggestion
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_system():
    """
    Initialise le système (une seule fois).

    Crée plusieurs workflows pour différents providers LLM.

    Returns:
        Tuple (config, workflows_dict)
    """
    with st.spinner("⏳ Initialisation du système..."):
        try:
            config = load_config()

            # Créer workflows pour chaque provider
            workflows = {}

            # GPT-4o (OpenAI) - Modèle fermé
            workflows["openai"] = create_rag_workflow(config, force_provider="openai")

            # Ollama (local) - Modèle ouvert
            try:
                workflows["local"] = create_rag_workflow(config, force_provider="local")
            except Exception as e:
                workflows["local"] = None  # Ollama non disponible

            return config, workflows
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation: {e}")
            st.stop()


# Initialiser
config, workflows = init_system()


# ═══════════════════════════════════════════════════════════════════════════════
# State session
# ═══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# Paramètres avancés de personnalisation
if "rigor_level" not in st.session_state:
    st.session_state.rigor_level = 3

if "num_examples" not in st.session_state:
    st.session_state.num_examples = 2

if "include_proofs" not in st.session_state:
    st.session_state.include_proofs = True

if "include_history" not in st.session_state:
    st.session_state.include_history = False

if "detailed_latex" not in st.session_state:
    st.session_state.detailed_latex = True

# Choix du modèle LLM
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Modèle fermé (GPT-4o)"


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar - Configuration et métriques
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Sélection du modèle LLM
    st.markdown("### Modèle de génération")

    # Déterminer les options disponibles selon la disponibilité d'Ollama
    available_options = ["Modèle fermé (GPT-4o)"]

    if workflows.get("local") is not None:
        # Ollama disponible, ajouter les options
        available_options.extend([
            "Modèle ouvert (Ollama)",
            "Les deux (combinaison)"
        ])
    else:
        # Ollama non disponible, afficher avertissement
        st.warning("⚠️ Ollama non disponible. Seul GPT-4o est utilisable.")

    llm_choice = st.selectbox(
        "Choisir le type de modèle",
        available_options,
        index=0,
        label_visibility="collapsed",
        help="Modèle fermé: GPT-4o uniquement | Modèle ouvert: Ollama uniquement | Les deux: combinaison intelligente"
    )
    st.session_state.llm_choice = llm_choice

    st.divider()

    # Niveau de détail
    st.markdown("### Niveau de détail")
    student_level = st.selectbox(
        "Choisir le niveau",
        ["Simple", "Détaillé", "Beaucoup de détails"],
        index=1,
        label_visibility="collapsed"
    )

    st.divider()

    # Personnalisation avancée
    st.markdown("### Personnalisation")

    # Rigueur mathématique
    st.session_state.rigor_level = st.slider(
        "Rigueur mathématique",
        min_value=1,
        max_value=5,
        value=st.session_state.rigor_level,
        help="1 = Intuitif, 5 = Très rigoureux et formel"
    )

    # Nombre d'exemples
    st.session_state.num_examples = st.slider(
        "Nombre d'exemples",
        min_value=0,
        max_value=3,
        value=st.session_state.num_examples,
        help="Nombre d'exemples concrets à inclure"
    )

    # Options supplémentaires
    st.session_state.include_proofs = st.checkbox(
        "Inclure démonstrations",
        value=st.session_state.include_proofs,
        help="Ajouter des démonstrations détaillées"
    )

    st.session_state.include_history = st.checkbox(
        "Ajouter contexte historique",
        value=st.session_state.include_history,
        help="Inclure l'origine et l'histoire du concept"
    )

    st.session_state.detailed_latex = st.checkbox(
        "Formules LaTeX détaillées",
        value=st.session_state.detailed_latex,
        help="Développer les formules avec étapes intermédiaires"
    )

    st.divider()

    # Métriques session
    st.markdown("### Statistiques")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.question_count)
    with col2:
        st.metric("Coût", f"${st.session_state.total_cost:.4f}")

    st.divider()

    # Bouton reset
    if st.button("Réinitialiser", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.question_count = 0
        st.session_state.rigor_level = 3
        st.session_state.num_examples = 2
        st.session_state.include_proofs = True
        st.session_state.include_history = False
        st.session_state.detailed_latex = True
        st.rerun()

    st.divider()

    # Informations système
    st.markdown("### Système")
    st.caption(f"**Modèle par défaut:** {config.llm.model}")
    st.caption(f"**Provider par défaut:** {config.llm.provider}")

    # Afficher les providers disponibles
    providers_available = []
    if workflows.get("openai"):
        providers_available.append("✅ GPT-4o")
    if workflows.get("local"):
        providers_available.append("✅ Ollama")
    else:
        providers_available.append("❌ Ollama")

    st.caption(f"**Providers disponibles:** {', '.join(providers_available)}")
    st.caption(f"**Embeddings:** {config.embeddings.model.split('/')[-1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main - Interface chat
# ═══════════════════════════════════════════════════════════════════════════════

# Titre principal minimaliste
st.markdown(f"""
<div class="main-title">Assistant Mathématiques</div>
<div class="subtitle">Niveau: {student_level}</div>
""", unsafe_allow_html=True)

# Afficher historique
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Render avec formules LaTeX visuelles
        render_text_with_latex(message["content"])

        # Afficher sources si présentes
        if "sources" in message and message["sources"]:
            with st.expander("Sources", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class="source-item">{source}</div>
                    """, unsafe_allow_html=True)

        # Afficher métadonnées si présentes
        if "metadata" in message and message["metadata"]:
            with st.expander("Détails", expanded=False):
                meta = message["metadata"]

                # Métriques de génération
                if "generation" in meta:
                    gen = meta["generation"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens", gen.get("tokens", "N/A"))
                    with col2:
                        st.metric("Coût", f"${gen.get('cost', 0):.4f}")
                    with col3:
                        st.metric("Temps", f"{gen.get('generation_time', 0):.2f}s")

                # Métriques de retrieval
                if "retrieval" in meta:
                    ret = meta["retrieval"]
                    st.caption(f"Documents: {ret.get('docs_found', 0)} • Score moyen: {ret.get('avg_score', 0):.3f}")

        # Afficher suggestions pour messages assistant
        if message["role"] == "assistant" and "metadata" in message and message["metadata"]:
            # Les suggestions sont dans metadata.generation.suggestions
            generation_meta = message["metadata"].get("generation", {})
            suggestions = generation_meta.get("suggestions", [])
            if suggestions:
                display_suggestions(suggestions, idx)


# Input utilisateur - toujours affiché
user_input = st.chat_input("Posez votre question mathématique...")

# Déterminer quelle question traiter (suggestion cliquée prioritaire)
clicked_suggestion = st.session_state.get("clicked_suggestion", None)
if clicked_suggestion:
    # Suggestion cliquée = priorité
    prompt = clicked_suggestion
    del st.session_state.clicked_suggestion
elif user_input:
    # Sinon, utiliser l'input utilisateur
    prompt = user_input
else:
    prompt = None

if prompt:
    # Ajouter message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer réponse
    with st.chat_message("assistant"):
        # Placeholder pour status
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            # Mapper choix utilisateur vers provider
            llm_choice_to_provider = {
                "Modèle fermé (GPT-4o)": "openai",
                "Modèle ouvert (Ollama)": "local",
                "Les deux (combinaison)": "hybrid"
            }
            provider = llm_choice_to_provider.get(llm_choice, "openai")

            # Sélectionner le workflow approprié
            if provider == "hybrid":
                # Mode hybride: combinaison intelligente des deux modèles
                # Utiliser modèle ouvert pour retrieval/classification, fermé pour génération
                workflow_open = workflows.get("local")
                workflow_closed = workflows.get("openai")

                if workflow_open is None:
                    st.warning("⚠️ Modèle ouvert (Ollama) non disponible. Utilisation de GPT-4o uniquement.")
                    workflow_1 = workflow_closed
                    hybrid_mode = False
                else:
                    workflow_1 = workflow_closed  # Génération finale avec GPT-4o
                    hybrid_mode = True
            else:
                # Mode simple: un seul modèle
                workflow_1 = workflows.get(provider)
                if workflow_1 is None:
                    st.error(f"❌ Le modèle sélectionné n'est pas disponible. Vérifiez la configuration.")
                    st.stop()
                hybrid_mode = False

            # Workflow steps
            with status_placeholder.container():
                st.info("⏳ Traitement en cours...")

                # Progress bar simulé
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔍 Classification de la question...")
                progress_bar.progress(20)
                time.sleep(0.2)

                status_text.text("📚 Recherche de documents pertinents...")
                progress_bar.progress(40)

                # Mode hybride: combinaison des deux modèles
                if hybrid_mode:
                    # Étape 1: Modèle ouvert génère un brouillon
                    status_text.text("📝 Génération du brouillon (modèle ouvert)...")
                    progress_bar.progress(50)

                    draft_result = invoke_workflow(
                        workflow=workflow_open,
                        question=prompt,
                        student_level=student_level,
                        rigor_level=st.session_state.rigor_level,
                        num_examples=st.session_state.num_examples,
                        include_proofs=st.session_state.include_proofs,
                        include_history=st.session_state.include_history,
                        detailed_latex=st.session_state.detailed_latex
                    )

                    if not draft_result["success"]:
                        # Si le modèle ouvert échoue, utiliser uniquement le modèle fermé
                        status_text.text("⚠️ Modèle ouvert indisponible, utilisation du modèle fermé...")
                        progress_bar.progress(60)

                        result = invoke_workflow(
                            workflow=workflow_1,
                            question=prompt,
                            student_level=student_level,
                            rigor_level=st.session_state.rigor_level,
                            num_examples=st.session_state.num_examples,
                            include_proofs=st.session_state.include_proofs,
                            include_history=st.session_state.include_history,
                            detailed_latex=st.session_state.detailed_latex
                        )
                    else:
                        # Étape 2: Modèle fermé raffine le brouillon
                        draft_response = draft_result["final_response"]

                        status_text.text("✨ Raffinement de la réponse (modèle fermé)...")
                        progress_bar.progress(60)

                        # Question modifiée pour le raffinement
                        refinement_question = f"""Question originale: {prompt}

Un modèle a généré cette réponse initiale:

{draft_response}

Améliore et raffine cette réponse en:
1. Vérifiant l'exactitude mathématique
2. Ajoutant de la clarté et de la précision
3. Améliorant les explications
4. Conservant le même niveau de détail ({student_level})

Génère une réponse finale de haute qualité."""

                        result = invoke_workflow(
                            workflow=workflow_1,
                            question=refinement_question,
                            student_level=student_level,
                            rigor_level=st.session_state.rigor_level,
                            num_examples=st.session_state.num_examples,
                            include_proofs=st.session_state.include_proofs,
                            include_history=st.session_state.include_history,
                            detailed_latex=st.session_state.detailed_latex
                        )

                        # Combiner les sources des deux modèles
                        if result["success"]:
                            draft_sources = draft_result.get("sources_cited") or []
                            refined_sources = result.get("sources_cited") or []
                            # Fusionner et dédupliquer les sources
                            all_sources = list(set(draft_sources + refined_sources))
                            result["sources_cited"] = all_sources
                else:
                    # Mode simple: un seul modèle
                    result = invoke_workflow(
                        workflow=workflow_1,
                        question=prompt,
                        student_level=student_level,
                        rigor_level=st.session_state.rigor_level,
                        num_examples=st.session_state.num_examples,
                        include_proofs=st.session_state.include_proofs,
                        include_history=st.session_state.include_history,
                        detailed_latex=st.session_state.detailed_latex
                    )

                status_text.text("✍️ Génération de la réponse...")
                progress_bar.progress(70)
                time.sleep(0.2)

                status_text.text("✅ Vérification de la qualité...")
                progress_bar.progress(90)
                time.sleep(0.2)

                progress_bar.progress(100)
                status_text.text("✓ Terminé!")
                time.sleep(0.3)

            # Effacer status, afficher réponse
            status_placeholder.empty()

            if result["success"]:
                # Afficher réponse
                with response_placeholder.container():
                    # Afficher mode sélectionné si hybride
                    if hybrid_mode:
                        st.info("ℹ️ **Mode hybride activé** : Brouillon généré par le modèle ouvert (Ollama), raffiné par le modèle fermé (GPT-4o)")

                    render_text_with_latex(result["final_response"])

                # Extraire sources
                sources = result.get("sources_cited", [])

                # Afficher sources
                if sources:
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            st.markdown(f"""
                            <div class="source-item">{source}</div>
                            """, unsafe_allow_html=True)

                # Afficher métadonnées
                with st.expander("Détails", expanded=False):
                    meta = result.get("metadata", {})

                    # Métriques de génération
                    if "generation" in meta:
                        gen = meta["generation"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tokens", gen.get("tokens", "N/A"))
                        with col2:
                            st.metric("Coût", f"${gen.get('cost', 0):.4f}")
                        with col3:
                            st.metric("Temps", f"{meta.get('total_time', 0):.2f}s")

                    # Métriques de retrieval
                    if "retrieval" in meta:
                        ret = meta["retrieval"]
                        st.caption(f"Documents: {ret.get('docs_found', 0)} • Score moyen: {ret.get('avg_score', 0):.3f}")

                # Afficher suggestions pour la nouvelle réponse
                generation_meta = result.get("metadata", {}).get("generation", {})
                suggestions = generation_meta.get("suggestions", [])
                if suggestions:
                    # Utiliser l'index du prochain message (qui sera ajouté)
                    display_suggestions(suggestions, len(st.session_state.messages))

                # Ajouter au chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["final_response"],
                    "sources": sources,
                    "metadata": result.get("metadata", {})
                })

                # Mettre à jour métriques
                st.session_state.question_count += 1

                if "generation" in result.get("metadata", {}):
                    cost = result["metadata"]["generation"].get("cost", 0)
                    st.session_state.total_cost += cost

            else:
                # Erreur
                error_msg = result.get("error_message", "Erreur inconnue")
                response_placeholder.error(f"❌ Erreur: {error_msg}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Erreur: {error_msg}"
                })

        except Exception as e:
            status_placeholder.empty()
            response_placeholder.error(f"❌ Erreur système: {e}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Erreur système: {e}"
            })

            logger.error(f"Interface error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <p>Système RAG hybride • Embeddings open-source + GPT-4o</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# STREAMLIT:
# - Framework web Python pour ML/Data apps
# - Réactif (auto-rerun quand input change)
# - Composants natifs (chat, metrics, expanders, etc.)
# - st.session_state pour persistance entre reruns
#
# LAYOUT:
# - Sidebar: Configuration + Métriques
# - Main: Chat interface
# - Expanders: Sources + Détails techniques
#
# CHAT:
# - st.chat_message() pour affichage type ChatGPT
# - st.chat_input() pour input utilisateur
# - st.session_state.messages pour historique
#
# WORKFLOW STATUS:
# - Progress bar simulée avec étapes
# - Async pas nécessaire (Streamlit single-threaded)
# - Status effacé quand réponse prête
#
# MÉTRIQUES:
# - Compteurs session (questions, coût)
# - Par message (tokens, temps, confiance)
# - Sidebar pour vue d'ensemble
#
# EXTENSIONS POSSIBLES:
# - Export conversation (PDF, MD)
# - Feedback thumbs up/down
# - Graphes métriques (plotly)
# - Visualisation workflow (graphviz)
# - Mode dark/light
#
# LANCEMENT:
# streamlit run src/interface/app.py
# Ou: make run
#
# ═══════════════════════════════════════════════════════════════════════════════
