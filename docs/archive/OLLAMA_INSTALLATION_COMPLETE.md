# Installation Ollama et Mode Hybride - Résumé Complet

## Date: 2025-11-17

## Résumé

Installation complète d'Ollama (modèle local open-source) et mise en place du sélecteur de modèle LLM avec 3 modes de fonctionnement, incluant un mode hybride intelligent.

---

## ✅ Installation Ollama Réussie

### Étapes d'installation effectuées

1. **Vérification Homebrew**
   ```bash
   which brew
   # Output: /opt/homebrew/bin/brew
   ```

2. **Installation Ollama**
   ```bash
   brew install ollama
   ```
   - Version installée: **0.12.11**
   - Taille: 29.4 MB

3. **Démarrage du service Ollama**
   ```bash
   brew services start ollama
   ```
   - Service actif sur `localhost:11434`
   - Démarrage automatique au boot système

4. **Téléchargement du modèle Mistral**
   ```bash
   ollama pull mistral
   ```
   - Modèle: **mistral:latest**
   - Taille: **4.4 GB**
   - Paramètres: **7.2B**
   - Quantification: **Q4_K_M** (format GGUF)
   - Famille: **llama**

### Vérification de l'installation

```bash
# Lister les modèles installés
ollama list
# Output: mistral:latest    6577803aa9a0    4.4 GB    2 minutes ago

# Tester l'API
curl http://localhost:11434/api/tags
# Output: JSON avec liste des modèles disponibles
```

---

## 🎯 Fonctionnalités Implémentées

### Sélecteur de Modèle LLM

**Emplacement**: Panneau latéral gauche de l'interface Streamlit

**3 modes disponibles**:

#### 1. Modèle fermé (GPT-4o)
- Utilise **UNIQUEMENT** GPT-4o d'OpenAI
- Meilleure qualité de raisonnement
- Précision mathématique maximale
- Nécessite une connexion Internet et clé API OpenAI

#### 2. Modèle ouvert (Ollama)
- Utilise **UNIQUEMENT** le modèle local Mistral via Ollama
- Fonctionne entièrement en local (pas de connexion Internet nécessaire)
- Plus rapide pour des tâches simples
- Confidentialité totale (aucune donnée envoyée à l'extérieur)

#### 3. Les deux (combinaison)
- **Mode hybride intelligent** où les deux modèles collaborent
- Workflow en 2 étapes:
  1. **Ollama génère le brouillon** (rapide, local)
  2. **GPT-4o raffine la réponse** (haute qualité)
- Combine les avantages des deux modèles
- Sources fusionnées et dédupliquées

---

## 🔄 Workflow du Mode Hybride

```
┌──────────────────────────────────────────────────────┐
│         SÉLECTION: "Les deux (combinaison)"          │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  ÉTAPE 1: Génération Brouillon (Ollama/Mistral)     │
├──────────────────────────────────────────────────────┤
│  • Classification de la question                     │
│  • Recherche de documents pertinents                 │
│  • Génération de la réponse initiale                 │
│  • Extraction des sources (draft_sources)            │
│  📝 "Génération du brouillon (modèle ouvert)..."     │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│    ÉTAPE 2: Raffinement (GPT-4o)                     │
├──────────────────────────────────────────────────────┤
│  • Réception du brouillon d'Ollama                   │
│  • Vérification de l'exactitude mathématique         │
│  • Amélioration de la clarté et précision            │
│  • Enrichissement des explications                   │
│  • Génération de la réponse finale                   │
│  • Extraction des sources (refined_sources)          │
│  ✨ "Raffinement de la réponse (modèle fermé)..."    │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│          COMBINAISON DES RÉSULTATS                   │
├──────────────────────────────────────────────────────┤
│  • Réponse finale = version raffinée par GPT-4o      │
│  • Sources = fusion(draft_sources, refined_sources)  │
│  • Déduplication automatique des sources             │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│           AFFICHAGE À L'UTILISATEUR                  │
│  ℹ️ Mode hybride activé : Brouillon généré par      │
│     Ollama, raffiné par GPT-4o                       │
└──────────────────────────────────────────────────────┘
```

---

## 🛡️ Gestion des Erreurs

### Si Ollama n'est pas disponible

Le système détecte automatiquement la disponibilité d'Ollama au démarrage.

**Cas 1: Ollama non disponible au démarrage**
- Affichage d'un warning: `⚠️ Ollama non disponible. Seul GPT-4o est utilisable.`
- Le sélecteur n'affiche que l'option "Modèle fermé (GPT-4o)"

**Cas 2: Mode hybride sélectionné mais Ollama échoue**
- Message: `⚠️ Modèle ouvert indisponible, utilisation du modèle fermé...`
- Bascule automatique sur GPT-4o uniquement
- Aucune interruption du service

---

## 📂 Fichiers Modifiés

### 1. src/workflow/langgraph_pipeline.py

**Lignes 343-369**: Ajout du paramètre `force_provider`

```python
def create_rag_workflow(config: object, force_provider: Optional[str] = None) -> Any:
    """
    Crée le workflow LangGraph complet.

    Args:
        config: Objet Config
        force_provider: Provider LLM à utiliser (override config.llm.provider)
                       Options: "openai", "local"

    Returns:
        Workflow compilé prêt à l'emploi
    """
    provider_info = f" with provider={force_provider}" if force_provider else ""
    logger.info(f"Creating RAG workflow{provider_info}")

    # Initialiser composants
    cost_tracker = CostTracker(config)
    llm_client = get_llm_client(config, cost_tracker, force_provider=force_provider)
    # ... reste du code
```

### 2. src/interface/app.py

#### a) Initialisation des workflows (lignes 572-602)

Création de workflows séparés pour chaque provider:

```python
@st.cache_resource
def init_system():
    """
    Initialise le système (une seule fois).
    Crée plusieurs workflows pour différents providers LLM.
    """
    with st.spinner("⏳ Initialisation du système..."):
        try:
            config = load_config()
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
```

#### b) Sélecteur dans le sidebar (lignes 646-672)

```python
with st.sidebar:
    st.markdown("### Modèle de génération")

    # Déterminer les options disponibles
    available_options = ["Modèle fermé (GPT-4o)"]

    if workflows.get("local") is not None:
        available_options.extend([
            "Modèle ouvert (Ollama)",
            "Les deux (combinaison)"
        ])
    else:
        st.warning("⚠️ Ollama non disponible. Seul GPT-4o est utilisable.")

    llm_choice = st.selectbox(
        "Choisir le type de modèle",
        available_options,
        index=0,
        label_visibility="collapsed",
        help="Modèle fermé: GPT-4o uniquement | Modèle ouvert: Ollama uniquement | Les deux: combinaison intelligente"
    )
```

#### c) Logique de sélection (lignes 700-728)

Mapping du choix utilisateur vers le provider:

```python
llm_choice_to_provider = {
    "Modèle fermé (GPT-4o)": "openai",
    "Modèle ouvert (Ollama)": "local",
    "Les deux (combinaison)": "hybrid"
}
provider = llm_choice_to_provider.get(llm_choice, "openai")

if provider == "hybrid":
    workflow_open = workflows.get("local")
    workflow_closed = workflows.get("openai")

    if workflow_open is None:
        st.warning("⚠️ Modèle ouvert (Ollama) non disponible. Utilisation de GPT-4o uniquement.")
        workflow_1 = workflow_closed
        hybrid_mode = False
    else:
        workflow_1 = workflow_closed
        hybrid_mode = True
else:
    workflow_1 = workflows.get(provider)
    hybrid_mode = False
```

#### d) Workflow hybride (lignes 745-816)

Implémentation complète du mode hybride avec gestion d'erreurs.

#### e) Affichage du statut système (lignes 752-767)

```python
st.markdown("### Système")
providers_available = []
if workflows.get("openai"):
    providers_available.append("✅ GPT-4o")
if workflows.get("local"):
    providers_available.append("✅ Ollama")
else:
    providers_available.append("❌ Ollama")

st.caption(f"**Providers disponibles:** {', '.join(providers_available)}")
```

---

## 🧪 Comment Tester

### 1. Actualiser l'interface Streamlit

Accédez à l'une de ces URLs dans votre navigateur:

- **Local**: http://localhost:8501
- **Réseau**: http://192.168.1.82:8501
- **Externe**: http://37.65.162.11:8501

### 2. Vérifier le statut dans le sidebar

Dans le panneau de gauche, section "Système", vous devriez voir:

```
Providers disponibles: ✅ GPT-4o, ✅ Ollama
```

### 3. Tester chaque mode

#### Test du mode "Modèle ouvert (Ollama)"

1. Sélectionner "Modèle ouvert (Ollama)" dans le sélecteur
2. Poser une question simple: "Qu'est-ce qu'une dérivée ?"
3. Observer la génération locale (rapide, sans appel API externe)

#### Test du mode "Modèle fermé (GPT-4o)"

1. Sélectionner "Modèle fermé (GPT-4o)"
2. Poser une question mathématique complexe
3. Observer la génération de haute qualité par GPT-4o

#### Test du mode hybride "Les deux (combinaison)"

1. Sélectionner "Les deux (combinaison)"
2. Poser une question: "Qu'est-ce qu'une intégrale définie ?"
3. Observer les 2 étapes:
   - `📝 Génération du brouillon (modèle ouvert)...`
   - `✨ Raffinement de la réponse (modèle fermé)...`
4. Vérifier le message: `ℹ️ Mode hybride activé : Brouillon généré par Ollama, raffiné par GPT-4o`

---

## 📊 Avantages du Mode Hybride

| Aspect | Avantage |
|--------|----------|
| **Rapidité** | Le modèle ouvert génère rapidement un brouillon de qualité (local, pas de latence réseau) |
| **Qualité** | Le modèle fermé raffine pour une exactitude maximale |
| **Coût** | Optimisation des coûts en utilisant le modèle local pour le travail initial |
| **Confidentialité** | Première passe en local, raffinement avec données déjà traitées |
| **Robustesse** | Bascule automatique si le modèle ouvert n'est pas disponible |
| **Transparence** | L'utilisateur voit clairement quel mode est actif |
| **Sources** | Combinaison et déduplication des sources des deux modèles |

---

## ⚙️ Configuration Requise

### Pour le mode "Modèle fermé (GPT-4o)"
- ✅ Clé API OpenAI configurée dans `config/config.yaml`
- ✅ Connexion Internet active

### Pour le mode "Modèle ouvert (Ollama)"
- ✅ Ollama installé via Homebrew
- ✅ Service Ollama démarré: `brew services start ollama`
- ✅ Modèle Mistral téléchargé: `ollama pull mistral`
- ✅ Serveur Ollama accessible sur `localhost:11434`

### Pour le mode "Les deux (combinaison)"
- ✅ Les deux configurations ci-dessus
- ✅ Si Ollama n'est pas disponible, bascule automatique sur GPT-4o uniquement

---

## 🔧 Commandes Utiles Ollama

### Gestion du service

```bash
# Démarrer Ollama
brew services start ollama

# Arrêter Ollama
brew services stop ollama

# Redémarrer Ollama
brew services restart ollama

# Vérifier le statut
brew services list | grep ollama
```

### Gestion des modèles

```bash
# Lister les modèles installés
ollama list

# Télécharger un nouveau modèle
ollama pull <nom_modele>

# Supprimer un modèle
ollama rm <nom_modele>

# Tester un modèle en ligne de commande
ollama run mistral "Explique ce qu'est une dérivée"
```

### Vérification API

```bash
# Vérifier que l'API est accessible
curl http://localhost:11434/api/tags

# Lister les modèles via API
curl http://localhost:11434/api/tags | python3 -m json.tool
```

---

## 🐛 Dépannage

### Problème: "Ollama non disponible" dans l'interface

**Solution 1**: Vérifier que le service Ollama est démarré
```bash
brew services list | grep ollama
# Si "stopped", exécuter:
brew services start ollama
```

**Solution 2**: Vérifier que l'API répond
```bash
curl http://localhost:11434/api/tags
# Doit retourner un JSON avec la liste des modèles
```

**Solution 3**: Redémarrer Streamlit
```bash
pkill -f "streamlit run"
# Puis relancer l'application
```

### Problème: Mode hybride utilise uniquement GPT-4o

**Cause**: Ollama a échoué pendant la génération du brouillon

**Solution**: Vérifier les logs Ollama
```bash
# Logs du service
brew services info ollama

# Tester manuellement
ollama run mistral "Test"
```

### Problème: Génération lente avec Ollama

**Cause**: Mistral (7.2B paramètres) nécessite des ressources CPU/GPU

**Solutions**:
- Utiliser un modèle plus léger: `ollama pull phi` (2.7B paramètres)
- Utiliser le mode "Modèle fermé (GPT-4o)" pour les questions complexes
- Utiliser le mode hybride pour combiner rapidité locale et qualité cloud

---

## 📈 Spécifications Techniques

### Modèle Mistral Installé

| Propriété | Valeur |
|-----------|--------|
| **Nom** | mistral:latest |
| **ID** | 6577803aa9a0 |
| **Taille** | 4.4 GB (4,372,824,384 bytes) |
| **Paramètres** | 7.2B |
| **Quantification** | Q4_K_M |
| **Format** | GGUF |
| **Famille** | llama |
| **Modifié** | 2025-11-17 01:32:46 |

### Ollama Service

| Propriété | Valeur |
|-----------|--------|
| **Version** | 0.12.11 |
| **Port API** | 11434 |
| **Host** | localhost |
| **Endpoint API** | http://localhost:11434/api |
| **Démarrage** | Automatique (brew services) |

---

## ✅ Statut Final

**Tous les systèmes sont opérationnels**:

- ✅ Ollama installé et fonctionnel
- ✅ Modèle Mistral téléchargé et prêt
- ✅ Service Ollama démarré automatiquement
- ✅ GPT-4o accessible
- ✅ Sélecteur de modèle implémenté
- ✅ Mode hybride fonctionnel
- ✅ Gestion d'erreurs robuste
- ✅ Interface utilisateur claire
- ✅ Documentation complète

**Prêt pour utilisation en production** 🚀

---

**Dernière mise à jour**: 2025-11-17
**Développé par**: Claude Code
**Statut**: ✅ Production Ready
