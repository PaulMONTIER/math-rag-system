# Implémentation du Mode Hybride - Math RAG System

## Date: 2025-11-17

## Résumé

J'ai implémenté avec succès le sélecteur de modèle LLM dans le panneau latéral avec 3 options distinctes, incluant un mode hybride intelligent où les deux modèles travaillent ensemble.

## Fonctionnalités Implémentées

### 1. Sélecteur de Modèle (Sidebar)

**Emplacement**: Panneau de gauche de l'interface Streamlit

**3 Options disponibles**:

1. **Modèle fermé (GPT-4o)**
   - Utilise UNIQUEMENT GPT-4o pour toute la génération
   - Meilleure qualité de raisonnement et précision mathématique

2. **Modèle ouvert (Ollama)**
   - Utilise UNIQUEMENT le modèle local Ollama pour toute la génération
   - Plus rapide, fonctionne localement sans API externe

3. **Les deux (combinaison)**
   - **Mode hybride intelligent** où les deux modèles collaborent
   - Division intelligente du travail basée sur les forces de chaque modèle

### 2. Mode Hybride - Comment ça Fonctionne

Lorsque "Les deux (combinaison)" est sélectionné, le système exécute un workflow en 2 étapes:

#### Étape 1: Génération du Brouillon (Modèle Ouvert)
```
📝 Génération du brouillon (modèle ouvert)...
```
- Le modèle local Ollama génère une réponse initiale
- Plus rapide pour le raisonnement de base
- Extrait les documents pertinents
- Crée une première version de la réponse

#### Étape 2: Raffinement (Modèle Fermé)
```
✨ Raffinement de la réponse (modèle fermé)...
```
- GPT-4o reçoit le brouillon du modèle ouvert
- Vérifie l'exactitude mathématique
- Ajoute de la clarté et de la précision
- Améliore les explications
- Génère la réponse finale de haute qualité

#### Gestion des Erreurs
Si le modèle ouvert n'est pas disponible:
```
⚠️ Modèle ouvert indisponible, utilisation du modèle fermé...
```
Le système bascule automatiquement sur GPT-4o uniquement.

### 3. Combinaison des Sources

En mode hybride, les sources citées des deux modèles sont:
- Fusionnées automatiquement
- Dédupliquées pour éviter les répétitions
- Affichées dans la section "Sources"

## Fichiers Modifiés

### 1. src/workflow/langgraph_pipeline.py (lignes 343-369)

**Ajout du paramètre `force_provider`**:
```python
def create_rag_workflow(config: object, force_provider: Optional[str] = None) -> Any:
    """
    Crée le workflow LangGraph complet.

    Args:
        config: Objet Config
        force_provider: Provider LLM à utiliser (override config.llm.provider)
                       Options: "openai", "local"
    """
```

### 2. src/interface/app.py

#### a) Initialisation des Workflows (lignes 610-620)
Création de workflows séparés pour chaque provider:
```python
workflows = {}
workflows["openai"] = create_rag_workflow(config, force_provider="openai")
workflows["local"] = create_rag_workflow(config, force_provider="local")
```

#### b) Sélecteur dans le Sidebar (lignes 677-690)
```python
llm_choice = st.selectbox(
    "Choisir le type de modèle",
    [
        "Modèle fermé (GPT-4o)",
        "Modèle ouvert (Ollama)",
        "Les deux (combinaison)"
    ],
    help="Modèle fermé: GPT-4o uniquement | Modèle ouvert: Ollama uniquement | Les deux: combinaison intelligente"
)
```

#### c) Logique de Sélection (lignes 862-890)
Mapping du choix utilisateur vers le provider et gestion du mode hybride.

#### d) Workflow Hybride (lignes 901-984)
Implémentation complète du workflow en 2 étapes avec gestion des erreurs.

#### e) Affichage Hybride (lignes 1005-1006)
```python
if hybrid_mode:
    st.info("ℹ️ **Mode hybride activé** : Brouillon généré par le modèle ouvert (Ollama), raffiné par le modèle fermé (GPT-4o)")
```

## Workflow du Mode Hybride

```
┌─────────────────────────────────────────────────────────────┐
│                    SÉLECTION UTILISATEUR                     │
│              "Les deux (combinaison)"                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ÉTAPE 1: MODÈLE OUVERT (Ollama)                │
├─────────────────────────────────────────────────────────────┤
│ • Classification de la question                             │
│ • Recherche de documents pertinents                         │
│ • Génération du brouillon initial                           │
│ • Extraction des sources (draft_sources)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ÉTAPE 2: MODÈLE FERMÉ (GPT-4o)                 │
├─────────────────────────────────────────────────────────────┤
│ • Réception du brouillon                                    │
│ • Vérification de l'exactitude mathématique                 │
│ • Ajout de clarté et précision                              │
│ • Amélioration des explications                             │
│ • Génération de la réponse finale                           │
│ • Extraction des sources (refined_sources)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  COMBINAISON DES RÉSULTATS                   │
├─────────────────────────────────────────────────────────────┤
│ • Réponse finale = version raffinée par GPT-4o              │
│ • Sources = fusion(draft_sources, refined_sources)          │
│ • Dédupliquer les sources                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    AFFICHAGE À L'UTILISATEUR                 │
│  "Mode hybride activé : Brouillon généré par Ollama,        │
│   raffiné par GPT-4o"                                       │
└─────────────────────────────────────────────────────────────┘
```

## Avantages du Mode Hybride

1. **Rapidité**: Le modèle ouvert génère rapidement un brouillon de qualité
2. **Qualité**: Le modèle fermé raffine pour une exactitude maximale
3. **Coût**: Optimisation des coûts en utilisant le modèle local pour le travail initial
4. **Robustesse**: Bascule automatique si le modèle ouvert n'est pas disponible
5. **Transparence**: L'utilisateur voit clairement quel mode est actif

## Comment Tester

1. **Actualiser la page Streamlit** dans votre navigateur (Cmd+R ou F5)

2. **Dans le panneau de gauche**, sélectionner le modèle souhaité:
   - Modèle fermé (GPT-4o)
   - Modèle ouvert (Ollama)
   - Les deux (combinaison)

3. **Poser une question mathématique**, par exemple:
   - "Qu'est-ce qu'une dérivée ?"
   - "Explique le théorème de Pythagore"
   - "Comment calculer une intégrale définie ?"

4. **Observer le workflow**:
   - En mode hybride, vous verrez les deux étapes s'exécuter
   - Un message d'information indiquera le mode hybride
   - La réponse finale sera le résultat du raffinement

## URL d'Accès

- **Local**: http://localhost:8501
- **Réseau**: http://192.168.1.82:8501
- **Externe**: http://37.65.162.11:8501

## Configuration Requise

### Modèle Fermé (GPT-4o)
- Clé API OpenAI configurée dans `config/config.yaml`
- Connexion Internet active

### Modèle Ouvert (Ollama)
- Serveur Ollama en cours d'exécution localement
- Modèle configuré dans `config/config.yaml`
- Si Ollama n'est pas disponible, l'option sera désactivée avec un message

### Mode Hybride
- Les deux configurations ci-dessus
- Si Ollama n'est pas disponible, bascule automatiquement sur GPT-4o uniquement

## Statut

✅ **Implémenté et Testé**

- Sélecteur de modèle fonctionnel
- Mode hybride avec workflow en 2 étapes
- Gestion des erreurs robuste
- Combinaison des sources
- Interface utilisateur claire
- Messages de progression informatifs

---

**Dernière mise à jour**: 2025-11-17
**Développeur**: Claude Code
**Statut**: Prêt pour utilisation
