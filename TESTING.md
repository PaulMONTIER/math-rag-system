# Guide de Test - Système RAG Mathématique

Ce document décrit comment tester le système RAG étape par étape, depuis la configuration initiale jusqu'au déploiement.

---

## Table des matières

1. [Pré-requis](#pré-requis)
2. [Tests de configuration](#tests-de-configuration)
3. [Tests des modules de base](#tests-des-modules-de-base)
4. [Tests de vectorisation (CRITIQUE)](#tests-de-vectorisation-critique)
5. [Tests du workflow complet](#tests-du-workflow-complet)
6. [Tests de l'interface](#tests-de-linterface)
7. [Tests automatisés](#tests-automatisés)
8. [Validation finale](#validation-finale)
9. [Troubleshooting](#troubleshooting)

---

## Pré-requis

### 1. Installation des dépendances

```bash
# Installation complète
make install

# Ou avec pip directement
pip install -r requirements.txt
```

### 2. Configuration de l'environnement

```bash
# Copier .env.example vers .env
cp .env.example .env

# Éditer .env et ajouter vos clés API
nano .env
```

**Clés requises:**
- `OPENAI_API_KEY` - Clé OpenAI (génération)
- `GDRIVE_FOLDER_ID` - ID du dossier Google Drive (optionnel)
- `LANGFUSE_PUBLIC_KEY` et `LANGFUSE_SECRET_KEY` - Monitoring (optionnel)

### 3. Vérification de la configuration

```bash
# Tester que la config se charge
python -c "from src.utils.config_loader import load_config; c = load_config(); print('✓ Config OK')"
```

**Résultat attendu:**
```
✓ Config OK
```

---

## Tests de configuration

### Test 1: Logger

```bash
python -c "
from src.utils.logger import get_logger
logger = get_logger('test')
logger.info('Test message')
print('✓ Logger OK')
"
```

**Vérifier:**
- Pas d'erreur
- Fichier `data/logs/app.log` créé
- Message apparaît dans le log

### Test 2: Cost Tracker

```bash
python -c "
from src.utils.config_loader import load_config
from src.utils.cost_tracker import CostTracker
config = load_config()
tracker = CostTracker(config)
cost = tracker.calculate_cost('gpt-4o', 100, 50)
print(f'Cost: \${cost[2]:.4f}')
print('✓ Cost Tracker OK')
"
```

**Résultat attendu:**
```
Cost: $0.XXXX
✓ Cost Tracker OK
```

---

## Tests des modules de base

### Test 3: Embedder (IMPORTANT)

```bash
python -c "
from src.vectorization.embedder import Embedder
from src.utils.config_loader import load_config
config = load_config()
embedder = Embedder(config)
vector = embedder.embed_text('Test de texte')
print(f'Embedding dimension: {len(vector)}')
print(f'Device: {embedder.device}')
print('✓ Embedder OK')
"
```

**Résultat attendu:**
```
Embedding dimension: 384
Device: cpu (ou mps/cuda si GPU disponible)
✓ Embedder OK
```

### Test 4: LaTeX Handler (CRITIQUE)

```bash
python -c "
from src.extraction.latex_handler import LatexHandler
handler = LatexHandler()

# Test détection formule simple
text = 'La formule est \$f(x) = x^2\$ et voici \$\$E = mc^2\$\$'
formulas = handler.detect_formulas(text)
print(f'Formulas detected: {len(formulas)}')
for f in formulas:
    print(f'  - {f.formula_type}: {f.content}')
print('✓ LaTeX Handler OK')
"
```

**Résultat attendu:**
```
Formulas detected: 2
  - inline_dollar: f(x) = x^2
  - display_dollar: E = mc^2
✓ LaTeX Handler OK
```

---

## Tests de vectorisation (CRITIQUE)

⚠️ **Cette section est CRITIQUE** - Les formules LaTeX ne doivent JAMAIS être coupées.

### Test 5: Chunker avec formules LaTeX

Créer un fichier de test `test_chunking.py`:

```python
from src.vectorization.chunker import Chunker
from src.utils.config_loader import load_config

config = load_config()
chunker = Chunker(config)

# Texte avec formules
text = """
Introduction à la dérivée.

La dérivée d'une fonction $f$ en un point $x$ est définie par:
$$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$

Cette limite représente le taux de variation instantané.

Pour une fonction polynomiale $f(x) = x^n$, on a:
$$f'(x) = nx^{n-1}$$

Exemple: Si $f(x) = x^3$, alors $f'(x) = 3x^2$.
"""

# Chunker
chunks = chunker.chunk_text(text)

print(f"✓ Generated {len(chunks)} chunks")
print()

# Vérifier formules
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(f"  Formulas: {chunk.num_formulas}")
    print(f"  Length: {chunk.char_count} chars")
    print(f"  Text preview: {chunk.text[:100]}...")
    print()

# VALIDATION CRITIQUE
formulas_in_original = ['f\'(x) = \\lim', 'f\'(x) = nx^{n-1}', 'f\'(x) = 3x^2']
for formula in formulas_in_original:
    found = any(formula in chunk.text for chunk in chunks)
    status = "✓" if found else "❌"
    print(f"{status} Formula '{formula}' is intact")

print("\n✓ Chunker test complete")
```

Exécuter:
```bash
python test_chunking.py
```

**Résultat attendu:**
- Plusieurs chunks générés
- Chaque formule complète dans UN SEUL chunk
- ✓ pour toutes les formules
- AUCUNE formule coupée

**Si une formule est coupée → BUG CRITIQUE à corriger immédiatement!**

### Test 6: PDF Extraction

Si vous avez un PDF de test:

```bash
python -c "
from src.extraction.pdf_processor import PDFProcessor
from src.utils.config_loader import load_config
from pathlib import Path

config = load_config()
processor = PDFProcessor(config)

# Remplacer par votre PDF
pdf_path = Path('data/raw/test.pdf')

if pdf_path.exists():
    doc = processor.process_pdf(pdf_path)
    print(f'✓ PDF processed')
    print(f'  Pages: {len(doc.pages)}')
    print(f'  Formulas: {len(doc.formulas)}')
    print(f'  Text length: {len(doc.full_text)} chars')
else:
    print('⚠️ No test PDF found at data/raw/test.pdf')
"
```

---

## Tests du workflow complet

### Test 7: Construction du vector store

**IMPORTANT:** Vous devez avoir des PDFs dans `data/raw/` avant cette étape.

```bash
# Option 1: Télécharger depuis Google Drive (si configuré)
python scripts/download_pdfs.py

# Option 2: Copier manuellement des PDFs dans data/raw/

# Construire le vector store
python scripts/build_vector_store.py
```

**Résultat attendu:**
```
═══════════════════════════════════════════════════════════════════════════════
  CONSTRUCTION DE LA BASE VECTORIELLE
═══════════════════════════════════════════════════════════════════════════════

Found X PDF files:
  - file1.pdf
  - file2.pdf

Step 1/3: Processing PDFs and chunking text...
Processing PDFs: 100%|██████████████████████████| X/X
✓ Generated XXX chunks

Step 2/3: Generating embeddings and building vector store...
Embedding batches: 100%|████████████████████████| XX/XX
✓ Generated XXX embeddings

Step 3/3: Saving vector store...
✓ Saved to: data/vector_store/default.index

═══════════════════════════════════════════════════════════════════════════════
  BUILD COMPLETE!
═══════════════════════════════════════════════════════════════════════════════
```

**Vérifier:**
- Fichiers créés: `data/vector_store/default.index` et `default.metadata`
- Pas d'erreur "Formula may be split" (sinon → bug critique)
- Nombre de vecteurs = nombre de chunks

### Test 8: Retrieval

```bash
python scripts/test_retrieval.py --query "Qu'est-ce qu'une dérivée ?"
```

**Résultat attendu:**
```
═══════════════════════════════════════════════════════════════════════════════
  TEST DE RETRIEVAL VECTORIEL
═══════════════════════════════════════════════════════════════════════════════

Loading configuration...
✓ Configuration loaded

Initializing embedder...
✓ Embedder ready (model: sentence-transformers/all-MiniLM-L6-v2)

Loading vector store...
✓ Vector store loaded
  Total vectors: XXX

═══════════════════════════════════════════════════════════════════════════════
Query 1/1: Qu'est-ce qu'une dérivée ?
═══════════════════════════════════════════════════════════════════════════════

Found 3 results:

────────────────────────────────────────────────────────────────────────────────
Rank #1 - Score: 0.XXXX
────────────────────────────────────────────────────────────────────────────────
Source: math_analysis.pdf
Chunk ID: chunk_XX
Formulas: X

Text:
[Texte avec définition de dérivée et formules LaTeX intactes]
```

**Vérifier:**
- Scores entre 0 et 1
- Résultats pertinents (score > 0.5)
- Sources identifiées
- **Formules LaTeX présentes et complètes**

### Test 9: Workflow LangGraph

```bash
python -c "
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config

config = load_config()
workflow = create_rag_workflow(config)

result = invoke_workflow(
    workflow=workflow,
    question='Qu\'est-ce qu\'une dérivée ?',
    student_level='L2'
)

print('✓ Workflow executed')
print(f'  Success: {result[\"success\"]}')
print(f'  Response length: {len(result[\"final_response\"])} chars')
print(f'  Sources cited: {len(result.get(\"sources_cited\", []))}')
print(f'  Confidence: {result.get(\"confidence_score\", 0):.2f}')
print()
print('Response preview:')
print(result['final_response'][:300])
"
```

**Résultat attendu:**
```
✓ Workflow executed
  Success: True
  Response length: XXX chars
  Sources cited: X
  Confidence: 0.XX

Response preview:
[Réponse pédagogique avec formules LaTeX et sources citées]
```

---

## Tests de l'interface

### Test 10: Lancer Streamlit

```bash
make run
# Ou directement:
streamlit run src/interface/app.py
```

**Naviguer vers:** http://localhost:8501

**Tester:**

1. **Interface se charge:**
   - ✓ Pas d'erreur Python
   - ✓ Page s'affiche correctement
   - ✓ Sidebar avec configuration visible

2. **Poser une question:**
   - Entrer: "Qu'est-ce qu'une dérivée ?"
   - ✓ Progress bar s'affiche
   - ✓ Étapes du workflow visibles
   - ✓ Réponse générée

3. **Vérifier la réponse:**
   - ✓ Texte cohérent et pédagogique
   - ✓ Formules LaTeX rendues (avec MathJax)
   - ✓ Sources affichées dans expander
   - ✓ Métadonnées (tokens, coût, temps)

4. **Tester différents niveaux:**
   - Changer niveau dans sidebar (L1, L2, L3, M1, M2)
   - Poser même question
   - ✓ Réponse adaptée au niveau

5. **Tester questions hors-sujet:**
   - Entrer: "Quelle est la capitale de la France ?"
   - ✓ Message "hors de mon domaine d'expertise"

6. **Métriques:**
   - Poser plusieurs questions
   - ✓ Compteur de questions s'incrémente
   - ✓ Coût total s'accumule
   - ✓ Bouton reset fonctionne

---

## Tests automatisés

### Test 11: Suite de tests complète

```bash
# Exécuter toutes les questions de test
python tests/run_test_questions.py
```

**Résultat attendu:**
```
═══════════════════════════════════════════════════════════════════════════════
  TEST QUESTIONS - VALIDATION DU SYSTÈME RAG
═══════════════════════════════════════════════════════════════════════════════

Total questions to test: 20

Initializing RAG system...
✓ System initialized

Running test questions...
Testing: 100%|████████████████████████████████████| 20/20

Generating report...
✓ Report saved to: data/logs/test_report.json

═══════════════════════════════════════════════════════════════════════════════
  TEST SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Total questions: 20
Passed: XX (XX%)
Failed: X

Check statistics:
  ✓ length: 20/20 (100%)
  ✓ sources_cited: 18/20 (90%)
  ✓ latex_preservation: 15/15 (100%)
  ✓ confidence: 17/20 (85%)

✓ Test suite passed!
```

**Critères de succès:**
- Pass rate >= 70%
- `latex_preservation`: 100% (CRITIQUE!)
- `sources_cited`: >= 80%
- `confidence`: >= 75%

**Si échec:**
- Consulter `data/logs/test_report.json` pour détails
- Identifier questions échouées
- Vérifier logs: `data/logs/app.log`

### Test 12: Tests par catégorie

```bash
# Tester seulement analyse
python tests/run_test_questions.py --category analyse --limit 5

# Tester seulement algèbre
python tests/run_test_questions.py --category algebre --limit 5
```

---

## Validation finale

### Checklist avant déploiement

- [ ] Configuration:
  - [ ] `.env` avec toutes les clés API
  - [ ] `config.yaml` correctement configuré
  - [ ] Logs fonctionnent

- [ ] Vectorisation:
  - [ ] Vector store construit avec succès
  - [ ] **AUCUNE formule LaTeX coupée** (CRITIQUE!)
  - [ ] Retrieval retourne résultats pertinents

- [ ] LLM:
  - [ ] Connexion API fonctionne
  - [ ] Génération produit réponses cohérentes
  - [ ] Sources citées correctement
  - [ ] Coûts trackés

- [ ] Workflow:
  - [ ] Classification fonctionne
  - [ ] Routing conditionnel correct
  - [ ] Vérification évalue confiance
  - [ ] Métriques collectées

- [ ] Interface:
  - [ ] Streamlit se lance sans erreur
  - [ ] Questions/réponses fonctionnent
  - [ ] Formulas LaTeX rendues (MathJax)
  - [ ] Métriques affichées
  - [ ] Performance acceptable (< 10s par question)

- [ ] Tests automatisés:
  - [ ] Suite de tests passe (>= 70%)
  - [ ] **LaTeX preservation à 100%** (CRITIQUE!)
  - [ ] Pas de régression

---

## Troubleshooting

### Problème: Vector store vide

**Symptômes:**
- `total_vectors: 0`
- Erreur "Vector store is empty"

**Solutions:**
1. Vérifier PDFs dans `data/raw/`
2. Re-exécuter: `python scripts/build_vector_store.py --rebuild`
3. Vérifier logs pour erreurs d'extraction

### Problème: Formules LaTeX coupées

**Symptômes:**
- Formules incomplètes dans réponses
- Warning "Formula may be split" dans logs

**Solutions:**
1. **C'est un BUG CRITIQUE!**
2. Vérifier `src/vectorization/chunker.py`
3. Augmenter `chunk_size` dans config
4. Vérifier `formula_boundary_protection: true` dans config
5. Reporter le bug avec exemple

### Problème: API rate limit

**Symptômes:**
- Erreur "Rate limit exceeded"
- HTTP 429

**Solutions:**
1. Attendre quelques secondes
2. Réduire `requests_per_minute` dans config
3. Vérifier quota API sur dashboard provider

### Problème: Pas de GPU détecté

**Symptômes:**
- Device: cpu (alors que GPU disponible)

**Solutions:**
1. Vérifier PyTorch installé avec support GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Installer version GPU:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Problème: Streamlit ne se lance pas

**Symptômes:**
- Erreur lors de `streamlit run`
- Port 8501 déjà utilisé

**Solutions:**
1. Changer port:
   ```bash
   streamlit run src/interface/app.py --server.port 8502
   ```
2. Tuer process existant:
   ```bash
   lsof -ti:8501 | xargs kill
   ```

### Problème: Réponses de mauvaise qualité

**Symptômes:**
- Hallucinations
- Pas de sources
- Réponses hors-sujet

**Solutions:**
1. Vérifier retrieval retourne docs pertinents
2. Augmenter `top_k` dans config
3. Ajuster `similarity_threshold`
4. Améliorer prompts dans `generator.py`
5. Vérifier PDFs sources sont de bonne qualité

---

## Prochaines étapes

Une fois tous les tests passés:

1. **Production:**
   - Configurer monitoring (Langfuse)
   - Mettre en place backups
   - Documenter procédures

2. **Amélioration continue:**
   - Collecter feedback utilisateurs
   - Analyser métriques
   - Ajouter nouveaux PDFs
   - Re-build vector store périodiquement

3. **Extensions:**
   - Human validation pour faible confiance
   - Multi-query pour meilleure recherche
   - Re-ranking avec cross-encoder
   - Export conversations (PDF, MD)

---

**Documentation complète:** Voir [README.md](README.md) pour architecture et usage détaillé.

**Support:** Consulter logs dans `data/logs/` pour debugging.
