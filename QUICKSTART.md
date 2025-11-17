# Démarrage Rapide - Système RAG Mathématique

Guide pour lancer le système en **5 minutes** ⏱️

---

## 🚀 Installation Express

### Étape 1: Installation des dépendances (2 min)

```bash
# Cloner/naviguer vers le projet
cd math-rag-system

# Installer dépendances
make install
# Ou: pip install -r requirements.txt
```

### Étape 2: Configuration (1 min)

```bash
# Copier template .env
cp .env.example .env

# Éditer .env et ajouter votre clé OpenAI
nano .env
```

**Minimum requis dans `.env`:**
```bash
OPENAI_API_KEY=sk-proj-...  # Votre clé OpenAI
```

### Étape 3: Setup initial (30 sec)

```bash
make setup
```

Cela crée les dossiers nécessaires:
- `data/raw/` - Pour vos PDFs
- `data/vector_store/` - Base vectorielle
- `data/logs/` - Logs système

---

## 📚 Ajouter vos PDFs

### Option A: Copie manuelle (recommandé pour démarrer)

```bash
# Copier vos PDFs de maths dans data/raw/
cp ~/Downloads/mon_cours_math.pdf data/raw/
```

### Option B: Google Drive (optionnel)

```bash
# Configurer Google Drive
python scripts/setup_gdrive.py

# Télécharger PDFs
python scripts/download_pdfs.py
```

---

## 🔧 Construction de la base vectorielle

```bash
# Construire l'index FAISS à partir des PDFs
make build-index
# Ou: python scripts/build_vector_store.py
```

**Attendez:** Environ 1-2 min pour quelques PDFs.

**Résultat:**
```
✓ Generated XXX chunks
✓ Generated XXX embeddings
✓ Vector store built with XXX vectors
```

---

## ✅ Test rapide

```bash
# Tester le retrieval
python scripts/test_retrieval.py --query "Qu'est-ce qu'une dérivée ?"
```

**Si vous voyez des résultats pertinents → Tout fonctionne! 🎉**

---

## 🌐 Lancer l'interface

```bash
make run
# Ou: streamlit run src/interface/app.py
```

**Ouvrir navigateur:** http://localhost:8501

**Tester:**
1. Poser une question: "Qu'est-ce qu'une intégrale ?"
2. Vérifier la réponse apparaît avec sources
3. Vérifier les formules LaTeX sont bien rendues

---

## 🎯 Exemples de questions

Essayez ces questions dans l'interface:

**Niveau L1:**
- "Qu'est-ce qu'une dérivée ?"
- "Comment calculer la dérivée de x^n ?"
- "Énoncé du théorème de Pythagore"

**Niveau L2:**
- "Qu'est-ce qu'une intégrale définie ?"
- "Formule de l'intégration par parties"
- "Qu'est-ce qu'un espace vectoriel ?"

**Niveau L3:**
- "Qu'est-ce qu'une série convergente ?"
- "Formule de Taylor"
- "Comment résoudre une équation différentielle ?"

---

## 📊 Vérifications importantes

### ✓ Les formules LaTeX sont-elles intactes ?

Dans l'interface, vérifier que les formules s'affichent correctement:
- `$f(x) = x^2$` → Formule inline
- `$$\int_a^b f(x)dx$$` → Formule display

**Si formules coupées ou mal affichées → Problème critique!** Voir [TESTING.md](TESTING.md).

### ✓ Les sources sont-elles citées ?

Chaque réponse doit avoir:
- Section "📚 Sources" avec noms des PDFs
- Format: `[Source: nom_fichier.pdf, page X]`

### ✓ Les coûts sont-ils trackés ?

Dans la sidebar:
- Compteur de questions
- Coût total en $
- Devrait s'incrémenter à chaque question

---

## 🔍 Troubleshooting Express

### Problème: "Vector store not found"

**Solution:**
```bash
# Re-construire
python scripts/build_vector_store.py --rebuild
```

### Problème: "OpenAI API error"

**Solutions:**
1. Vérifier clé dans `.env`
2. Vérifier solde sur compte OpenAI
3. Tester connexion:
   ```bash
   python -c "import openai; print('✓ OpenAI library OK')"
   ```

### Problème: Pas de résultats pertinents

**Solutions:**
1. Vérifier PDFs contiennent bien le sujet
2. Re-construire vector store
3. Augmenter `top_k` dans `config/config.yaml`

### Problème: Streamlit ne se lance pas

**Solution:**
```bash
# Port différent
streamlit run src/interface/app.py --server.port 8502
```

---

## 📖 Documentation complète

- **Architecture:** [README.md](README.md)
- **Tests détaillés:** [TESTING.md](TESTING.md)
- **Configuration:** [config/config.yaml](config/config.yaml)

---

## 🎓 Prochaines étapes

Une fois le système qui fonctionne:

1. **Ajouter plus de PDFs:**
   - Copier dans `data/raw/`
   - Re-exécuter: `make build-index`

2. **Personnaliser:**
   - Éditer prompts dans `src/agents/generator.py`
   - Ajuster paramètres dans `config/config.yaml`

3. **Tester en profondeur:**
   - Exécuter: `python tests/run_test_questions.py`
   - Consulter rapport: `data/logs/test_report.json`

4. **Monitoring:**
   - Configurer Langfuse pour tracking
   - Analyser logs: `data/logs/app.log`

---

## 💡 Conseils

**Performance:**
- GPU recommandé mais pas obligatoire
- Embeddings fonctionnent bien sur CPU
- Génération nécessite API (OpenAI/Anthropic)

**Coûts:**
- Embeddings: gratuits (local)
- Génération: ~$0.01-0.05 par question (GPT-4o)
- Surveillez la sidebar pour tracking

**Qualité:**
- PDFs de meilleure qualité → meilleures réponses
- PDFs structurés avec LaTeX → formules préservées
- Plus de documents → meilleure couverture

---

**Besoin d'aide?** Consultez [TESTING.md](TESTING.md) section Troubleshooting.

**Tout fonctionne?** 🎉 Vous êtes prêt à utiliser votre assistant mathématiques!
