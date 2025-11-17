# Guide: Obtenir credentials.json pour Google Drive

## Étapes pour obtenir credentials.json

### 1. Aller sur Google Cloud Console
Ouvrir: **https://console.cloud.google.com**

### 2. Créer/Sélectionner un projet
- Cliquer sur le sélecteur de projet (en haut)
- "Nouveau projet" ou sélectionner un existant
- Nom: "Math RAG System" (ou autre)
- Cliquer "Créer"

### 3. Activer Google Drive API
- Menu gauche → "APIs et services" → "Bibliothèque"
- Chercher: "Google Drive API"
- Cliquer sur "Google Drive API"
- Cliquer "ACTIVER"

### 4. Créer des identifiants OAuth 2.0
- Menu gauche → "APIs et services" → "Identifiants"
- Cliquer "+ CRÉER DES IDENTIFIANTS" (en haut)
- Sélectionner "ID client OAuth"

### 5. Configurer l'écran de consentement (si demandé)
- Type: "Externe"
- Nom de l'application: "Math RAG System"
- E-mail assistance utilisateur: votre email
- Domaine de l'application: laisser vide
- E-mail du développeur: votre email
- Cliquer "Enregistrer et continuer"
- Champs d'application: cliquer "Enregistrer et continuer" (pas besoin d'ajouter)
- Utilisateurs test: ajouter votre email
- Cliquer "Enregistrer et continuer"

### 6. Créer l'ID client OAuth
- Type d'application: **"Application de bureau"**
- Nom: "Math RAG Desktop"
- Cliquer "Créer"

### 7. Télécharger credentials.json
- Une popup apparaît avec votre ID client
- Cliquer sur l'icône **télécharger** (flèche vers le bas)
- Un fichier JSON est téléchargé

### 8. Renommer et déplacer le fichier
- Le fichier téléchargé s'appelle quelque chose comme:
  `client_secret_XXXXX.apps.googleusercontent.com.json`
- **Renommer en:** `credentials.json`
- **Déplacer dans:** la racine du projet `math-rag-system/`

### 9. Vérifier
```bash
ls -la credentials.json
```

Devrait afficher: `-rw-r--r-- ... credentials.json`

---

## ✅ Résumé rapide

1. https://console.cloud.google.com
2. Nouveau projet
3. Activer "Google Drive API"
4. Créer "ID client OAuth" → Type: "Application de bureau"
5. Télécharger le JSON
6. Renommer en `credentials.json`
7. Placer à la racine du projet

---

## 🔐 Sécurité

**⚠️ credentials.json contient des secrets!**
- Ne JAMAIS commit dans Git (déjà dans .gitignore)
- Ne JAMAIS partager publiquement
- C'est normal qu'il soit dans .gitignore
