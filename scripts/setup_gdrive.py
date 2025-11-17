"""
Script de configuration Google Drive.

Guide interactif pour configurer l'accès à Google Drive:
1. Vérifie si credentials.json existe
2. Teste la connexion
3. Affiche les informations du dossier
4. Sauvegarde la configuration dans .env

Usage:
    python scripts/setup_gdrive.py
"""

import sys
from pathlib import Path
import os

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Imports Google Drive (optionnels)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def print_banner():
    """Affiche le banner."""
    print()
    print("═" * 80)
    print("  CONFIGURATION GOOGLE DRIVE")
    print("═" * 80)
    print()


def check_credentials_file() -> bool:
    """Vérifie si credentials.json existe."""
    credentials_path = Path("credentials.json")

    if credentials_path.exists():
        print("✓ credentials.json found")
        return True
    else:
        print("❌ credentials.json not found")
        print()
        print("To get credentials.json:")
        print("  1. Go to: https://console.cloud.google.com")
        print("  2. Create a new project (or select existing)")
        print("  3. Enable Google Drive API:")
        print("     - APIs & Services → Enable APIs and Services")
        print("     - Search: Google Drive API")
        print("     - Click Enable")
        print("  4. Create OAuth2 credentials:")
        print("     - APIs & Services → Credentials")
        print("     - Create Credentials → OAuth client ID")
        print("     - Application type: Desktop app")
        print("     - Name: RAG System (or anything)")
        print("     - Click Create")
        print("  5. Download credentials.json:")
        print("     - Click download icon next to your OAuth client")
        print("     - Rename to: credentials.json")
        print("     - Move to project root")
        print()
        return False


def authenticate() -> object:
    """Lance le flow OAuth et retourne le service."""
    creds = None
    token_path = Path("token.pickle")
    credentials_path = Path("credentials.json")

    # Charger token existant
    if token_path.exists():
        logger.info("Loading existing token")
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # Vérifier validité
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing token")
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            # OAuth flow
            logger.info("Starting OAuth flow")
            print()
            print("🔐 Starting Google Drive authentication...")
            print("   A browser window will open.")
            print("   Please sign in with your Google account.")
            print()
            input("Press Enter to continue...")
            print()

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                SCOPES
            )
            creds = flow.run_local_server(port=0)

            print()
            print("✓ Authentication successful!")

        # Sauvegarder token
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        logger.info("Token saved")
        print("✓ Credentials saved to token.pickle")

    # Build service
    service = build('drive', 'v3', credentials=creds)
    return service


def test_connection(service: object) -> bool:
    """Teste la connexion en listant quelques fichiers."""
    try:
        print()
        print("Testing connection...")

        # Lister 5 fichiers (n'importe lesquels)
        results = service.files().list(
            pageSize=5,
            fields="files(id, name)"
        ).execute()

        files = results.get('files', [])
        print(f"✓ Connection successful! Found {len(files)} test files")
        return True

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        print(f"❌ Connection test failed: {e}")
        return False


def get_folder_info(service: object, folder_id: str) -> dict:
    """Récupère les informations d'un dossier."""
    try:
        # Metadata du dossier
        folder = service.files().get(
            fileId=folder_id,
            fields="id, name, mimeType"
        ).execute()

        # Compter fichiers PDFs
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, size)"
        ).execute()

        files = results.get('files', [])
        total_size = sum(int(f.get('size', 0)) for f in files)

        return {
            "name": folder.get('name', 'Unknown'),
            "id": folder_id,
            "pdf_count": len(files),
            "total_size_mb": total_size / 1024 / 1024
        }

    except Exception as e:
        logger.error(f"Failed to get folder info: {e}")
        return None


def update_env_file(folder_id: str) -> bool:
    """Met à jour le fichier .env avec le folder ID."""
    env_path = Path(".env")

    try:
        # Lire .env existant
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []

        # Chercher ligne GDRIVE_FOLDER_ID
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("GDRIVE_FOLDER_ID="):
                lines[i] = f"GDRIVE_FOLDER_ID={folder_id}\n"
                updated = True
                break

        # Ajouter si pas trouvé
        if not updated:
            lines.append(f"\nGDRIVE_FOLDER_ID={folder_id}\n")

        # Écrire
        with open(env_path, 'w') as f:
            f.writelines(lines)

        print(f"✓ Updated .env with folder ID: {folder_id}")
        logger.info(f"Updated .env with GDRIVE_FOLDER_ID={folder_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to update .env: {e}")
        print(f"❌ Failed to update .env: {e}")
        return False


def main():
    """Point d'entrée principal."""
    print_banner()

    # Vérifier dépendances
    if not GDRIVE_AVAILABLE:
        print("❌ Google Drive libraries not installed!")
        print()
        print("Install with:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        print()
        return

    # 1. Vérifier credentials.json
    print("Step 1/4: Checking credentials.json...")
    if not check_credentials_file():
        print()
        print("❌ Setup cannot continue without credentials.json")
        print("   Please follow the instructions above to obtain it.")
        return

    print()

    # 2. Authentifier
    print("Step 2/4: Authenticating with Google Drive...")
    try:
        service = authenticate()
        print()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        print(f"❌ Authentication failed: {e}")
        return

    # 3. Tester connexion
    print("Step 3/4: Testing connection...")
    if not test_connection(service):
        print("❌ Connection test failed")
        return

    print()

    # 4. Demander folder ID
    print("Step 4/4: Configure folder ID")
    print()
    print("To get your Google Drive folder ID:")
    print("  1. Open your folder in Google Drive")
    print("  2. Copy the ID from the URL:")
    print("     https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE")
    print()

    # Vérifier si déjà dans .env
    current_folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if current_folder_id:
        print(f"Current folder ID in .env: {current_folder_id}")
        print()

        # Tester folder actuel
        print(f"Checking folder: {current_folder_id}...")
        folder_info = get_folder_info(service, current_folder_id)

        if folder_info:
            print()
            print("✓ Folder found!")
            print(f"  Name: {folder_info['name']}")
            print(f"  PDF files: {folder_info['pdf_count']}")
            print(f"  Total size: {folder_info['total_size_mb']:.2f} MB")
            print()

            response = input("Use this folder? (Y/n): ").strip().lower()
            if response != 'n':
                folder_id = current_folder_id
            else:
                folder_id = input("Enter new folder ID: ").strip()
        else:
            print("⚠️  Folder not found or not accessible")
            folder_id = input("Enter folder ID: ").strip()
    else:
        folder_id = input("Enter folder ID: ").strip()

    # Valider folder ID
    print()
    print(f"Validating folder: {folder_id}...")
    folder_info = get_folder_info(service, folder_id)

    if not folder_info:
        print("❌ Folder not found or not accessible")
        print("   Please check:")
        print("   - Folder ID is correct")
        print("   - Folder is shared with your Google account")
        print("   - Folder contains PDF files")
        return

    print()
    print("✓ Folder validated!")
    print(f"  Name: {folder_info['name']}")
    print(f"  PDF files: {folder_info['pdf_count']}")
    print(f"  Total size: {folder_info['total_size_mb']:.2f} MB")
    print()

    if folder_info['pdf_count'] == 0:
        print("⚠️  Warning: No PDF files found in this folder")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return

    # Sauvegarder dans .env
    print()
    update_env_file(folder_id)

    # Résumé
    print()
    print("═" * 80)
    print("  SETUP COMPLETE!")
    print("═" * 80)
    print()
    print("Configuration summary:")
    print(f"  ✓ credentials.json: Found")
    print(f"  ✓ token.pickle: Saved")
    print(f"  ✓ Folder ID: {folder_id}")
    print(f"  ✓ Folder name: {folder_info['name']}")
    print(f"  ✓ PDF files: {folder_info['pdf_count']}")
    print()
    print("Next steps:")
    print("  1. Download PDFs: python scripts/download_pdfs.py")
    print("  2. Build vector store: python scripts/build_vector_store.py")
    print("  3. Launch interface: make run")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# OBJECTIF:
# Guide interactif pour configurer Google Drive en quelques étapes simples.
# Vérifie chaque étape et donne feedback clair.
#
# WORKFLOW:
# 1. Vérifier credentials.json existe
#    - Si non: afficher instructions détaillées
# 2. Lancer OAuth flow
#    - Browser window s'ouvre
#    - User se connecte avec Google
#    - Token sauvegardé dans token.pickle
# 3. Tester connexion
#    - Liste 5 fichiers pour vérifier API fonctionne
# 4. Demander folder ID
#    - Check si déjà dans .env
#    - Si oui: afficher info et demander confirmation
#    - Si non: demander saisie
# 5. Valider folder
#    - Récupérer metadata
#    - Compter PDFs
#    - Afficher summary
# 6. Sauvegarder dans .env
#    - Ajouter ou update GDRIVE_FOLDER_ID
#
# INSTRUCTIONS CREDENTIALS.JSON:
# - Très détaillées pour guider utilisateur
# - Inclut tous les steps avec screenshots possibles
# - Liens directs vers Console
#
# ERROR HANDLING:
# - Chaque step peut échouer → message clair
# - Instructions pour corriger
# - Pas de crash brutal
#
# UX:
# - Banner clair
# - Steps numérotés
# - Checkmarks ✓ pour succès
# - ❌ pour erreurs
# - Instructions claires
#
# OUTPUTS:
# - token.pickle (credentials)
# - .env updated (GDRIVE_FOLDER_ID)
#
# TESTING:
# - Test connection avec API call simple
# - Test folder access
# - Count PDFs pour vérifier données disponibles
#
# ═══════════════════════════════════════════════════════════════════════════════
