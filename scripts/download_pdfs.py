"""
Script de téléchargement des PDFs depuis Google Drive.

Ce script télécharge tous les PDFs d'un dossier Google Drive
vers le dossier local data/raw/.

Pré-requis:
    - credentials.json (OAuth2 Google Drive API)
    - GDRIVE_FOLDER_ID dans .env

Usage:
    python scripts/download_pdfs.py
    python scripts/download_pdfs.py --folder-id YOUR_FOLDER_ID

Configuration:
    1. Activer Google Drive API: https://console.cloud.google.com
    2. Créer OAuth2 credentials
    3. Télécharger credentials.json
    4. Placer dans la racine du projet
    5. Définir GDRIVE_FOLDER_ID dans .env
"""

import sys
from pathlib import Path
import argparse
import os
from typing import List, Optional

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Imports Google Drive (optionnels)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import pickle
    import io
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logger.warning("Google Drive libraries not installed")


# Scopes pour Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_credentials() -> Optional[object]:
    """
    Obtient les credentials Google Drive (OAuth2).

    Workflow:
    1. Cherche token.pickle (credentials sauvegardées)
    2. Si existe et valide: utiliser
    3. Sinon: lancer OAuth flow avec credentials.json
    4. Sauvegarder token.pickle pour prochaine fois

    Returns:
        Credentials object ou None si échec
    """
    creds = None
    token_path = Path("token.pickle")
    credentials_path = Path("credentials.json")

    # 1. Charger token sauvegardé
    if token_path.exists():
        logger.info("Loading saved credentials from token.pickle")
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # 2. Vérifier validité
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials")
            creds.refresh(Request())
        else:
            # 3. OAuth flow
            if not credentials_path.exists():
                logger.error(
                    "credentials.json not found! "
                    "Please download from Google Cloud Console."
                )
                print("❌ credentials.json not found!")
                print("   1. Go to: https://console.cloud.google.com")
                print("   2. Enable Google Drive API")
                print("   3. Create OAuth2 credentials")
                print("   4. Download credentials.json")
                print("   5. Place in project root")
                return None

            logger.info("Starting OAuth flow")
            print("\n🔐 Google Drive Authentication Required")
            print("   A browser window will open for authentication...")
            print()

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        # 4. Sauvegarder token
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        logger.info("Credentials saved to token.pickle")

    return creds


def list_files_in_folder(service: object, folder_id: str) -> List[dict]:
    """
    Liste tous les fichiers PDF dans un dossier Google Drive.

    Args:
        service: Google Drive service
        folder_id: ID du dossier Google Drive

    Returns:
        Liste de fichiers avec metadata
    """
    logger.info(f"Listing files in folder: {folder_id}")

    try:
        # Query: tous les PDFs dans le dossier
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"

        results = service.files().list(
            q=query,
            fields="files(id, name, size, modifiedTime)",
            orderBy="name"
        ).execute()

        files = results.get('files', [])
        logger.info(f"Found {len(files)} PDF files")

        return files

    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise


def download_file(service: object, file_id: str, file_name: str, output_dir: Path) -> bool:
    """
    Télécharge un fichier depuis Google Drive.

    Args:
        service: Google Drive service
        file_id: ID du fichier
        file_name: Nom du fichier
        output_dir: Dossier de destination

    Returns:
        True si succès
    """
    output_path = output_dir / file_name

    # Skip si déjà téléchargé
    if output_path.exists():
        logger.info(f"  Skipping {file_name} (already exists)")
        return True

    try:
        logger.info(f"  Downloading {file_name}...")

        # Download
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"    Progress: {progress}%", end='\r')

        # Sauvegarder
        with open(output_path, 'wb') as f:
            f.write(fh.getvalue())

        print(f"    ✓ Downloaded: {file_name}")
        logger.info(f"  Saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"  Failed to download {file_name}: {e}")
        print(f"    ❌ Error: {e}")
        return False


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Download PDFs from Google Drive"
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        help="Google Drive folder ID (overrides .env)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )

    args = parser.parse_args()

    # Banner
    print("═" * 80)
    print("  TÉLÉCHARGEMENT DES PDFs DEPUIS GOOGLE DRIVE")
    print("═" * 80)
    print()

    # Vérifier dépendances
    if not GDRIVE_AVAILABLE:
        print("❌ Google Drive libraries not installed!")
        print("   Install with: pip install google-auth google-auth-oauthlib google-api-python-client")
        return

    try:
        # 1. Charger configuration
        config = load_config()

        # 2. Obtenir folder ID
        folder_id = args.folder_id or os.getenv("GDRIVE_FOLDER_ID")

        if not folder_id:
            logger.error("No Google Drive folder ID provided")
            print("❌ Google Drive folder ID not found!")
            print("   Option 1: Set GDRIVE_FOLDER_ID in .env")
            print("   Option 2: Use --folder-id argument")
            print()
            print("   To get folder ID:")
            print("   1. Open folder in Google Drive")
            print("   2. Copy ID from URL: drive.google.com/drive/folders/YOUR_FOLDER_ID")
            return

        print(f"Folder ID: {folder_id}")

        # 3. Créer dossier output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
        print()

        # 4. Authentifier
        print("Step 1/3: Authenticating with Google Drive...")
        creds = get_credentials()

        if not creds:
            print("❌ Authentication failed")
            return

        service = build('drive', 'v3', credentials=creds)
        print("✓ Authenticated")
        print()

        # 5. Lister fichiers
        print("Step 2/3: Listing PDF files in folder...")
        files = list_files_in_folder(service, folder_id)

        if not files:
            print("⚠️  No PDF files found in this folder")
            return

        print(f"✓ Found {len(files)} PDF files:")
        total_size = sum(int(f.get('size', 0)) for f in files)
        print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
        print()

        for f in files:
            size_mb = int(f.get('size', 0)) / 1024 / 1024
            print(f"  - {f['name']} ({size_mb:.2f} MB)")
        print()

        # 6. Télécharger
        print("Step 3/3: Downloading files...")
        print()

        success_count = 0
        for file in files:
            success = download_file(
                service,
                file['id'],
                file['name'],
                output_dir
            )
            if success:
                success_count += 1

        print()

        # 7. Résumé
        print("═" * 80)
        print("  DOWNLOAD COMPLETE!")
        print("═" * 80)
        print(f"Successfully downloaded: {success_count}/{len(files)} files")
        print(f"Location: {output_dir.absolute()}")
        print()
        print("Next steps:")
        print("  1. Build vector store: python scripts/build_vector_store.py")
        print("  2. Or use Makefile: make build-index")
        print()

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        logger.info("Download cancelled by user")

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        print(f"\n❌ Download failed: {e}")
        print("   Check logs for details: data/logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# GOOGLE DRIVE API:
# - Utilise OAuth2 pour authentification
# - Scopes: drive.readonly (lecture seule)
# - Token sauvegardé dans token.pickle (réutilisable)
#
# WORKFLOW:
# 1. Charger credentials (OAuth2 flow si première fois)
# 2. Lister fichiers dans folder (query: mimeType=application/pdf)
# 3. Pour chaque fichier:
#    - Skip si déjà téléchargé
#    - Download avec progress
#    - Sauvegarder dans data/raw/
#
# CREDENTIALS:
# credentials.json:
#   - Téléchargé depuis Google Cloud Console
#   - OAuth2 client ID + secret
#   - Utilisé pour OAuth flow
#
# token.pickle:
#   - Généré après premier OAuth flow
#   - Contient access token + refresh token
#   - Réutilisé pour éviter re-auth
#   - Expiration: auto-refresh
#
# DÉPENDANCES:
# pip install google-auth google-auth-oauthlib google-api-python-client
#
# SETUP:
# 1. Google Cloud Console: https://console.cloud.google.com
# 2. Créer projet
# 3. Activer Google Drive API
# 4. Créer OAuth2 credentials (Desktop app)
# 5. Télécharger credentials.json
# 6. Placer dans racine projet
# 7. Première exécution: OAuth flow dans browser
# 8. token.pickle sauvegardé automatiquement
#
# USAGE:
# ```bash
# # Normal (utilise GDRIVE_FOLDER_ID de .env)
# python scripts/download_pdfs.py
#
# # Folder ID spécifique
# python scripts/download_pdfs.py --folder-id 1V0C3tUWpIHGFhcJ1sNhhubJlSw_IkMej
#
# # Custom output directory
# python scripts/download_pdfs.py --output-dir /path/to/output
# ```
#
# FOLDER ID:
# - Obtenu depuis URL Google Drive
# - Format: drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE
# - Exemple: 1V0C3tUWpIHGFhcJ1sNhhubJlSw_IkMej
#
# ERROR HANDLING:
# - Credentials manquants: message clair avec instructions
# - Folder vide: warning mais pas erreur
# - Download échoue: log error, continue avec suivant
# - Skip fichiers déjà téléchargés
#
# SÉCURITÉ:
# - credentials.json contient secrets → .gitignore
# - token.pickle contient tokens → .gitignore
# - Scopes minimal (readonly)
#
# ═══════════════════════════════════════════════════════════════════════════════
