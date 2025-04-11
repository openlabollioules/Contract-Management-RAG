import importlib
import logging
import os
import platform
import time

import torch
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PyPDF2 import PdfReader, PdfWriter
import fitz  # PyMuPDF

# Détection de l'architecture
is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
if is_apple_silicon:
    print("🍎 Détection d'un processeur Apple Silicon")
    if torch.backends.mps.is_available():
        print("🎮 GPU MPS disponible")
        device = torch.device("mps")
    else:
        print("⚠️ GPU MPS non disponible, utilisation du CPU")
        device = torch.device("cpu")
else:
    print("💻 Architecture non Apple Silicon")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Utilisation du device: {device}")

# Désactiver les logs de PostHog
logging.getLogger("posthog").setLevel(logging.CRITICAL)
logging.getLogger("backoff").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Désactiver complètement la télémétrie et les téléchargements
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["DISABLE_TELEMETRY"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["S3_OFFLINE"] = "1"  # Désactiver les téléchargements S3
os.environ["NO_PROXY"] = "*"  # Désactiver les proxies
os.environ["http_proxy"] = ""  # Désactiver les proxies HTTP
os.environ["https_proxy"] = ""  # Désactiver les proxies HTTPS


# Patch pour désactiver PostHog
def patch_posthog():
    try:
        import posthog

        posthog.capture = lambda *args, **kwargs: None
        posthog.identify = lambda *args, **kwargs: None
        posthog.group = lambda *args, **kwargs: None
        posthog.alias = lambda *args, **kwargs: None
        posthog.screen = lambda *args, **kwargs: None
        posthog.flush = lambda *args, **kwargs: None
        posthog.shutdown = lambda *args, **kwargs: None
    except ImportError:
        pass


# Patch pour désactiver les téléchargements S3
def patch_s3_download():
    try:
        import boto3

        original_download_file = boto3.s3.transfer.S3Transfer.download_file

        def patched_download_file(self, *args, **kwargs):
            print("S3 download blocked")
            return None

        boto3.s3.transfer.S3Transfer.download_file = patched_download_file
    except ImportError:
        pass


# Patch pour la fonction create_model_dict
def patch_create_model_dict():
    # Importer le module marker.models
    marker_models = importlib.import_module("marker.models")

    # Sauvegarder la fonction originale
    original_create_model_dict = marker_models.create_model_dict

    def patched_create_model_dict():
        print("Using patched create_model_dict")
        model_dict = {}

        # Charger les modèles locaux
        model_paths = {
            "layout_model": "offline_models/marker/layout/model.safetensors",
            "texify_model": "offline_models/marker/texify/model.safetensors",
            "recognition_model": "offline_models/marker/recognition_model/model.pkl",
            "table_rec_model": "offline_models/marker/table_rec_model/model.pkl",
            "detection_model": "offline_models/marker/detection_model/model.pkl",
            "inline_detection_model": "offline_models/marker/inline_detection_model/model.pkl",
        }

        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    if model_path.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        model_dict[model_name] = load_file(model_path)
                    else:
                        model_dict[model_name] = torch.load(model_path)
                    print(f"Loaded local model for {model_name}")
                else:
                    print(f"Local model not found for {model_name}")
            except Exception as e:
                print(f"Warning: Could not load local model {model_name}: {str(e)}")

        return model_dict

    # Remplacer la fonction originale
    marker_models.create_model_dict = patched_create_model_dict


# Patch pour désactiver les téléchargements S3 dans marker.models
def patch_marker_models():
    try:
        marker_models = importlib.import_module("marker.models")

        # Patcher la fonction qui charge les modèles depuis S3
        if hasattr(marker_models, "load_model_from_s3"):
            original_load_model = marker_models.load_model_from_s3

            def patched_load_model(*args, **kwargs):
                print("S3 model loading blocked")
                return None

            marker_models.load_model_from_s3 = patched_load_model

        # Patcher la fonction qui vérifie les modèles S3
        if hasattr(marker_models, "check_s3_model"):
            original_check_model = marker_models.check_s3_model

            def patched_check_model(*args, **kwargs):
                print("S3 model checking blocked")
                return False

            marker_models.check_s3_model = patched_check_model
    except ImportError:
        pass


# Appliquer les patches
patch_posthog()
patch_s3_download()
patch_marker_models()
patch_create_model_dict()


def correct_pdf_orientation(pdf_path):
    """
    Corrige l'orientation des pages PDF qui sont dans le mauvais sens.
    Utilise PyMuPDF pour détecter l'orientation et PyPDF2 pour appliquer la correction.
    """
    try:
        # Ouvrir le PDF avec PyMuPDF pour analyser l'orientation
        doc = fitz.open(pdf_path)
        writer = PdfWriter()
        
        # Lire le PDF avec PyPDF2
        reader = PdfReader(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Obtenir l'orientation de la page
            rotation = page.rotation
            
            # Ajouter la page au writer
            writer.add_page(reader.pages[page_num])
            
            # Corriger l'orientation selon la rotation détectée
            if rotation == 90:
                writer.pages[page_num].rotate(-90)  # Rotation vers la droite
            elif rotation == 180:
                writer.pages[page_num].rotate(-180)  # Retourner la page
            elif rotation == 270:
                writer.pages[page_num].rotate(-270)  # Rotation vers la gauche
            
            print(f"Page {page_num + 1}: rotation détectée = {rotation}°")
        
        # Sauvegarder le PDF corrigé
        output_path = pdf_path.replace('.pdf', '_oriented.pdf')
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✅ PDF corrigé sauvegardé sous: {output_path}")
        print(f"📄 Nombre de pages traitées: {len(doc)}")
        
        return output_path
        
    except Exception as e:
        print(f"Erreur lors de la correction de l'orientation: {str(e)}")
        return pdf_path  # Retourner le chemin original en cas d'erreur


def extract_text_contract(pdf_path):
    print("📄 Chargement des modèles...")
    start_time = time.time()

    # Corriger l'orientation du PDF si nécessaire
    print("🔄 Vérification de l'orientation des pages...")
    pdf_path = correct_pdf_orientation(pdf_path)

    # Configure Ollama service (local mode)
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

    # Setup model paths
    model_paths = {
        "layout": "offline_models/marker/layout",
        "texify": "offline_models/marker/texify",
        "text_recognition": "offline_models/marker/text_recognition",
        "table_recognition": "offline_models/marker/table_recognition",
        "text_detection": "offline_models/marker/text_detection",
        "inline_math_detection": "offline_models/marker/inline_math_detection",
    }

    print("🔍 Configuration de Marker...")
    # Setup the configuration for Marker with enhanced options
    config = {
        "output_format": "markdown",
        "use_llm": True,
        "llm_service": "marker.services.ollama.OllamaService",
        "ollama_model": "mistral-small3.1:latest",
        "ocr_engine": "tesseract",
        "ocr_language": "fra+eng",
        "table_structure": True,
        "preserve_layout": True,
        "extract_images": True,
        "clean_text": True,
        "remove_headers_footers": True,
        "detect_columns": True,
        "model_paths": model_paths,
        "max_workers": 1,
        "batch_size": 1,
        "disable_telemetry": True,
        "offline_mode": True,
        "force_offline": True,
        "skip_download": True,
        "s3_offline": True,
        "no_proxy": True,
        "device": str(device),
    }

    try:
        print("🔄 Création des modèles...")
        model_dict = create_model_dict()

        # Déplacer les modèles sur le device approprié
        for model_name, model in model_dict.items():
            if isinstance(model, torch.nn.Module):
                model.to(device)
                print(
                    f"Loaded {model_name} on device {device} with dtype {model.dtype}"
                )

        config_parser = ConfigParser(config)

        print("📝 Conversion du PDF...")
        # Convert PDF with Marker
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        # Process the PDF and extract text
        rendered = converter(pdf_path)
        text, metadata, _ = text_from_rendered(rendered)

        # Utiliser le nom complet du fichier (avec extension) comme titre du document
        document_title = os.path.basename(pdf_path)

        # S'assurer que metadata est un dictionnaire
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}

        # Add metadata to the text in a safe way
        text_with_metadata = f"""
Document Metadata:
- Title: {document_title}
- Author: {metadata.get('author', 'Unknown')}
- Pages: {metadata.get('pages', 'Unknown')}

Content:
{text}
"""

        print(f"✅ PDF traité en {time.time() - start_time:.2f} secondes")
        print(f"📊 Métriques:")
        print(f"  - Pages: {metadata.get('pages', 'Unknown')}")
        print(f"  - Mots: {len(text.split())}")
        print(f"  - Device utilisé: {device}")
        print(
            f"  - Vitesse: {len(text.split())/(time.time() - start_time):.2f} mots/seconde"
        )

        return text_with_metadata, document_title

    except Exception as e:
        print(f"❌ Erreur lors du traitement avancé: {str(e)}")
        # En cas d'erreur, essayer une approche plus simple
        try:
            print("🔄 Tentative de traitement basique...")
            from pdfminer.high_level import extract_text

            simple_text = extract_text(pdf_path)
            print("✅ Texte extrait avec succès (mode basique)")
            # Utiliser le nom complet du fichier comme titre en cas d'erreur
            document_title = os.path.basename(pdf_path)
            return (
                f"Error with advanced processing. Using basic text extraction:\n\n{simple_text}",
                document_title,
            )
        except Exception as e2:
            print(f"❌ Échec du traitement du PDF: {str(e2)}")
            return f"Failed to process PDF: {str(e2)}", None
