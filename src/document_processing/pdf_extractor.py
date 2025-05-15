import importlib
import logging
import os
import platform
import re
import time
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
import pytesseract
import torch
from dotenv import load_dotenv
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
from pytesseract import Output
from transformers import (SegformerForSemanticSegmentation,
                          SegformerImageProcessor)

from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)

# Load environment variables
load_dotenv("config.env")

# Détection de l'architecture
is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
use_mps = os.getenv("USE_MPS", "true").lower() == "true"

if is_apple_silicon and torch.backends.mps.is_available() and use_mps:
    logger.info("🍎 Détection d'un processeur Apple Silicon avec MPS disponible")
    device = torch.device("mps")
    logger.info("🎮 GPU MPS activé pour les modèles Marker")
elif torch.cuda.is_available():
    logger.info("🚀 GPU CUDA disponible")
    device = torch.device("cuda")
else:
    logger.info("💻 Utilisation du CPU (pas de GPU disponible)")
    device = torch.device("cpu")

logger.info(f"⚙️ Utilisation du device: {device}")

# Désactiver les logs de PostHog et autres télémetries
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

# Configurer les chemins de modèles
MARKER_DIR = os.getenv("MARKER_DIR", "offline_models/marker")


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
        logger.info(
            "🔧 Utilisation de patched_create_model_dict pour charger les modèles locaux"
        )
        model_dict = {}

        # Obtenir le chemin du dossier marker depuis config.env
        marker_dir = os.getenv("MARKER_DIR", "offline_models/marker")
        logger.info(f"📂 Utilisation du dossier de modèles Marker: {marker_dir}")

        # Modèles Marker avec les noms de répertoires corrects selon Hugging Face
        # Les modèles principaux de Marker sont maintenant dans VikParuchuri/marker_layout_segmenter et VikParuchuri/marker_texify
        model_paths = {
            # Modèles principaux avec les noms de répertoires corrects
            "layout": f"{marker_dir}/layout",  # VikParuchuri/marker_layout_segmenter
            "texify": f"{marker_dir}/texify",  # VikParuchuri/marker_texify
            # Les autres modèles (reconnaissance) sont intégrés au package Marker/Surya
            "text_recognition": f"{marker_dir}/text_recognition",
            "table_recognition": f"{marker_dir}/table_recognition",
            "text_detection": f"{marker_dir}/text_detection",
            "inline_math_detection": f"{marker_dir}/inline_math_detection",
            # Noms alternatifs de modèles (utilisés dans certaines parties du code)
            "layout_model": f"{marker_dir}/layout",
            "texify_model": f"{marker_dir}/texify",
            "recognition_model": f"{marker_dir}/text_recognition",
            "table_rec_model": f"{marker_dir}/table_recognition",
            "detection_model": f"{marker_dir}/text_detection",
            "inline_detection_model": f"{marker_dir}/inline_math_detection",
            "ocr_error_model": f"{marker_dir}/ocr_error",
        }

        # Extensions de fichiers possibles pour les modèles
        model_extensions = [
            "/model.safetensors",
            "/model.pkl",
            "/model.bin",
            "/pytorch_model.bin",
            "/config.json",  # Certains modèles ont besoin de config.json
        ]

        # Charger chaque modèle avec gestion d'erreurs approfondie
        for model_name, model_base_path in model_paths.items():
            if model_name in model_dict:
                logger.debug(f"✅ Modèle {model_name} déjà chargé, ignoré")
                continue

            logger.debug(
                f"🔍 Tentative de chargement du modèle {model_name} depuis {model_base_path}"
            )

            # Essayer chaque extension possible
            loaded = False
            for ext in model_extensions:
                model_path = f"{model_base_path}{ext}"

                if os.path.exists(model_path):
                    try:
                        if ext == "/model.safetensors":
                            from safetensors.torch import load_file

                            model_dict[model_name] = load_file(model_path)
                        elif ext == "/config.json":
                            # Ne pas charger config.json comme un modèle, mais juste vérifier s'il existe
                            continue
                        else:
                            model_dict[model_name] = torch.load(
                                model_path, map_location=device
                            )

                        logger.info(
                            f"✅ Modèle {model_name} chargé avec succès depuis {model_path}"
                        )
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Échec du chargement de {model_path}: {str(e)}"
                        )

            # Si le modèle n'a pas été chargé, essayer des chemins alternatifs
            if not loaded:
                # Essayer le sous-dossier avec le nom du modèle
                alt_base_path = f"{marker_dir}/{model_name.replace('_model', '')}"

                for ext in model_extensions:
                    alt_path = f"{alt_base_path}{ext}"

                    if os.path.exists(alt_path):
                        try:
                            if ext == "/model.safetensors":
                                from safetensors.torch import load_file

                                model_dict[model_name] = load_file(alt_path)
                            elif ext == "/config.json":
                                continue
                            else:
                                model_dict[model_name] = torch.load(
                                    alt_path, map_location=device
                                )

                            logger.info(
                                f"✅ Modèle {model_name} chargé depuis chemin alternatif {alt_path}"
                            )
                            loaded = True
                            break
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Échec du chargement alternatif {alt_path}: {str(e)}"
                            )

            # Rapporter si le modèle n'a pas pu être chargé
            if not loaded:
                logger.warning(
                    f"⚠️ Impossible de charger le modèle {model_name}. Il sera téléchargé automatiquement si nécessaire."
                )

        # Remplacer la fonction originale
        marker_models.create_model_dict = patched_create_model_dict
        logger.info("✅ Patch create_model_dict appliqué avec succès")

        return model_dict


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
        output_path_str = str(pdf_path).replace(".pdf", "_oriented.pdf")
        output_path = Path(output_path_str)
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        print(f"✅ PDF corrigé sauvegardé sous: {output_path}")
        print(f"📄 Nombre de pages traitées: {len(doc)}")

        return output_path

    except Exception as e:
        print(f"Erreur lors de la correction de l'orientation: {str(e)}")
        return pdf_path  # Retourner le chemin original en cas d'erreur


def get_text_regions(image):
    """
    Détecte les régions contenant du texte avec Tesseract OCR.
    """
    # Obtenir les données de détection de texte
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Créer un masque pour les zones de texte
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    n_boxes = len(data["level"])
    for i in range(n_boxes):
        if int(data["conf"][i]) > 60:  # Seuil de confiance pour le texte
            (x, y, w, h) = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            # Ajouter un petit padding autour du texte
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def detect_and_mask_signatures(image, sess, input_name, text_mask, padding=200):
    """
    Détecte et masque les signatures dans une image en évitant les zones de texte.
    Utilise une approche plus agressive avec post-traitement.
    """
    # Redimensionner l'image pour la détection
    h, w = image.shape[:2]
    small = cv2.resize(image, (640, 640))

    # Préparer l'image pour le modèle
    tensor = small[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # Détecter les signatures
    outputs = sess.run(None, {input_name: tensor})
    boxes = outputs[0]

    # Créer une copie de l'image pour le masquage
    masked_image = image.copy()
    signature_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Première passe : détection des signatures
    for box in boxes:
        x1, y1, x2, y2, score = box[:5]
        if score > 0.0001:  # Seuil extrêmement bas
            # Convertir les coordonnées à la taille originale
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)

            # Ajouter le padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # Marquer la zone dans le masque
            cv2.rectangle(signature_mask, (x1, y1), (x2, y2), 255, -1)

    # Post-traitement : expansion des zones détectées
    kernel = np.ones((50, 50), np.uint8)
    signature_mask = cv2.dilate(signature_mask, kernel, iterations=2)

    # Deuxième passe : masquage final en évitant le texte
    for y in range(h):
        for x in range(w):
            if signature_mask[y, x] > 0 and text_mask[y, x] == 0:
                masked_image[y, x] = [255, 255, 255]

    return masked_image


def clean_pdf(pdf_path):
    """
    Nettoie le PDF en masquant les signatures et tampons tout en protégeant le texte.
    Utilise une approche plus agressive avec post-traitement.
    """
    try:
        # Charger le modèle ONNX pour les signatures
        sig_model_path = "offline_models/handwritten-detector-onnx/model_clean.onnx"
        sess = ort.InferenceSession(sig_model_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        # Charger le modèle SegFormer pour les tampons
        processor = SegformerImageProcessor.from_pretrained(
            "offline_models/segformer-stamp", local_files_only=True
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "offline_models/segformer-stamp", local_files_only=True
        )

        # Convertir le PDF en images
        images = convert_from_path(pdf_path, dpi=400)
        cleaned_images = []

        for i, image in enumerate(images):
            print(f"Traitement de la page {i+1}/{len(images)}...")

            # Convertir en numpy array
            img_np = np.array(image)

            # Détecter les zones de texte
            print("Détection des zones de texte...")
            text_mask = get_text_regions(img_np)

            # Masquer les signatures en évitant le texte
            print("Masquage des signatures...")
            img_np = detect_and_mask_signatures(
                img_np, sess, input_name, text_mask, padding=200
            )

            # Masquer les tampons en évitant le texte
            print("Masquage des tampons...")
            inputs = processor(images=img_np, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            stamp_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

            # Post-traitement du masque des tampons
            kernel = np.ones((30, 30), np.uint8)
            stamp_mask = cv2.dilate(stamp_mask.astype(np.uint8), kernel, iterations=2)

            # Ne masquer que les zones de tampon qui ne contiennent pas de texte
            combined_mask = np.logical_and(stamp_mask == 1, text_mask == 0)
            img_np[combined_mask] = [255, 255, 255]

            cleaned_images.append(img_np)
            print(f"Page {i+1} traitée.")

        # Sauvegarder le PDF nettoyé
        cleaned_path = pdf_path.replace(".pdf", "_cleaned.pdf")
        cleaned_images[0].save(
            cleaned_path, save_all=True, append_images=cleaned_images[1:]
        )

        print(f"✅ PDF nettoyé sauvegardé sous: {cleaned_path}")
        return cleaned_path

    except Exception as e:
        print(f"Erreur lors du nettoyage du PDF: {str(e)}")
        return pdf_path


def remove_headers_footers_by_similarity(
    text, similarity_threshold=0.8, occurrence_threshold=3
):
    """
    Detects and removes headers and footers from text based on line similarity.
    Also removes image references in Markdown format.

    Args:
        text (str): The text to process
        similarity_threshold (float): Threshold for considering lines as similar (0-1)
        occurrence_threshold (int): Minimum occurrences to consider a line as header/footer

    Returns:
        str: Text with headers, footers, and image references removed
    """
    # First, remove image references (Markdown format)
    image_pattern = r"!\[\]\(.*?\.(jpeg|jpg|png|gif)\)"
    text = re.sub(image_pattern, "", text)

    # Split text into lines
    lines = text.split("\n")
    if not lines:
        return text

    # Count occurrences of similar lines
    line_occurrences = {}

    # Function to calculate similarity between two strings
    def similarity(s1, s2):
        # Skip empty lines
        if not s1.strip() or not s2.strip():
            return 0

        # For very short lines, use exact matching
        if len(s1) < 10 or len(s2) < 10:
            return 1.0 if s1 == s2 else 0.0

        # Simple Jaccard similarity for longer lines
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())

        if not s1_words or not s2_words:
            return 0

        intersection = len(s1_words.intersection(s2_words))
        union = len(s1_words.union(s2_words))

        return intersection / union if union > 0 else 0

    # Group similar lines
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Skip lines that are likely part of tables or structured content
        if line.startswith("|") and line.endswith("|"):
            continue

        found_similar = False
        for key in line_occurrences:
            if similarity(key, line) >= similarity_threshold:
                line_occurrences[key].append(i)
                found_similar = True
                break

        if not found_similar:
            line_occurrences[line] = [i]

    # Identify potential headers/footers (lines that appear multiple times)
    potential_headers_footers = set()
    for line, indices in line_occurrences.items():
        if len(indices) >= occurrence_threshold:
            potential_headers_footers.update(indices)

    # Remove headers/footers
    cleaned_lines = [
        line for i, line in enumerate(lines) if i not in potential_headers_footers
    ]

    # Join lines back into text
    cleaned_text = "\n".join(cleaned_lines)

    # Add a note about removed content
    if len(potential_headers_footers) > 0:
        removed_count = len(potential_headers_footers)
        info_text = f"\n[Note: {removed_count} repeated header/footer lines were automatically removed]\n"
        cleaned_text = info_text + cleaned_text

    return cleaned_text


def extract_pdf_text(pdf_path):
    logger.info("📄 Chargement des modèles...")
    start_time = time.time()
    
    original_pdf_path = pdf_path  # Sauvegarde du chemin original
    oriented_pdf_created = False  # Pour suivre si un fichier orienté a été créé

    try:
        # Nettoyer le PDF (masquer signatures et tampons)
        # pdf_path = clean_pdf(pdf_path)

        # Corriger l'orientation du PDF si nécessaire
        logger.info("🔄 Vérification de l'orientation des pages...")
        oriented_pdf_path = correct_pdf_orientation(pdf_path)
        
        # Vérifier si un nouveau fichier orienté a été créé
        if oriented_pdf_path != pdf_path:
            pdf_path = oriented_pdf_path
            oriented_pdf_created = True
            logger.info(f"✅ Utilisation du PDF orienté: {pdf_path}")

        # Configure Ollama service (local mode)
        os.environ["OLLAMA_BASE_URL"] = os.getenv(
            "OLLAMA_URL", "http://localhost:11434"
        ).split("/api")[0]
        logger.debug(f"OLLAMA_BASE_URL configuré à {os.environ['OLLAMA_BASE_URL']}")

        # S'assurer que tous les indicateurs de mode hors ligne sont activés
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["S3_OFFLINE"] = "1"
        os.environ["NO_PROXY"] = "*"

        # Setup model paths from config.env
        marker_dir = os.getenv("MARKER_DIR", "offline_models/marker")
        logger.info(f"📁 Utilisation du dossier Marker: {marker_dir}")

        # Vérifier si les répertoires existent
        if not os.path.exists(marker_dir):
            logger.warning(f"⚠️ Le répertoire Marker n'existe pas: {marker_dir}")
            # Créer le répertoire si nécessaire
            os.makedirs(marker_dir, exist_ok=True)

        # Chemins des modèles configurés à partir de config.env
        model_paths = {
            "layout": f"{marker_dir}/layout",
            "texify": f"{marker_dir}/texify",
            "text_recognition": f"{marker_dir}/text_recognition",
            "table_recognition": f"{marker_dir}/table_recognition",
            "text_detection": f"{marker_dir}/text_detection",
            "inline_math_detection": f"{marker_dir}/inline_math_detection",
        }

        # Vérifier les chemins des modèles
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Chemin de modèle inexistant: {model_path}")
            else:
                logger.debug(f"✅ Modèle trouvé: {model_name} dans {model_path}")

        logger.info("🔍 Configuration de Marker...")
        # Setup the configuration for Marker with enhanced options
        config = {
            "converter_cls": "marker.converters.table.TableConverter",
            "output_format": "markdown",
            "ocr": True,
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
            "debug": True,
        }

        logger.info("🔄 Création des modèles...")
        # Appliquer tous les patches avant de créer les modèles
        patch_posthog()
        patch_s3_download()
        patch_marker_models()
        patch_create_model_dict()

        # Créer le dictionnaire des modèles
        model_dict = create_model_dict()

        # Déplacer les modèles sur le device approprié
        for model_name, model in model_dict.items():
            if isinstance(model, torch.nn.Module):
                model.to(device)
                # Use half precision for MPS/CUDA
                use_half_precision = (
                    os.getenv("USE_HALF_PRECISION", "true").lower() == "true"
                )
                if use_half_precision and (
                    device.type == "mps" or device.type == "cuda"
                ):
                    model.half()
                    logger.info(f"Modèle {model_name} converti en FP16 pour {device}")
                logger.info(
                    f"✅ Modèle {model_name} chargé sur {device} avec dtype {model.dtype}"
                )

        config_parser = ConfigParser(config)

        logger.info("📝 Conversion du PDF...")
        # Convert PDF with Marker
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        # Process the PDF and extract text
        # Ensure pdf_path is a string for PdfConverter
        rendered = converter(str(pdf_path))
        text, metadata, _ = text_from_rendered(rendered)

        # Apply header/footer removal by similarity
        logger.info(
            "🔍 Détection et suppression des en-têtes, pieds de page et références d'images..."
        )
        text = remove_headers_footers_by_similarity(
            text, similarity_threshold=0.8, occurrence_threshold=3
        )
        logger.info(
            "✅ Traitement des en-têtes, pieds de page et références d'images terminé"
        )

        # Extraire le nom complet du fichier (avec extension) comme titre du document
        filename = os.path.basename(pdf_path)
        logger.info(f"📄 Nom de fichier extrait comme titre de document: {filename}")

        # Extraire un titre lisible à partir du contenu du PDF ou utiliser le nom de fichier
        document_title = ""
        try:
            # Essayer d'extraire un titre des premières lignes du texte
            lines = text.strip().split("\n")
            for i in range(min(10, len(lines))):
                if lines[i] and len(lines[i]) > 10 and lines[i].upper() == lines[i]:
                    document_title = lines[i].strip()
                    break

            # Si aucun titre n'a été trouvé, chercher des lignes avec "contrat", "accord", etc.
            if not document_title:
                title_keywords = [
                    "contrat",
                    "accord",
                    "convention",
                    "attestation",
                    "avenant",
                ]
                for i in range(min(20, len(lines))):
                    if any(keyword in lines[i].lower() for keyword in title_keywords):
                        document_title = lines[i].strip()
                        break
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de l'extraction du titre: {str(e)}")

        # Si aucun titre n'est extrait, utiliser le nom de fichier
        if not document_title:
            document_title = os.path.splitext(filename)[0]

        # Ajouter toujours le nom de fichier au titre pour assurer la traçabilité
        full_document_title = f"{filename}"
        logger.info(f"📄 Titre final du document: {full_document_title}")

        # S'assurer que metadata est un dictionnaire
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}

        # Ajouter le nom de fichier et le titre aux métadonnées
        metadata["filename"] = filename
        metadata["document_title"] = full_document_title

        # Add metadata to the text in a safe way
        text_with_metadata = f"""
Document Metadata:
- Filename: {filename}

Content:
{text}
"""

        # Calculate time taken and log
        end_time = time.time()
        logger.info(f"✅ Extraction terminée en {end_time - start_time:.2f} secondes")
        
        # Nettoyer le fichier temporaire d'orientation si créé
        if oriented_pdf_created and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info(f"🧹 Fichier temporaire supprimé: {pdf_path}")
            except Exception as clean_e:
                logger.warning(f"⚠️ Impossible de supprimer le fichier temporaire {pdf_path}: {str(clean_e)}")

        return text_with_metadata, full_document_title

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte: {str(e)}")
        
        # Nettoyer le fichier temporaire en cas d'erreur aussi
        if oriented_pdf_created and pdf_path != original_pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info(f"🧹 Fichier temporaire supprimé après erreur: {pdf_path}")
            except Exception as clean_e:
                logger.warning(f"⚠️ Impossible de supprimer le fichier temporaire {pdf_path}: {str(clean_e)}")
        
        # Fallback to simpler extraction if marker fails
        logger.warning(
            "⚠️ Échec de l'extraction avancée, utilisation de la méthode simple..."
        )
        return fallback_extract_text(original_pdf_path)


def patch_marker_extract_text():
    # Import marker avec gestion d'erreur
    try:
        import marker
        from marker.extract_text import extract_text, load_and_extract_text

        # Sauvegarder les fonctions originales
        original_extract_text = extract_text
        original_load_and_extract_text = load_and_extract_text

        # Fonction modifiée extract_text
        def patched_extract_text(file_path, **kwargs):
            logger.info(
                "🔧 Utilisation de patched_extract_text pour charger les modèles locaux"
            )

            # S'assurer que tous les modèles sont chargés localement
            kwargs["layout"] = kwargs.get("layout", True)
            kwargs["texify"] = kwargs.get("texify", True)
            kwargs["use_tables"] = kwargs.get(
                "use_tables", False
            )  # Désactiver tables par défaut
            kwargs["use_ocr_error_corruptor"] = kwargs.get(
                "use_ocr_error_corruptor", False
            )  # Désactiver OCR error par défaut
            kwargs["use_gpu"] = kwargs.get("use_gpu", torch.cuda.is_available())
            kwargs["use_mps"] = kwargs.get("use_mps", torch.backends.mps.is_available())
            kwargs["save_preprocessed"] = kwargs.get("save_preprocessed", False)

            # Force le mode offline
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            try:
                return original_extract_text(file_path, **kwargs)
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'extraction avec Marker: {str(e)}")

                # Tentative de repli avec options simplifiées
                logger.info("🔄 Tentative d'extraction avec options simplifiées")
                try:
                    # Essayer sans mise en page (juste OCR)
                    kwargs["layout"] = False
                    kwargs["texify"] = False
                    return original_extract_text(file_path, **kwargs)
                except Exception as e2:
                    logger.error(
                        f"❌ Deuxième erreur d'extraction avec Marker: {str(e2)}"
                    )
                    raise ValueError(
                        f"Échec de l'extraction Marker avec toutes les options: {str(e)}, puis: {str(e2)}"
                    )

        # Fonction modifiée load_and_extract_text
        def patched_load_and_extract_text(file_path, **kwargs):
            logger.info(
                "🔧 Utilisation de patched_load_and_extract_text pour charger les modèles locaux"
            )

            # Force le mode offline
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            try:
                return original_load_and_extract_text(file_path, **kwargs)
            except Exception as e:
                logger.error(f"❌ Erreur lors de load_and_extract_text: {str(e)}")

                # Tentative de repli avec options simplifiées
                logger.info("🔄 Tentative d'extraction simplifiée")
                try:
                    # Essayer sans mise en page (juste OCR)
                    kwargs["layout"] = False
                    kwargs["texify"] = False
                    kwargs["use_tables"] = False
                    return original_load_and_extract_text(file_path, **kwargs)
                except Exception as e2:
                    logger.error(
                        f"❌ Deuxième erreur avec load_and_extract_text: {str(e2)}"
                    )
                    raise ValueError(
                        f"Échec de load_and_extract_text avec toutes les options: {str(e)}, puis: {str(e2)}"
                    )

        # Remplacer les fonctions originales
        marker.extract_text.extract_text = patched_extract_text
        marker.extract_text.load_and_extract_text = patched_load_and_extract_text
        logger.info("✅ Patch marker.extract_text appliqué avec succès")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Module marker.extract_text non disponible: {str(e)}")
        return False


def fallback_extract_text(pdf_path):
    """
    Méthode de secours pour extraire le texte d'un PDF en cas d'échec de Marker.
    Utilise PyMuPDF (fitz) directement.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        tuple: (texte extrait, titre du document)
    """
    logger.warning("⚠️ Utilisation de la méthode de secours pour l'extraction de texte")
    try:
        # Ouvrir le PDF avec PyMuPDF
        doc = fitz.open(pdf_path)

        # Extraire le texte de chaque page
        text_content = []
        for page_num, page in enumerate(doc):
            logger.debug(f"Extraction du texte de la page {page_num+1}")
            text = page.get_text()
            text_content.append(text)

        # Joindre le texte de toutes les pages
        full_text = "\n".join(text_content)

        # Appliquer la suppression des en-têtes/pieds de page
        cleaned_text = remove_headers_footers_by_similarity(full_text)

        # Obtenir le nom du fichier comme base du titre
        filename = os.path.basename(pdf_path)
        logger.info(f"📄 Nom de fichier extrait: {filename}")

        # Essayer d'extraire un titre significatif des premières lignes du document
        document_title = ""
        try:
            # Extraire un titre des premières lignes du texte
            lines = cleaned_text.strip().split("\n")
            for i in range(min(10, len(lines))):
                if lines[i] and len(lines[i]) > 10 and lines[i].upper() == lines[i]:
                    document_title = lines[i].strip()
                    break

            # Si aucun titre n'a été trouvé, chercher des lignes avec "contrat", "accord", etc.
            if not document_title:
                title_keywords = [
                    "contrat",
                    "accord",
                    "convention",
                    "attestation",
                    "avenant",
                ]
                for i in range(min(20, len(lines))):
                    if any(keyword in lines[i].lower() for keyword in title_keywords):
                        document_title = lines[i].strip()
                        break
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de l'extraction du titre: {str(e)}")

        # Si aucun titre n'est extrait, utiliser le nom de fichier sans extension
        if not document_title:
            document_title = os.path.splitext(filename)[0]

        # Ajouter le nom de fichier au titre pour la traçabilité
        full_document_title = f"{document_title} ({filename})"
        logger.info(f"📄 Titre final du document (fallback): {full_document_title}")

        logger.info(
            f"✅ Extraction de secours terminée: {len(cleaned_text.split())} mots extraits"
        )

        # Créer un texte avec métadonnées
        text_with_metadata = f"""
Document Metadata:
- Filename: {filename}

Content:
{cleaned_text}
"""

        return text_with_metadata, filename

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction de secours: {str(e)}")
        # En cas d'échec complet, retourner un texte vide avec le nom du fichier
        filename = os.path.basename(pdf_path)
        return f"Échec de l'extraction du texte: {str(e)}", filename


def init():
    """Initialize the PDF extractor module. Must be called before using the module."""
    logger.info("🔄 Initializing PDF extractor module")

    # Forcer les variables d'environnement de manière préventive
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Appliquer les patches pour utiliser les modèles en local
    patch_create_model_dict()
    patch_marker_extract_text()

    # Log info about parameters
    logger.info(
        f"📋 MARKER_DIR: {os.environ.get('MARKER_DIR', 'offline_models/marker')}"
    )
    logger.info(f"📋 Using HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
    logger.info(
        f"📋 Using TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}"
    )

    # Check if torch is available with CUDA or MPS
    if torch.cuda.is_available():
        logger.info(
            f"🚀 Using CUDA for PDF extraction: {torch.cuda.get_device_name(0)}"
        )
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        logger.info("🚀 Using MPS (Metal Performance Shaders) for PDF extraction")
    else:
        logger.info("⚠️ No GPU available, using CPU for PDF extraction")

    logger.info("✅ PDF extractor module initialized successfully")


# Ne pas initialiser à l'import - laisser l'application appeler init()
