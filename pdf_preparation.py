import os
import sys
import base64
import requests
import time
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import re

# === CONFIGURATION ===
OLLAMA_URL = "http://localhost:11434/api/generate"
DPI = 300
MODELS = [
    "llama3.2-vision:90b"
]
PROMPT = (
    "Transcribe ONLY the text visible on this page. Do not try to complete cut-off paragraphs.\n"
    "\n"
    "Rules:\n"
    "- Copy exactly what you see on this page only\n"
    "- If a paragraph is cut off at the bottom, stop at the last visible word\n"
    "- Do not try to guess or complete cut-off text\n"
    "- Do not add any text that isn't visible\n"
    "- Keep section numbers and titles as they appear\n"
    "- For tables: use | for columns, - for borders\n"
    "- For forms: use [FIELD] and [SIGNATURE]\n"
    "\n"
    "If a word is unreadable, write [UNREADABLE].\n"
    "Do not add any analysis or explanations.\n"
    "Only copy what you see on this page."
)

def clean_vision_output(text: str) -> str:
    """Nettoie et normalise la sortie du modèle vision."""
    
    # Supprimer les marqueurs de page
    text = re.sub(r'---\s*Page \d+.*?---', '', text, flags=re.MULTILINE)
    
    # Supprimer les commentaires descriptifs du modèle
    text = re.sub(r'The (scanned page|document|image) (shows|is|appears).*?\n', '', text)
    text = re.sub(r'This document appears to be.*?\n', '', text)
    text = re.sub(r'The text is.*?\n', '', text)
    
    # Supprimer les textes en gras qui ne sont pas des numéros de section
    def is_section_number(text):
        # Patterns pour détecter les numéros de section
        patterns = [
            r'^\d+(\.\d+)*\.',  # 1., 1.1., 1.1.1., etc.
            r'^Article \d+(\.\d+)*\.',  # Article 1., Article 1.1., etc.
            r'^Section \d+(\.\d+)*\.',  # Section 1., Section 1.1., etc.
            r'^Clause \d+(\.\d+)*\.'    # Clause 1., Clause 1.1., etc.
        ]
        return any(re.match(pattern, text.strip()) for pattern in patterns)
    
    # Fonction pour nettoyer les textes en gras
    def clean_bold(match):
        text = match.group(1)
        if is_section_number(text):
            return text  # Garder le texte s'il s'agit d'un numéro de section
        return ''  # Supprimer le texte sinon
    
    # Appliquer le nettoyage des textes en gras
    text = re.sub(r'\*\*(.*?)\*\*', clean_bold, text)
    
    # Supprimer les lignes vides multiples
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Supprimer les espaces en début et fin de ligne
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Ajouter un saut de ligne final
    text = text.strip() + '\n'
    
    return text

def merge_pages(pages: list) -> str:
    """Fusionne les pages en un seul texte cohérent."""
    merged = []
    current_section = None
    
    for page in pages:
        # Nettoyer la page
        clean_page = clean_vision_output(page)
        
        # Si la page commence par une continuation de section
        if current_section and not re.match(r'^###|^\d+(?:\.\d+)*\.', clean_page.lstrip()):
            merged[-1] += '\n' + clean_page
        else:
            merged.append(clean_page)
            # Trouver la dernière section de la page
            sections = re.findall(r'(###.*?$|\d+(?:\.\d+)*\..*?$)', clean_page, re.MULTILINE)
            if sections:
                current_section = sections[-1]
            else:
                current_section = None
    
    return '\n\n'.join(merged)

def usage():
    print("❌ Usage : python benchmark_ocr_models.py fichier.pdf")
    sys.exit(1)

if len(sys.argv) != 2 or not sys.argv[1].endswith(".pdf"):
    usage()

PDF_PATH = sys.argv[1]
filename_base = os.path.splitext(os.path.basename(PDF_PATH))[0]

# === Convertit le PDF en images une seule fois ===
TEMP_IMG_DIR = f"tmp_{filename_base}"
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

print("📄 Conversion du PDF en images...")
images = convert_from_path(PDF_PATH, dpi=DPI, output_folder=TEMP_IMG_DIR, fmt="jpeg")

def process_page(image, model, page_num, width, height, txt_path, pdf_dir, all_texts):
    """Traite une page individuelle et met à jour les fichiers."""
    print(f"\n🧹 Page {page_num+1} - Prétraitement...")

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_path = os.path.join(TEMP_IMG_DIR, f"{model_slug}_page_{page_num+1:03}.jpg")
    cv2.imwrite(img_path, thresh)

    with open(img_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [image_base64],
        "stream": False
    }

    print(f"📤 Envoi à Ollama ({model}) - Page {page_num+1}")
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        duration = time.time() - start_time
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            print(f"✅ Réponse OK ({duration:.2f}s)")
        else:
            text = "[ERREUR OCR]"
            print(f"❌ Échec de la génération : {response.text}")
    except Exception as e:
        text = "[EXCEPTION]"
        print(f"❌ Exception : {e}")

    # Nettoyer et formater le texte
    clean_text = clean_vision_output(text)
    
    # Ajouter le texte à la liste des textes
    all_texts.append(clean_text)
    
    # Ajouter le texte au fichier texte
    with open(txt_path, "a", encoding="utf-8") as f_txt:
        f_txt.write(clean_text + "\n\n")
    
    # Créer un nouveau PDF avec toutes les pages traitées jusqu'à présent
    pdf_path = os.path.join(pdf_dir, f"page_{page_num+1:03d}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    
    # Configuration de la mise en page
    margin = 40
    line_height = 14
    font_size = 10
    max_width = width - (2 * margin)
    
    # Fonction pour ajuster le texte à la largeur de la page
    def wrap_text(text, max_width, font_size):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = c.stringWidth(word, "Helvetica", font_size)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + c.stringWidth(" ", "Helvetica", font_size)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width + c.stringWidth(" ", "Helvetica", font_size)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    # Ajouter tout le contenu traité jusqu'à présent
    c.setFont("Helvetica", font_size)
    y = height - margin
    
    for text in all_texts:
        for line in text.split("\n"):
            # Ignorer les lignes vides
            if not line.strip():
                y -= line_height
                continue
                
            # Ajuster le texte à la largeur de la page
            wrapped_lines = wrap_text(line, max_width, font_size)
            
            for wrapped_line in wrapped_lines:
                if y < margin:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", font_size)
                
                c.drawString(margin, y, wrapped_line)
                y -= line_height
    
    # Sauvegarder le PDF
    c.save()
    
    return clean_text

# === Appliquer OCR avec chaque modèle ===
for model in MODELS:
    print(f"\n🚀 Traitement avec le modèle : {model}")

    model_slug = model.replace(":", "_")
    base_name = f"{filename_base}_{model_slug}"
    txt_path = f"{base_name}.txt"
    pdf_dir = f"{base_name}_pdfs"

    # Créer le dossier pour les PDFs
    os.makedirs(pdf_dir, exist_ok=True)

    # Définir les dimensions de la page
    width, height = A4

    # Vider les fichiers s'ils existent déjà
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("")

    text_results = []
    all_texts = []
    
    for i, image in enumerate(images):
        text = process_page(image, model, i, width, height, txt_path, pdf_dir, all_texts)
        text_results.append(text)
        
        # Afficher la progression
        print(f"📊 Progression : {i+1}/{len(images)} pages traitées")
        print(f"📄 Fichiers mis à jour : {txt_path}, {pdf_dir}/page_{i+1:03d}.pdf")

    print(f"\n📦 Traitement terminé pour le modèle {model}")
    print(f"📄 Fichiers finaux :")
    print(f"  - Texte : {txt_path}")
    print(f"  - PDFs : {pdf_dir}/")

print("\n✅ Benchmark terminé pour tous les modèles.")

# Nettoyage
print("\n🧹 Nettoyage des fichiers temporaires...")
for file in os.listdir(TEMP_IMG_DIR):
    os.remove(os.path.join(TEMP_IMG_DIR, file))
os.rmdir(TEMP_IMG_DIR)
print("✅ Nettoyage terminé")
