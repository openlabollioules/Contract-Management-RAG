import fitz  # PyMuPDF
import sys
from PIL import Image
import pytesseract
import io

def est_document_scanné(pdf_path):
    """
    Détecte si un PDF est scanné en analysant son contenu.
    Retourne True si le document est probablement scanné, False sinon.
    """
    try:
        # Ouvrir le PDF
        doc = fitz.open(pdf_path)
        
        # Vérifier le nombre de pages
        if len(doc) == 0:
            return False
            
        # Analyser la première page
        page = doc[0]
        
        # Vérifier s'il y a des images
        images = page.get_images()
        if not images:
            # Si pas d'images, c'est probablement un document natif
            return False
        
        # Vérifier la taille des images
        total_image_area = 0
        page_area = page.rect.width * page.rect.height
        
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            total_image_area += image.width * image.height
        
        # Si les images couvrent moins de 20% de la page, c'est probablement un document natif
        if total_image_area < (0.2 * page_area):
            return False
        
        # Extraire le texte
        text = page.get_text()
        
        # Si le texte est long et de bonne qualité, c'est probablement un document natif
        if len(text.strip()) > 200:
            # Vérifier la qualité du texte
            special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?()-_')
            if special_chars < (0.05 * len(text)):  # Moins de 5% de caractères spéciaux
                return False
        
        # Si on arrive ici, c'est probablement un document scanné
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'analyse du PDF: {str(e)}")
        return False

# Exemple d'utilisation
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scanned_pdf.py <chemin_vers_pdf>")
        sys.exit(1)
        
    fichier_pdf = sys.argv[1]
    if est_document_scanné(fichier_pdf):
        print("Type de document: Document scanné (image d'un document papier)")
        print("Caractéristiques: Contient des images de pages, peut avoir des artefacts de numérisation")
    else:
        print("Type de document: Document natif (créé directement en PDF)")
        print("Caractéristiques: Texte sélectionnable, pas d'artefacts de numérisation")
