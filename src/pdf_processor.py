# pdf_processor.py
#
# This file contains helper functions for extracting text from PDF files.
# It uses the pypdf library to extract text from all pages of a PDF file.

import os # For file existence and size checks
import pypdf # Library for reading and processing PDF files
from logger import log_error # For error logging function

# The extract_text_from_pdf function extracts text from a specific PDF file.
# pdf_path (str): The path to the PDF file.
# Returns: The extracted text from the PDF (str).
def normalize_pdf_text(text: str) -> str:
    """
    Normalizes text extracted from PDF documents.
    - Preserves paragraphs and intentional double linebreaks.
    - Heals fragmented word-by-word linebreaks (common with PDF print drivers).
    - Reconnects broken hyphenated words at line endings.
    - Cleans up excessive horizontal whitespace while preserving table alignment.
    """
    import re
    if not text or not str(text).strip():
        return ""

    # 1. Normalize vertical spacing: replace 3+ newlines with double newline
    normalized = re.sub(r'\n{3,}', '\n\n', text)

    # 2. Reconnect hyphenated words split across linebreaks in any language/alphabet
    normalized = re.sub(r'(\b[^\W\d_]+)-\s*\n\s*([^\W\d_]+\b)', r'\1\2', normalized)

    # 3. Detect and heal word-by-word line breaks (where almost every line is an isolated word)
    lines = normalized.split('\n')
    non_empty = [l.strip() for l in lines if l.strip()]
    if non_empty and len(non_empty) > 10:
        avg_words_per_line = sum(len(l.split()) for l in non_empty) / len(non_empty)
        if avg_words_per_line < 3.0:
            paras = normalized.split('\n\n')
            rebuilt_paras = []
            for p in paras:
                p_lines = [l.strip() for l in p.split('\n') if l.strip()]
                if not p_lines:
                    continue
                cur_block = []
                for pl in p_lines:
                    # Preserve list items or markdown headers on their own line
                    if re.match(r'^(?:[#\-*•]|\d+[\.)])\s+', pl):
                        if cur_block:
                            rebuilt_paras.append(' '.join(cur_block))
                            cur_block = []
                        cur_block.append(pl)
                    else:
                        cur_block.append(pl)
                if cur_block:
                    rebuilt_paras.append(' '.join(cur_block))
            normalized = '\n\n'.join(rebuilt_paras)

    # 4. Collapse runs of horizontal whitespace, leaving max 2 spaces for alignment
    normalized = re.sub(r'[ \t]{3,}', '  ', normalized)

    return normalized.strip()


def extract_text_from_pdf(pdf_path):
    """
    Extracts and normalizes text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted and normalized text from the PDF.
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF is encrypted, empty, or has no extractable text.
        RuntimeError: If an error occurs during text extraction.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        err_msg = f"PDF file not found: '{pdf_path}'"
        log_error(err_msg)
        raise FileNotFoundError(err_msg)

    if os.path.getsize(pdf_path) == 0:
        err_msg = f"PDF file is empty (0 bytes): '{pdf_path}'"
        log_error(err_msg)
        raise ValueError(err_msg)

    try:
        with open(pdf_path, "rb") as file: # Read the PDF file in binary mode
            reader = pypdf.PdfReader(file) # Create a PdfReader object
            
            # Check if the PDF file is password protected/encrypted
            if reader.is_encrypted:
                log_error(f"PDF is encrypted: '{pdf_path}'")
                raise ValueError("PDF is encrypted and password-protected.")

            # Check if PDF has at least one page
            if len(reader.pages) == 0:
                log_error(f"PDF contains no pages: '{pdf_path}'")
                raise ValueError("PDF contains no pages.")

            extracted_pages = []
            for idx, page in enumerate(reader.pages): # Loop for each page
                try:
                    try:
                        page_text = page.extract_text(extraction_mode="layout")
                    except Exception:
                        page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text)
                except Exception as pe:
                    log_error(f"Error extracting text from page {idx + 1} of '{pdf_path}': {pe}")

            raw_text = "\n\n".join(extracted_pages).strip()
            if not raw_text:
                log_error(f"PDF contains no readable text: '{pdf_path}'")
                raise ValueError("Selected PDF file contains no readable text.")

            return normalize_pdf_text(raw_text)
    except (ValueError, FileNotFoundError):
        raise
    except pypdf.errors.PdfReadError as pre:
        log_error(f"Failed to read PDF file (corrupt or invalid format): {pre}")
        raise ValueError(f"Failed to read PDF file (corrupt or invalid format): {pre}") from pre
    except Exception as e:
        log_error(f"Error extracting text from PDF: {e}") # Log error if an exception occurs
        raise RuntimeError(f"Error extracting text from PDF: {e}") from e

# This block provides an example usage when the file is run directly (for testing purposes).
if __name__ == '__main__':
    # Example usage (for testing purposes)
    # If you don't have a PDF file, you can create a dummy PDF file for testing.
    # from reportlab.pdfgen import canvas
    # c = canvas.Canvas("dummy.pdf")
    # c.drawString(100, 750, "This is a test PDF document.")
    # c.save()

    # extracted_text = extract_text_from_pdf("dummy.pdf")
    # if extracted_text:
    #     print("Extracted Text:")
    #     print(extracted_text)
    pass

