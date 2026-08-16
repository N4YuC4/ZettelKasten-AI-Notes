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
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
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
                    page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text)
                except Exception as pe:
                    log_error(f"Error extracting text from page {idx + 1} of '{pdf_path}': {pe}")

            text = "\n\n".join(extracted_pages).strip()
            if not text:
                log_error(f"PDF contains no readable text: '{pdf_path}'")
                raise ValueError("Selected PDF file contains no readable text.")

            return text # Return all the extracted text
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

