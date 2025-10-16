# pdf_processor.py
#
# This file contains helper functions for extracting text from PDF files.
# It uses the PyPDF2 library to extract text from all pages of a PDF file.

import PyPDF2 # Library for reading and processing PDF files

# The extract_text_from_pdf function extracts text from a specific PDF file.
# pdf_path (str): The path to the PDF file.
# Returns: The extracted text from the PDF (str) or None if an error occurs.
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = "" # An empty string to store the extracted text
    try:
        with open(pdf_path, "rb") as file: # Read the PDF file in binary mode
            reader = PyPDF2.PdfReader(file) # Create a PdfReader object
            for page_num in range(len(reader.pages)): # Loop for each page
                text += reader.pages[page_num].extract_text() # Extract and append the text from the page
    except Exception as e:
        print(f"Error extracting text from PDF: {e}") # Print if an error occurs
        return None # Return None
    return text # Return all the extracted text

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
