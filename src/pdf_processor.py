import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # Create a dummy PDF file for testing if you don't have one
    # from reportlab.pdfgen import canvas
    # c = canvas.Canvas("dummy.pdf")
    # c.drawString(100, 750, "This is a test PDF document.")
    # c.save()

    # extracted_text = extract_text_from_pdf("dummy.pdf")
    # if extracted_text:
    #     print("Extracted Text:")
    #     print(extracted_text)
    pass
