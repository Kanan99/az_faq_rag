# Extracts FAQs from PDF documents for the Azerbaijani FAQ RAG system

import PyPDF2
import re
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFFAQExtractor:
    """Extracts FAQs from PDF documents for the Azerbaijani FAQ RAG system."""
    
    def __init__(self, output_dir: str = 'data'):
        """
        Initialize the PDF FAQ Extractor.
        
        Args:
            output_dir: Directory to save the extracted FAQs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                logger.info(f"Successfully extracted {len(pdf_reader.pages)} pages")
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_faqs_from_text(self, text: str, category: str = "Süni İntellekt") -> List[Dict]:
        """
        Extract FAQs from the provided text.
        
        This function uses heuristic patterns to identify question-answer pairs
        in the document. It looks for:
        1. Section titles/headings that appear to be questions
        2. Numbered/bulleted questions followed by answers
        3. Question-like sentences followed by explanatory paragraphs
        
        Args:
            text: The text to extract FAQs from
            category: Default category for the extracted FAQs
            
        Returns:
            List of dictionaries containing questions and answers
        """
        logger.info("Extracting FAQs from text")
        
        faqs = []
        
        # Split the text into sections/paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Pattern for questions (ending with question mark or numbered/bulleted)
        question_patterns = [
            r'^(\d+[\.\)]\s*)(.*?)$',  # Numbered questions: "1. Question"
            r'^(•|\*|\-)\s*(.*?)$',     # Bulleted questions: "• Question"
            r'^([A-Za-z0-9ƏəİıÖöÜüĞğŞşÇç]+[^.?!]*\?)$',  # Questions ending with ?
            r'^((?:Nə|Necə|Niyə|Hansı|Hara|Kim|Nəyə|Harada|Nə üçün|Nə zaman|Neçə)[^.?!]*\?)$',  # Azeri question words
        ]
        
        current_question = None
        current_answer = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this paragraph is a question
            is_question = False
            for pattern in question_patterns:
                match = re.match(pattern, para, re.IGNORECASE | re.MULTILINE)
                if match:
                    if current_question and current_answer:
                        # Save the previous Q&A pair
                        faqs.append({
                            "question": current_question,
                            "answer": ' '.join(current_answer),
                            "category": category
                        })
                    
                    # Extract the new question
                    if match.groups() and len(match.groups()) > 1:
                        current_question = match.group(2).strip()
                    else:
                        current_question = para.strip()
                    
                    current_answer = []
                    is_question = True
                    break
            
            if not is_question and current_question:
                # This must be part of the answer
                current_answer.append(para)
        
        # Don't forget the last Q&A pair
        if current_question and current_answer:
            faqs.append({
                "question": current_question,
                "answer": ' '.join(current_answer),
                "category": category
            })
        
        logger.info(f"Extracted {len(faqs)} FAQs from text")
        return faqs
    
    def extract_faqs_from_pdf(self, pdf_path: str, category: str = "Süni İntellekt") -> List[Dict]:
        """
        Extract FAQs from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            category: Default category for the extracted FAQs
            
        Returns:
            List of dictionaries containing questions and answers
        """
        text = self.extract_text_from_pdf(pdf_path)
        return self.extract_faqs_from_text(text, category)
    
    def save_faqs_to_csv(self, faqs: List[Dict], output_file: str = None) -> str:
        """
        Save the extracted FAQs to a CSV file.
        
        Args:
            faqs: List of dictionaries containing questions and answers
            output_file: Path to save the CSV file (optional)
            
        Returns:
            Path to the saved CSV file
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, 'azerbaijani_ai_faqs.csv')
        
        df = pd.DataFrame(faqs)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(faqs)} FAQs to {output_file}")
        return output_file
    
    def process_pdf(self, pdf_path: str, category: str = "Süni İntellekt", output_file: str = None) -> str:
        """
        Process a PDF file to extract FAQs and save them to a CSV file.
        
        Args:
            pdf_path: Path to the PDF file
            category: Default category for the extracted FAQs
            output_file: Path to save the CSV file (optional)
            
        Returns:
            Path to the saved CSV file
        """
        faqs = self.extract_faqs_from_pdf(pdf_path, category)
        return self.save_faqs_to_csv(faqs, output_file)


if __name__ == "__main__":
    # Example usage
    extractor = PDFFAQExtractor()
    pdf_path = "sample.pdf"  # Replace with your PDF file
    csv_path = extractor.process_pdf(pdf_path)
    print(f"FAQs extracted and saved to: {csv_path}")
