# Core functionality for Azerbaijani FAQ RAG system with improved character handling

import os
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzerbaijaniFAQRAG:
    """
    A Retrieval-Augmented Generation system for Azerbaijani FAQs using LaBSE for embeddings.
    This system is specifically designed to work with Azerbaijan's 2025-2028 AI Strategy FAQs.
    """
    def __init__(self, model_name: str = 'sentence-transformers/LaBSE', cache_dir: Optional[str] = None):
        """
        Initialize the RAG system with LaBSE model.
        
        Args:
            model_name: Name of the model to use for embeddings
            cache_dir: Directory to cache the model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.faqs = None
        self.embeddings = None
        self.loaded = False
        
        # Define Azerbaijani character mappings (Latin to Cyrillic and various transliterations)
        self.char_mappings = {
            'sh': 'ş',
            'ch': 'ç',
            'w': 'v',
            'q': 'q',  # Keep q as is in Azerbaijani
            'x': 'x',  # Keep x as is (represents 'kh' sound)
            'sch': 'şç',
            # Special letters
            'ə': 'ə',
            'ü': 'ü',
            'ö': 'ö',
            'ğ': 'ğ',
            'ı': 'ı',
            # Latin alternatives that might be used
            'a': 'a',
            'b': 'b',
            'c': 'c',
            'd': 'd',
            'e': 'e',
            'f': 'f',
            'g': 'g',
            'h': 'h',
            'i': 'i',
            'j': 'j',
            'k': 'k',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'o': 'o',
            'p': 'p',
            'r': 'r',
            's': 's',
            't': 't',
            'u': 'u',
            'v': 'v',
            'y': 'y',
            'z': 'z'
        }
    
    def preprocess_azerbaijani_text(self, text: str) -> str:
        """
        Preprocess Azerbaijani text for better embedding quality.
        Handles various transliteration styles and normalizes characters.
        
        Args:
            text: Input text in Azerbaijani
            
        Returns:
            Preprocessed text
        """
        if text is None:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Apply replacements for character combinations
        for latin, azeri in self.char_mappings.items():
            if len(latin) > 1:  # Only process multi-character replacements here
                # Replace with word boundaries to avoid replacing parts of words
                text = re.sub(r'\b{}\b'.format(latin), azeri, text)
                text = re.sub(r'\b{}\b'.format(latin.capitalize()), azeri, text)
                
                # Also try to replace within words
                text = re.sub(r'{}'.format(latin), azeri, text)
        
        # Normalize specific Azerbaijani characters 
        # Example: Replace 'w' with 'v' when used as an alternative
        text = text.replace('w', 'v')
        
        # Remove excessive spaces
        text = ' '.join(text.split())
        
        return text
    
    def load_faqs_from_csv(self, file_path: str, question_col: str = 'question', 
                         answer_col: str = 'answer', category_col: Optional[str] = None) -> None:
        """
        Load FAQs from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            question_col: Name of the question column
            answer_col: Name of the answer column
            category_col: Name of the category column (optional)
        """
        logger.info(f"Loading FAQs from {file_path}")
        self.faqs = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if question_col not in self.faqs.columns:
            raise ValueError(f"Question column '{question_col}' not found in CSV")
        if answer_col not in self.faqs.columns:
            raise ValueError(f"Answer column '{answer_col}' not found in CSV")
        
        # Rename columns for consistency
        self.faqs = self.faqs.rename(columns={
            question_col: 'question',
            answer_col: 'answer'
        })
        
        if category_col and category_col in self.faqs.columns:
            self.faqs = self.faqs.rename(columns={category_col: 'category'})
        
        # Preprocess questions
        self.faqs['processed_question'] = self.faqs['question'].apply(self.preprocess_azerbaijani_text)
        
        # Create embeddings for all questions
        self._create_embeddings()
        
        logger.info(f"Loaded {len(self.faqs)} FAQs")
        self.loaded = True
    
    def add_faq(self, question: str, answer: str, category: Optional[str] = None) -> None:
        """
        Add a new FAQ entry to the system.
        
        Args:
            question: The question text
            answer: The answer text
            category: Optional category for the FAQ
        """
        # Initialize self.faqs if not already done
        if self.faqs is None:
            columns = ['question', 'answer', 'processed_question']
            if category is not None:
                columns.append('category')
            self.faqs = pd.DataFrame(columns=columns)
        
        # Preprocess the new question
        processed_question = self.preprocess_azerbaijani_text(question)
        
        # Create new row
        new_row = {
            'question': question,
            'answer': answer,
            'processed_question': processed_question
        }
        if category is not None and 'category' in self.faqs.columns:
            new_row['category'] = category
        elif category is not None:
            # Add category column if it doesn't exist
            self.faqs['category'] = 'Artificial Intelligence'  # Default category for existing rows
            new_row['category'] = category
        
        # Add to dataframe
        self.faqs = pd.concat([self.faqs, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update embeddings
        self._create_embeddings()
        self.loaded = True
        logger.info(f"Added new FAQ: {question[:50]}...")
    
    def load_faqs_from_text(self, text_content: str, delimiter: str = "###") -> None:
        """
        Load FAQs from raw text content with a specified delimiter.
        
        Args:
            text_content: Raw text containing FAQs
            delimiter: Delimiter that separates questions and answers
        """
        lines = text_content.strip().split('\n')
        faq_data = []
        
        current_faq = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(delimiter):
                # Save previous FAQ if exists
                if current_faq and 'question' in current_faq and 'answer' in current_faq:
                    faq_data.append(current_faq)
                
                # Start new FAQ
                current_faq = {'question': line[len(delimiter):].strip()}
            elif current_faq and 'question' in current_faq and 'answer' not in current_faq:
                current_faq['answer'] = line
                current_faq['category'] = 'Artificial Intelligence'  # Default category
        
        # Add last FAQ if exists
        if current_faq and 'question' in current_faq and 'answer' in current_faq:
            faq_data.append(current_faq)
        
        # Convert to DataFrame
        if faq_data:
            self.faqs = pd.DataFrame(faq_data)
            # Preprocess questions
            self.faqs['processed_question'] = self.faqs['question'].apply(self.preprocess_azerbaijani_text)
            
            # Create embeddings
            self._create_embeddings()
            
            logger.info(f"Loaded {len(self.faqs)} FAQs from text")
            self.loaded = True
        else:
            logger.warning("No FAQs found in the provided text")
    
    def _create_embeddings(self) -> None:
        """Create embeddings for all preprocessed questions."""
        if self.faqs is None or len(self.faqs) == 0:
            logger.warning("No FAQs to embed")
            return
        
        logger.info(f"Creating embeddings for {len(self.faqs)} questions...")
        questions = self.faqs['processed_question'].tolist()
        self.embeddings = self.model.encode(questions, show_progress_bar=True)
        logger.info("Embeddings created successfully")
    
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve the most relevant FAQ entries for a given query.
        
        Args:
            query: The query text in Azerbaijani
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of dictionaries containing questions, answers, and similarity scores
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded. Call load_faqs_from_csv() or add_faq() first.")
        
        # Preprocess query
        processed_query = self.preprocess_azerbaijani_text(query)
        
        # Encode query
        query_embedding = self.model.encode([processed_query])[0]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create result list with threshold filtering
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                result = {
                    'question': self.faqs.iloc[idx]['question'],
                    'answer': self.faqs.iloc[idx]['answer'],
                    'similarity': similarities[idx]
                }
                if 'category' in self.faqs.columns:
                    result['category'] = self.faqs.iloc[idx]['category']
                results.append(result)
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of query texts in Azerbaijani
            top_k: Number of top results to return for each query
            
        Returns:
            List of result lists, one for each query
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded. Call load_faqs_from_csv() or add_faq() first.")
        
        # Process all queries
        processed_queries = [self.preprocess_azerbaijani_text(query) for query in queries]
        
        # Encode all queries
        query_embeddings = self.model.encode(processed_queries, show_progress_bar=True)
        
        # Calculate similarities for all queries
        all_similarities = cosine_similarity(query_embeddings, self.embeddings)
        
        # Create results for each query
        all_results = []
        for i, similarities in enumerate(all_similarities):
            # Get top_k indices for this query
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create result list
            results = []
            for idx in top_indices:
                result = {
                    'question': self.faqs.iloc[idx]['question'],
                    'answer': self.faqs.iloc[idx]['answer'],
                    'similarity': similarities[idx]
                }
                if 'category' in self.faqs.columns:
                    result['category'] = self.faqs.iloc[idx]['category']
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def search_by_keyword(self, keyword: str, field: str = 'question') -> pd.DataFrame:
        """
        Simple keyword search in questions or answers.
        
        Args:
            keyword: Keyword or phrase to search for
            field: Field to search in ('question', 'answer', or 'both')
            
        Returns:
            DataFrame with matching FAQs
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded. Call load_faqs_from_csv() or add_faq() first.")
        
        keyword = keyword.lower()
        
        if field == 'question':
            return self.faqs[self.faqs['question'].str.lower().str.contains(keyword)]
        elif field == 'answer':
            return self.faqs[self.faqs['answer'].str.lower().str.contains(keyword)]
        elif field == 'both':
            return self.faqs[
                self.faqs['question'].str.lower().str.contains(keyword) | 
                self.faqs['answer'].str.lower().str.contains(keyword)
            ]
        else:
            raise ValueError("Field must be 'question', 'answer', or 'both'")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded FAQs.
        
        Returns:
            Dictionary with statistics
        """
        if not self.loaded:
            return {"loaded": False, "count": 0}
        
        stats = {
            "loaded": True,
            "count": len(self.faqs),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
        }
        
        if 'category' in self.faqs.columns:
            category_counts = self.faqs['category'].value_counts().to_dict()
            stats["categories"] = category_counts
            stats["category_count"] = len(category_counts)
        
        return stats
    
    def save_faqs(self, file_path: str) -> None:
        """
        Save the current FAQs to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
        """
        if not self.loaded:
            raise ValueError("No FAQs to save.")
        
        # Save only the essential columns
        save_columns = ['question', 'answer']
        if 'category' in self.faqs.columns:
            save_columns.append('category')
        
        self.faqs[save_columns].to_csv(file_path, index=False)
        logger.info(f"FAQs saved to {file_path}")
    
    def export_to_json(self, file_path: str) -> None:
        """
        Export FAQs to a JSON file for easier integration with other systems.
        
        Args:
            file_path: Path to save the JSON file
        """
        if not self.loaded:
            raise ValueError("No FAQs to export.")
        
        export_columns = ['question', 'answer']
        if 'category' in self.faqs.columns:
            export_columns.append('category')
        
        self.faqs[export_columns].to_json(file_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"FAQs exported to {file_path}")
    
    def get_faq_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a specific FAQ by its index.
        
        Args:
            index: Index of the FAQ to retrieve
            
        Returns:
            Dictionary containing the FAQ details
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded.")
        
        if index < 0 or index >= len(self.faqs):
            raise IndexError(f"Index {index} out of bounds (0-{len(self.faqs)-1})")
        
        faq = self.faqs.iloc[index].to_dict()
        return faq
    
    def update_faq(self, index: int, question: Optional[str] = None, 
                  answer: Optional[str] = None, category: Optional[str] = None) -> None:
        """
        Update an existing FAQ.
        
        Args:
            index: Index of the FAQ to update
            question: New question text (optional)
            answer: New answer text (optional)
            category: New category (optional)
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded.")
        
        if index < 0 or index >= len(self.faqs):
            raise IndexError(f"Index {index} out of bounds (0-{len(self.faqs)-1})")
        
        # Update fields
        if question is not None:
            self.faqs.at[index, 'question'] = question
            self.faqs.at[index, 'processed_question'] = self.preprocess_azerbaijani_text(question)
        
        if answer is not None:
            self.faqs.at[index, 'answer'] = answer
        
        if category is not None and 'category' in self.faqs.columns:
            self.faqs.at[index, 'category'] = category
        elif category is not None:
            # Add category column if it doesn't exist
            self.faqs['category'] = 'Artificial Intelligence'  # Default category for existing rows
            self.faqs.at[index, 'category'] = category
        
        # Update embeddings
        if question is not None:
            self._create_embeddings()
        
        logger.info(f"Updated FAQ at index {index}")

    def delete_faq(self, index: int) -> None:
        """
        Delete an FAQ by its index.
        
        Args:
            index: Index of the FAQ to delete
        """
        if not self.loaded:
            raise ValueError("FAQs not loaded.")
        
        if index < 0 or index >= len(self.faqs):
            raise IndexError(f"Index {index} out of bounds (0-{len(self.faqs)-1})")
        
        # Delete the row
        self.faqs = self.faqs.drop(index).reset_index(drop=True)
        
        # Update embeddings
        self._create_embeddings()
        
        logger.info(f"Deleted FAQ at index {index}")


if __name__ == "__main__":
    # Example usage
    rag = AzerbaijaniFAQRAG()
    
    # Sample FAQ data (in Azerbaijani)
    sample_faqs = [
        {
            "question": "Azərbaycanın süni intellekt strategiyası hansı illər üçün nəzərdə tutulub?", 
            "answer": "Azərbaycan Respublikasının 2025–2028-ci illər üçün süni intellekt Strategiyası ölkəmizdə süni intellekt texnologiyalarının inkişafı və tətbiqi üçün nəzərdə tutulmuşdur.", 
            "category": "Süni İntellekt Strategiyası"
        },
        {
            "question": "Azərbaycanın süni intellekt üzrə prioritet istiqamətləri hansılardır?", 
            "answer": "Strategiyanın prioritet istiqamətlərinə süni intellektin idarə edilməsinin inkişafı və təkmilləşdirilməsi, məlumat idarəetməsinin və infrastrukturunun təkmilləşdirilməsi, süni intellekt sahəsində istedad və bacarıqların təkmilləşdirilməsi və əlverişli biznes mühitinin yaradılması daxildir.", 
            "category": "Prioritet İstiqamətlər"
        },
        {
            "question": "Süni intellekt sahəsində kadr hazırlığı necə həyata keçiriləcək?", 
            "answer": "Strategiya çərçivəsində süni intellekt sahəsində ixtisaslaşmış mütəxəssislərin hazırlanması üçün təlim və elmi proqramlar genişləndiriləcək, ali təhsil müəssisələrində süni intellekt üzrə yeni tədris proqramları hazırlanacaq və dövlət qulluqçuları üçün süni intellekt və məlumatların idarə edilməsi üzrə təlimlər keçiriləcəkdir.", 
            "category": "Təhsil və Təlim"
        }
    ]
    
    # Add sample FAQs
    for faq in sample_faqs:
        rag.add_faq(faq["question"], faq["answer"], faq["category"])
    
    # Test a query
    query = "Süni intellekt strategiyasının əsas məqsədləri nədir?"
    print(f"\nQuery: {query}")
    results = rag.retrieve(query)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. Question: {result['question']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Category: {result['category']}")
        print(f"   Similarity: {result['similarity']:.4f}\n")
