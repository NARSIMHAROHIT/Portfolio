"""
Document loader for multiple file types
Supports TXT, PDF, HTML files
"""

from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Container for a loaded document"""
    content: str
    metadata: Dict
    source: str
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'source': self.source
        }


class DocumentLoader:
    """
    Load documents from various file formats
    Supports: TXT, PDF, HTML
    """
    
    def __init__(self):
        """Initialize document loader"""
        self.supported_types = ['.txt', '.pdf', '.html', '.htm']
    
    def load(self, file_path: str) -> Document:
        """
        Load a single document
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.txt':
            content = self._load_txt(path)
        elif suffix == '.pdf':
            content = self._load_pdf(path)
        elif suffix in ['.html', '.htm']:
            content = self._load_html(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        metadata = {
            'filename': path.name,
            'file_type': suffix,
            'file_size': path.stat().st_size,
        }
        
        logger.info(f"Loaded {path.name}: {len(content)} characters")
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(path)
        )
    
    def load_directory(self, directory: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory: Path to directory
            
        Returns:
            List of Document objects
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() in self.supported_types:
                try:
                    doc = self.load(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        
        return documents
    
    def _load_txt(self, path: Path) -> str:
        """Load text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF file with OCR fallback for scanned PDFs"""
        try:
            import PyPDF2
            
            text = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text.append(page_text)
                    else:
                        logger.warning(f"Page {page_num + 1} has no text, attempting OCR...")
                        ocr_text = self._ocr_pdf_page(path, page_num)
                        if ocr_text:
                            text.append(ocr_text)
            
            full_text = '\n\n'.join(text)
            
            if not full_text.strip():
                logger.error(f"No text extracted from {path.name}. This might be a scanned PDF without OCR.")
                raise ValueError("PDF contains no extractable text. Try using OCR or convert to text first.")
            
            return full_text
        
        except ImportError:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    def _ocr_pdf_page(self, path: Path, page_num: int) -> str:
        """Extract text from PDF page using OCR"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            images = convert_from_path(str(path), first_page=page_num + 1, last_page=page_num + 1)
            
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
            
            return ""
        
        except ImportError:
            logger.warning("OCR libraries not installed. Install with: pip install pdf2image pytesseract")
            logger.warning("Also install system dependency: brew install tesseract (Mac) or apt-get install tesseract-ocr (Linux)")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            return ""
    
    def _load_html(self, path: Path) -> str:
        """Load HTML file and extract text"""
        try:
            from bs4 import BeautifulSoup
            
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except ImportError:
            raise ImportError("BeautifulSoup not installed. Install with: pip install beautifulsoup4")


if __name__ == "__main__":
    loader = DocumentLoader()
    
    doc = loader.load("sample.txt")
    print(f"Loaded: {doc.metadata['filename']}")
    print(f"Content length: {len(doc.content)}")
    print(f"Preview: {doc.content[:200]}...")