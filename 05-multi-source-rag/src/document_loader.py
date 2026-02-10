"""
Enhanced document loader with metadata tracking
Supports PDF, TXT, HTML, CSV, JSON with full source attribution
"""

from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with enhanced metadata"""
    content: str
    metadata: Dict
    source: str
    doc_type: str
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'source': self.source,
            'doc_type': self.doc_type
        }


class MultiSourceLoader:
    """
    Load documents from multiple sources with metadata
    Supports: PDF, TXT, HTML, CSV, JSON
    """
    
    def __init__(self):
        """Initialize loader"""
        self.supported_types = {
            '.txt': self._load_txt,
            '.pdf': self._load_pdf,
            '.html': self._load_html,
            '.htm': self._load_html,
            '.csv': self._load_csv,
            '.json': self._load_json
        }
    
    def load(self, file_path: str, collection: str = 'default') -> List[Document]:
        """
        Load document with metadata
        
        Args:
            file_path: Path to file
            collection: Collection name for organization
            
        Returns:
            List of Document objects (one per page/row)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.supported_types:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        base_metadata = {
            'filename': path.name,
            'file_type': suffix.lstrip('.'),
            'file_size': path.stat().st_size,
            'collection': collection,
            'indexed_at': datetime.now().isoformat()
        }
        
        loader_func = self.supported_types[suffix]
        documents = loader_func(path, base_metadata)
        
        logger.info(f"Loaded {path.name}: {len(documents)} documents")
        
        return documents
    
    def load_directory(self, 
                      directory: str, 
                      collection: str = 'default') -> List[Document]:
        """
        Load all supported files from directory
        
        Args:
            directory: Directory path
            collection: Collection name
            
        Returns:
            List of all documents
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() in self.supported_types:
                try:
                    docs = self.load(str(file_path), collection)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory}")
        
        return all_documents
    
    def _load_txt(self, path: Path, base_metadata: Dict) -> List[Document]:
        """Load text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        metadata = base_metadata.copy()
        
        return [Document(
            content=content,
            metadata=metadata,
            source=str(path),
            doc_type='txt'
        )]
    
    def _load_pdf(self, path: Path, base_metadata: Dict) -> List[Document]:
        """Load PDF with page-level metadata"""
        try:
            import PyPDF2
            
            documents = []
            
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        metadata = base_metadata.copy()
                        metadata['page_number'] = page_num
                        metadata['total_pages'] = len(reader.pages)
                        
                        documents.append(Document(
                            content=text,
                            metadata=metadata,
                            source=f"{path}#page={page_num}",
                            doc_type='pdf'
                        ))
            
            return documents
        
        except ImportError:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    def _load_html(self, path: Path, base_metadata: Dict) -> List[Document]:
        """Load HTML file"""
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
            
            metadata = base_metadata.copy()
            
            title = soup.find('title')
            if title:
                metadata['title'] = title.get_text()
            
            return [Document(
                content=text,
                metadata=metadata,
                source=str(path),
                doc_type='html'
            )]
        
        except ImportError:
            raise ImportError("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    
    def _load_csv(self, path: Path, base_metadata: Dict) -> List[Document]:
        """Load CSV with row-level metadata"""
        documents = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                text_parts = []
                for key, value in row.items():
                    if value:
                        text_parts.append(f"{key}: {value}")
                
                content = "\n".join(text_parts)
                
                metadata = base_metadata.copy()
                metadata['row_number'] = row_num
                metadata['csv_data'] = row
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    source=f"{path}#row={row_num}",
                    doc_type='csv'
                ))
        
        return documents
    
    def _load_json(self, path: Path, base_metadata: Dict) -> List[Document]:
        """Load JSON documents"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data, 1):
                content = json.dumps(item, indent=2)
                
                metadata = base_metadata.copy()
                metadata['item_index'] = idx
                metadata['json_data'] = item
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    source=f"{path}#item={idx}",
                    doc_type='json'
                ))
        
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2)
            
            metadata = base_metadata.copy()
            metadata['json_data'] = data
            
            documents.append(Document(
                content=content,
                metadata=metadata,
                source=str(path),
                doc_type='json'
            ))
        
        return documents


if __name__ == "__main__":
    loader = MultiSourceLoader()
    
    doc = loader.load("sample.txt")
    print(f"Loaded: {doc[0].metadata['filename']}")
    print(f"Type: {doc[0].doc_type}")
    print(f"Content length: {len(doc[0].content)}")