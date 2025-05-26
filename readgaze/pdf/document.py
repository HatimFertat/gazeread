import fitz
from typing import List, Optional, Tuple
import logging

class PDFDocument:
    def __init__(self, file_path: str):
        self.doc = fitz.open(file_path)
        self.current_page = 0
        self.line_positions = []  # List of (y_position, line_text) tuples
        logging.info(f"Opened PDF document: {file_path}")
        logging.info(f"Total pages in document: {len(self.doc)}")
        
    def get_page_text(self, page_num: int) -> str:
        """Get the text content of a specific page."""
        if not 0 <= page_num < len(self.doc):
            logging.warning(f"Invalid page number: {page_num}")
            return ""
            
        self.current_page = page_num
        
        # Extract text using extract_line_data for this page only
        current_page_lines = self.extract_line_data(page_num)
        
        if not current_page_lines:
            logging.warning(f"No text found on page {page_num}")
            return ""
            
        # Store line positions for later use
        self.line_positions = [(line[2], line[5]) for line in current_page_lines]  # y0 and text
        
        # Build text by joining lines
        text = "\n".join(line[5] for line in current_page_lines)
        
        return text
        
    def get_line_at_position(self, y_position: float) -> Optional[str]:
        """Get the text of the line closest to the given y-position."""
        if not self.line_positions:
            return None
            
        # Find the closest line to the given y-position
        closest_line = min(self.line_positions, key=lambda x: abs(x[0] - y_position))
        return closest_line[1]
        
    def get_context_lines(self, current_line: str, num_lines: int = 3) -> List[str]:
        """Get the context lines around the current line."""
        if not self.line_positions:
            return []
            
        # Find the index of the current line
        try:
            current_index = next(i for i, (_, text) in enumerate(self.line_positions) 
                               if text == current_line)
        except StopIteration:
            return []
            
        # Get the surrounding lines
        start_idx = max(0, current_index - num_lines)
        end_idx = min(len(self.line_positions), current_index + num_lines + 1)
        
        return [text for _, text in self.line_positions[start_idx:end_idx]]
        
    def get_total_pages(self) -> int:
        """Get the total number of pages in the document."""
        return len(self.doc)
        
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close() 
    def extract_line_data(self, page_num: Optional[int] = None) -> List[Tuple[int, float, float, float, float, str]]:
        """Extract line data with bounding boxes and text from a specific page or the entire document.
        
        Args:
            page_num: If provided, only process this page number (0-based). If None, process all pages.
        """
        all_lines_data = []
        
        # Determine which pages to process
        if page_num is not None:
            if not 0 <= page_num < len(self.doc):
                logging.warning(f"Invalid page number: {page_num}")
                return []
            pages_to_process = [page_num]
        else:
            pages_to_process = range(len(self.doc))
        
        for page_number in pages_to_process:
            page = self.doc[page_number]
            # Get words with their positions and metadata
            words = page.get_text("words")
            
            # Group words by block and line numbers
            line_groups = {}
            for word in words:
                x0, y0, x1, y1, text, blockno, lineno, wordno = word
                key = (blockno, lineno)
                if key not in line_groups:
                    line_groups[key] = []
                line_groups[key].append(word)
            
            # Process each line group
            for (blockno, lineno) in sorted(line_groups.keys()):
                words_in_line = line_groups[(blockno, lineno)]
                
                # Calculate line bounding box
                x0 = min(word[0] for word in words_in_line)
                y0 = min(word[1] for word in words_in_line)
                x1 = max(word[2] for word in words_in_line)
                y1 = max(word[3] for word in words_in_line)
                
                # Sort words by x-coordinate and join their text
                sorted_words = sorted(words_in_line, key=lambda w: w[0])
                line_text = " ".join(word[4] for word in sorted_words)
                
                if line_text.strip():
                    line_data = (page_number + 1, x0, y0, x1, y1, line_text.strip())
                    all_lines_data.append(line_data)
        
        return all_lines_data