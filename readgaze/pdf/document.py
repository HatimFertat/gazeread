import logging
from typing import List, Optional, Tuple, Dict
import fitz  # PyMuPDF
import os
import re

class PDFDocument:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)
        self.current_page = 0
        self.total_pages = len(self.doc)
        
        # Multi-page line tracking: dictionary of {page_number: [(y_position, line_text)]}
        self.page_line_positions = {}
        # Multi-page paragraph tracking: dictionary of {page_number: [(y_position, paragraph_text)]}
        self.page_paragraph_positions = {}
        # Dictionary to store reading times for each line: {page_number: {line_text: time}}
        self.page_line_times = {}
        
        logging.info(f"Opened PDF document: {file_path}")
        logging.info(f"Total pages in document: {self.total_pages}")
    
    def get_page_text(self, page_num: int) -> str:
        """Get the text content of a specific page and extract line positions."""
        if not 0 <= page_num < self.total_pages:
            logging.warning(f"Invalid page number: {page_num}")
            return ""
            
        self.current_page = page_num
        
        # Check if we've already processed this page
        if page_num not in self.page_line_positions:
            # Extract text using extract_line_data for this page only
            current_page_lines = self.extract_line_data(page_num)
            
            if not current_page_lines:
                logging.warning(f"No text found on page {page_num}")
                self.page_line_positions[page_num] = []
                self.page_paragraph_positions[page_num] = []
                return ""
                
            # Store line positions for this page with blockno
            # Format: [(y0, text, blockno), ...]
            self.page_line_positions[page_num] = [(line[2], line[5], line[6]) for line in current_page_lines]  # y0, text, blockno
            
            # Process paragraphs for this page - using blocks directly
            self.process_paragraphs(page_num)
            
            # Initialize the line times dictionary for this page
            if page_num not in self.page_line_times:
                self.page_line_times[page_num] = {}
        
        # Return the text by joining all lines for this page
        return "\n".join(line[1] for line in self.page_line_positions[page_num])
    
    def process_paragraphs(self, page_num: int) -> None:
        """Identify paragraphs from lines based on block structure in the PDF."""
        if page_num not in self.page_line_positions:
            return
            
        # Extract blocks directly from PyMuPDF instead of inferring from lines
        page = self.doc[page_num]
        blocks = page.get_text("blocks")
        
        paragraphs = []
        for block in blocks:
            # block format is (x0, y0, x1, y1, "text", block_no, block_type)
            x0, y0, x1, y1, text, block_no, block_type = block
            # Skip empty blocks or non-text blocks
            if len(text.strip()) < 3 or block_type != 0:
                continue
            # Use the y0 position and text
            paragraphs.append((y0, text))
            
        # Store the paragraph positions sorted by y-position
        self.page_paragraph_positions[page_num] = sorted(paragraphs, key=lambda p: p[0])
        logging.info(f"Page {page_num}: Found {len(paragraphs)} paragraphs from {len(blocks)} blocks")
        
    def get_line_positions(self, page_num: int, granularity: str = "line") -> List[Tuple]:
        """
        Get the text positions for a specific page.
        
        Args:
            page_num: The page number to get positions for
            granularity: "line" or "block" to determine what type of positions to return
            
        Returns:
            For line granularity: List of (y_position, text, block_number) tuples
            For block granularity: List of (y_position, text) tuples
        """
        if granularity == "block":
            # Return paragraph positions if available, otherwise process them
            if page_num not in self.page_paragraph_positions:
                # This will trigger paragraph processing
                self.get_page_text(page_num)
                
            paragraphs = self.page_paragraph_positions.get(page_num, [])
            logging.info(f"Page {page_num}: Found {len(paragraphs)} paragraphs")
            return paragraphs
        else:
            # Default to line positions
            return self.page_line_positions.get(page_num, [])

    def get_line_at_position(self, page_num: int, y_position: float) -> Optional[str]:
        """Get the text of the line closest to the given y-position on the specified page."""
        if page_num not in self.page_line_positions or not self.page_line_positions[page_num]:
            return None
            
        # Find the closest line to the given y-position
        line_positions = self.page_line_positions[page_num]
        closest_line = min(line_positions, key=lambda x: abs(x[0] - y_position))
        return closest_line[1]
    
    def get_paragraph_at_position(self, page_num: int, y_position: float) -> Optional[str]:
        """Get the text of the paragraph closest to the given y-position."""
        if page_num not in self.page_paragraph_positions or not self.page_paragraph_positions[page_num]:
            return None
            
        # Find the closest paragraph to the given y-position
        paragraph_positions = self.page_paragraph_positions[page_num]
        closest_paragraph = min(paragraph_positions, key=lambda x: abs(x[0] - y_position))
        return closest_paragraph[1]
    
    def get_context_lines(self, page_num: int, current_line: str, num_lines: int = 3) -> List[Tuple[int, str]]:
        """Get the context lines around the current line, possibly from adjacent pages.
        Returns list of (page_number, line_text) tuples."""
        results = []
        
        # If the page is not processed yet
        if page_num not in self.page_line_positions:
            return []
            
        # Try to find the current line in the current page
        try:
            current_page_lines = self.page_line_positions[page_num]
            current_index = next(i for i, (_, text) in enumerate(current_page_lines) 
                               if text == current_line)
                               
            # Get lines before the current line on the current page
            start_idx = max(0, current_index - num_lines)
            for i in range(start_idx, current_index):
                results.append((page_num, current_page_lines[i][1]))
                
            # Add the current line
            results.append((page_num, current_line))
            
            # Get lines after the current line on the current page
            end_idx = min(len(current_page_lines), current_index + num_lines + 1)
            for i in range(current_index + 1, end_idx):
                results.append((page_num, current_page_lines[i][1]))
                
        except StopIteration:
            # Current line not found on the current page
            pass
            
        # If we couldn't get enough context from the current page, 
        # check previous and next pages
        
        # Check previous page if needed
        if len(results) < 2*num_lines + 1 and page_num > 0:
            prev_page = page_num - 1
            # Load the previous page if not already loaded
            if prev_page not in self.page_line_positions:
                self.get_page_text(prev_page)
                
            if prev_page in self.page_line_positions:
                prev_lines = self.page_line_positions[prev_page]
                # Add lines from the end of previous page
                lines_needed = num_lines - len([l for p, l in results if p == page_num and results.index((p, l)) < results.index((page_num, current_line))])
                if lines_needed > 0 and prev_lines:
                    start_idx = max(0, len(prev_lines) - lines_needed)
                    prev_results = [(prev_page, line) for _, line in prev_lines[start_idx:]]
                    results = prev_results + results
                    
        # Check next page if needed
        if len(results) < 2*num_lines + 1 and page_num < self.total_pages - 1:
            next_page = page_num + 1
            # Load the next page if not already loaded
            if next_page not in self.page_line_positions:
                self.get_page_text(next_page)
                
            if next_page in self.page_line_positions:
                next_lines = self.page_line_positions[next_page]
                # Add lines from the beginning of next page
                lines_needed = num_lines - len([l for p, l in results if p == page_num and results.index((p, l)) > results.index((page_num, current_line))])
                if lines_needed > 0 and next_lines:
                    end_idx = min(lines_needed, len(next_lines))
                    next_results = [(next_page, line) for _, line in next_lines[:end_idx]]
                    results = results + next_results
        
        return results
    
    def get_total_pages(self) -> int:
        """Get the total number of pages in the document."""
        return self.total_pages
    
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
            logging.info(f"Page {page_number}: Extracted {len(words)} words")
            
            # Group words by block and line numbers
            line_groups = {}
            for word in words:
                x0, y0, x1, y1, text, blockno, lineno, wordno = word
                key = (blockno, lineno)
                if key not in line_groups:
                    line_groups[key] = []
                line_groups[key].append(word)
            
            block_line_count = {}
            for (blockno, lineno) in line_groups.keys():
                if blockno not in block_line_count:
                    block_line_count[blockno] = 0
                block_line_count[blockno] += 1
 
            logging.info(f"Page {page_number}: Found {len(block_line_count)} blocks")
            for block, line_count in block_line_count.items():
                logging.info(f"Page {page_number}: Block {block} has {line_count} lines")
            
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
                    line_data = (page_number + 1, x0, y0, x1, y1, line_text.strip(), blockno)
                    all_lines_data.append(line_data)
        
        return all_lines_data
        
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()