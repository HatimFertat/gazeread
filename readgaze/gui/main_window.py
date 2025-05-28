from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QScrollArea, QLabel, QPushButton, QFileDialog,
                             QMessageBox, QHBoxLayout, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PyQt6.QtGui import QFont, QTextDocument, QPainter, QColor, QPen, QKeyEvent, QPixmap, QImage, QFontMetrics
import os
import time
from ..eye_tracking.tracker import EyeTracker, FilterType
from ..pdf.document import PDFDocument
from ..ai.assistant import AIAssistant
import logging
import cv2
import numpy as np
import re
from typing import Optional, Dict, Tuple, List

#

class GazeIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.gaze_point = None
        self.text_label = None
        self.logger = logging.getLogger(__name__)
        self.scroll_area = None
        # Store detected bounding boxes - now by page
        self.page_screen_bboxes = {}
        # Store PDF line positions and bbox-to-line mapping - now by page
        self.page_line_positions = {}
        self.page_paragraph_positions = {}
        self.page_bbox_line_indices = {}
        # Current page
        self.current_page = 0
        # Granularity setting (line or paragraph)
        self.granularity = "line"
        # Timer for periodic bbox updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_bboxes)
        self.update_timer.start(500)  # Update every 500ms
        
    def set_text_label(self, label):
        """Set the reference to the text label for proper scaling."""
        self.text_label = label
        
    def set_scroll_area(self, scroll_area):
        """Store scroll area reference for viewport updates."""
        self.scroll_area = scroll_area
        # Connect to scrollbar value changes
        scroll_area.verticalScrollBar().valueChanged.connect(self.update_bboxes)
        scroll_area.horizontalScrollBar().valueChanged.connect(self.update_bboxes)
    
    def set_current_page(self, page_num: int):
        """Update the current page number."""
        self.current_page = page_num
        self.update_bboxes()

    def set_line_positions(self, page_num: int, line_positions: list, granularity: str = "line"):
        """
        Receive PDFDocument positions as a list of tuples, sorted by y, for a specific page.
        
        Args:
            page_num: The page number
            line_positions: For lines: List of (y_pos, text, block_num) tuples
                           For blocks: List of (y_pos, text) tuples
            granularity: "line" or "block" to determine where to store the positions
        """
        # Sort positions by PDF y-coordinate
        if granularity == "block":
            self.page_paragraph_positions = self.page_paragraph_positions or {}
            self.page_paragraph_positions[page_num] = sorted(line_positions, key=lambda lt: lt[0])
        else:
            # For lines, extract block mapping information
            sorted_lines = sorted(line_positions, key=lambda lt: lt[0])
            # Store positions without block numbers
            self.page_line_positions[page_num] = [(pos[0], pos[1]) for pos in sorted_lines]
            
            # Create block mapping for lines
            if sorted_lines and len(sorted_lines[0]) > 2:  # If block information is available
                # Create a mapping from line index to block number
                if hasattr(self, 'text_label') and self.text_label is not None:
                    if not hasattr(self.text_label, 'blockMap'):
                        self.text_label.blockMap = []
                    self.text_label.blockMap = [pos[2] for pos in sorted_lines]
                    self.logger.info(f"Page {page_num}: Line to block mapping: {self.text_label.blockMap}")
        
    def set_gaze_point(self, x: int, y: int):
        """Update gaze point and convert to local coordinates."""
        local_pt = self.mapFromGlobal(QPoint(x, y))
        self.gaze_point = (local_pt.x(), local_pt.y())
        self.update()
        
    def set_granularity(self, granularity: str):
        """Update the text detection granularity (line or paragraph)"""
        self.granularity = granularity
        self.update_bboxes()

    def update_bboxes(self):
        """Update text bounding boxes using OpenCV text detection."""
        if not self.text_label:
            return

        # Create QPixmap from the text label
        pixmap = QPixmap(self.text_label.size())
        self.text_label.render(pixmap)

        # Convert QPixmap to OpenCV format
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get white text on black background
        _, binary = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Estimate line height and create appropriate kernel based on granularity
        fm = QFontMetrics(self.text_label.font())
        line_h = fm.lineSpacing()
        
        # Always detect lines, even in block mode (we'll group them later)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.text_label.width() // 2, max(1, int(line_h * 0.3)))
        )
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            morph,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and store line bounding boxes
        line_screen_bboxes = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter using line criteria
            if w > 20 and (line_h * 0.5) < h < (line_h * 1.5):
                # Convert to global screen coordinates
                global_pos = self.text_label.mapToGlobal(QPoint(x, y))
                line_screen_bboxes.append(QRect(
                    self.mapFromGlobal(global_pos),
                    QSize(w, h)
                ))

        # Sort the lines by vertical position
        line_screen_bboxes.sort(key=lambda r: r.y())
        
        # Merge adjacent lines for better line detection
        if line_screen_bboxes:
            n = len(line_screen_bboxes)
            if n > 1:
                tops = [r.y() for r in line_screen_bboxes]
                bottoms = [r.y() + r.height() for r in line_screen_bboxes]
                
                # midpoints between adjacent boxes
                mids = [(bottoms[i] + tops[i+1]) // 2 for i in range(n-1)]
                merged_lines = []
                for i, r in enumerate(line_screen_bboxes):
                    x, y, w, h = r.x(), r.y(), r.width(), r.height()
                    if i == 0:
                        # first line: extend from its top to midpoint with second line
                        new_top = r.y()
                        new_bot = mids[0]
                        merged_lines.append(QRect(x, new_top, w, new_bot - new_top))
                    elif i == n - 1:
                        new_top = mids[i-1]
                        new_bot = r.y() + r.height()
                        merged_lines.append(QRect(x, new_top, w, new_bot - new_top))
                    else:
                        # middle lines: between adjacent midpoints
                        new_top = mids[i-1]
                        new_bot = mids[i]
                        merged_lines.append(QRect(x, new_top, w, new_bot - new_top))
                line_screen_bboxes = merged_lines
        
        # Now that we have line bounding boxes, either use them directly or group them by block
        if self.granularity == "block":
            # Get the line positions with block information from PDFDocument
            # We need to find out which lines belong to which block
            self.page_screen_bboxes[self.current_page] = []
            
            # Get block information from PDFDocument
            if hasattr(self.text_label, 'blockMap') and self.text_label.blockMap and len(self.text_label.blockMap) > 0:
                # If we already have block mapping information
                block_line_mapping = self.text_label.blockMap
                
                # Group line bounding boxes by block
                blocks = {}
                for line_idx, line_bbox in enumerate(line_screen_bboxes):
                    # Make sure line_idx is within bounds of our mapping
                    if line_idx < len(block_line_mapping):
                        block_num = block_line_mapping[line_idx]
                        if block_num not in blocks:
                            blocks[block_num] = []
                        blocks[block_num].append(line_bbox)
                    
                # Create a bounding box for each block by merging its lines
                for block_num, lines in blocks.items():
                    if lines:
                        # Get extremes
                        left = min(box.left() for box in lines)
                        top = min(box.top() for box in lines)
                        right = max(box.right() for box in lines)
                        bottom = max(box.bottom() for box in lines)
                        
                        # Create block bounding box
                        block_bbox = QRect(left, top, right - left, bottom - top)
                        self.page_screen_bboxes[self.current_page].append(block_bbox)
                
                self.logger.info(f"Page {self.current_page}: Created {len(self.page_screen_bboxes[self.current_page])} block boxes from {len(blocks)} blocks")
            else:
                # Fallback: If we don't have block mapping, use the PDF paragraph positions
                if self.current_page in self.page_paragraph_positions:
                    # Get paragraph positions
                    pdf_blocks = self.page_paragraph_positions[self.current_page]
                    
                    # Create a block for each paragraph position by finding lines that belong to it
                    if pdf_blocks:
                        pdf_block_centers = [(pos[0] + 10) for pos in pdf_blocks]  # Approximate center Y
                        
                        # Group lines by closest paragraph
                        grouped_lines = [[] for _ in range(len(pdf_blocks))]
                        
                        for line_bbox in line_screen_bboxes:
                            line_center = line_bbox.y() + line_bbox.height() / 2
                            # Find closest PDF block
                            distances = [abs(line_center - (y_pos * 10)) for y_pos in pdf_block_centers]  # Scale factor
                            closest_block = distances.index(min(distances))
                            grouped_lines[closest_block].append(line_bbox)
                        
                        # Create bounding box for each group
                        for block_idx, lines in enumerate(grouped_lines):
                            if lines:
                                # Get extremes
                                left = min(box.left() for box in lines)
                                top = min(box.top() for box in lines)
                                right = max(box.right() for box in lines)
                                bottom = max(box.bottom() for box in lines)
                                
                                # Create block bounding box
                                block_bbox = QRect(left, top, right - left, bottom - top)
                                self.page_screen_bboxes[self.current_page].append(block_bbox)
                        
                        self.logger.info(f"Page {self.current_page}: Created {len(self.page_screen_bboxes[self.current_page])} block boxes from PDF blocks")
                
                # If we still don't have blocks, add a marker to be improved next time
                if not self.page_screen_bboxes[self.current_page]:
                    self.logger.warning(f"Page {self.current_page}: No block information available, using line bounding boxes")
                    self.page_screen_bboxes[self.current_page] = line_screen_bboxes
        else:
            # Line granularity - use the line bounding boxes directly
            self.page_screen_bboxes[self.current_page] = line_screen_bboxes

        # Map each detected screen bbox to its PDF line/block index
        if self.current_page in self.page_line_positions:
            # Get the appropriate positions based on granularity
            if self.granularity == "block" and self.current_page in self.page_paragraph_positions:
                # For blocks: Map detected screen boxes to PDF blocks
                positions = self.page_paragraph_positions[self.current_page]
                if positions and self.page_screen_bboxes[self.current_page]:
                    # Create mapping from screen bbox to paragraph index by finding the best match
                    bbox_line_indices = []
                    for bbox in self.page_screen_bboxes[self.current_page]:
                        # Get bbox center y-coordinate
                        bbox_center_y = bbox.y() + bbox.height() / 2
                        
                        # Find the closest paragraph by vertical position
                        closest_idx = 0
                        min_distance = float('inf')
                        for i, (pdf_y, _) in enumerate(positions):
                            # Normalize PDF y-coordinate to screen coordinates 
                            # (approximate, depends on rendering scale)
                            distance = abs(bbox_center_y - pdf_y * 10)  # Scale factor may need adjustment
                            if distance < min_distance:
                                min_distance = distance
                                closest_idx = i
                        
                        bbox_line_indices.append(closest_idx)
                    
                    self.logger.info(f"Page {self.current_page}: Block mapping: {bbox_line_indices}")
                    self.page_bbox_line_indices[self.current_page] = bbox_line_indices
                else:
                    self.page_bbox_line_indices[self.current_page] = []
            else:
                # For lines: Use a simpler mapping - each screen bbox maps to the line with the same index
                positions = self.page_line_positions[self.current_page]
                # Create 1:1 mapping, truncating to the shortest length
                self.page_bbox_line_indices[self.current_page] = list(range(min(len(self.page_screen_bboxes[self.current_page]), len(positions))))

        self.update()
        
    def paintEvent(self, event):
        """Draw detected text boxes and gaze point."""
        if not self.gaze_point:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw text boxes for current page
        if self.current_page in self.page_screen_bboxes:
            rect_pen = QPen()
            rect_pen.setWidth(2)
            
            # Different color based on granularity
            if self.granularity == "block":
                rect_pen.setColor(QColor(0, 100, 200))  # Blue for blocks
            else:
                rect_pen.setColor(QColor(0, 180, 0))    # Green for lines
                
            painter.setPen(rect_pen)

            bboxes = self.page_screen_bboxes[self.current_page]
            # self.logger.info(f"Page {self.current_page}: Drawing {len(bboxes)} boxes")
            # for i, bbox in enumerate(bboxes):
                # self.logger.info(f"Box {i}: (x={bbox.x()}, y={bbox.y()}, w={bbox.width()}, h={bbox.height()})")
            for bbox in bboxes:
                painter.drawRect(bbox)
                
                # For blocks, add extra visual cue
                if self.granularity == "block":
                    painter.fillRect(
                        QRect(bbox.x(), bbox.y(), 5, bbox.height()),
                        QColor(0, 100, 200, 50)  # Semi-transparent blue marker
                    )
            
        # Draw gaze point
        pen = QPen()
        pen.setWidth(10)
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        x, y = self.gaze_point
        painter.drawPoint(x, y)

    def get_line_index_at_gaze(self) -> Optional[Tuple[int, int]]:
        """Return the (page_num, line_index) corresponding to the current gaze point."""
        if not (self.gaze_point and self.current_page in self.page_screen_bboxes):
            return None
            
        screen_bboxes = self.page_screen_bboxes[self.current_page]
        if not screen_bboxes:
            return None
            
        pt = QPoint(self.gaze_point[0], self.gaze_point[1])
        for idx, bbox in enumerate(screen_bboxes):
            if bbox.contains(pt):
                return (self.current_page, idx)
        return None

    def get_line_text_at_gaze(self) -> Optional[Tuple[int, str]]:
        """Return the (page_num, text) at the current gaze point."""
        line_info = self.get_line_index_at_gaze()
        if line_info is None:
            return None
            
        page_num, line_idx = line_info
        
        # Use the appropriate positions based on granularity
        if self.granularity == "block" and page_num in self.page_paragraph_positions:
            positions = self.page_paragraph_positions[page_num]
        else:
            positions = self.page_line_positions.get(page_num, [])
            
        if positions and line_idx < len(positions):
            return (page_num, positions[line_idx][1])
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.setWindowTitle("ReadGaze")
        self.setMinimumSize(1024, 768)  # Increased default size
        
        # Initialize components
        self.eye_tracker = EyeTracker()
        self.pdf_document = None
        self.current_page = 0
        self.is_fullscreen = False
        
        # Setup UI
        self.setup_ui()
        
        # Setup eye tracking timer
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self.update_eye_position)
        self.tracking_timer.start(50)  # 20Hz update rate
        
        # Reading state - now with page tracking
        self.reading_start_time = None
        self.current_line = None
        self.current_line_page = None
        self.line_start_time = None
        self.line_reading_time = 0
        self.page_line_times = {}  # Dictionary to store reading times for each line by page {page: {line: time}}
        
        # Track page positions in the scrollable area
        self.page_y_positions = []  # List of (start_y, end_y) for each page
        
        # Try to load existing calibration
        self.load_calibration()
        self.logger.info("MainWindow initialized")
        
    def resizeEvent(self, event):
        """Handle window resize events."""
        if hasattr(self, 'gaze_indicator'):
            self.gaze_indicator.setGeometry(self.centralWidget().rect())
        super().resizeEvent(event)
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Toolbar
        self.toolbar = QWidget()  # Store reference to toolbar
        toolbar_layout = QHBoxLayout(self.toolbar)
        
        # Open file button
        open_button = QPushButton("Open PDF")
        open_button.clicked.connect(self.open_pdf)
        toolbar_layout.addWidget(open_button)
        
        # Calibration button
        self.calibrate_button = QPushButton("Calibrate Eye Tracker")
        self.calibrate_button.clicked.connect(self.calibrate_eye_tracker)
        toolbar_layout.addWidget(self.calibrate_button)
        
        # Filter selection
        filter_label = QLabel("Filter:")
        toolbar_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "kalman", "kde"])
        self.filter_combo.currentTextChanged.connect(self.change_filter)
        toolbar_layout.addWidget(self.filter_combo)
        
        # Font size control
        font_label = QLabel("Font Size:")
        toolbar_layout.addWidget(font_label)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(12, 32)
        self.font_size_spin.setValue(20)
        self.font_size_spin.valueChanged.connect(self.change_font_size)
        toolbar_layout.addWidget(self.font_size_spin)
        
        # Granularity control
        granularity_label = QLabel("Granularity:")
        toolbar_layout.addWidget(granularity_label)
        
        self.granularity_combo = QComboBox()
        self.granularity_combo.addItems(["line", "block"])
        self.granularity_combo.currentTextChanged.connect(self.change_granularity)
        toolbar_layout.addWidget(self.granularity_combo)
        
        # Enhanced page indicator
        self.page_indicator_widget = QWidget()
        page_indicator_layout = QHBoxLayout(self.page_indicator_widget)
        page_indicator_layout.setContentsMargins(10, 0, 10, 0)
        
        page_icon_label = QLabel("📄")
        page_icon_label.setStyleSheet("font-size: 16px;")
        page_indicator_layout.addWidget(page_icon_label)
        
        self.page_label = QLabel("Page: 0/0")
        self.page_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        page_indicator_layout.addWidget(self.page_label)
        
        toolbar_layout.addWidget(self.page_indicator_widget)
        
        # Fullscreen button
        self.fullscreen_button = QPushButton("Fullscreen (Enter)")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        toolbar_layout.addWidget(self.fullscreen_button)
        
        # Status label
        self.status_label = QLabel("Eye tracker: Not connected")
        toolbar_layout.addWidget(self.status_label)
        
        toolbar_layout.addStretch()
        layout.addWidget(self.toolbar)
        
        # Content area - text display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.on_scroll_changed)
        
        self.text_content_widget = QWidget()
        self.text_content_layout = QVBoxLayout(self.text_content_widget)
        self.text_content_layout.setContentsMargins(50, 20, 50, 20)
        self.text_content_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        # This will contain all page text labels, one below the other
        self.page_labels = []
        
        self.scroll_area.setWidget(self.text_content_widget)
        
        # Add scroll area to main layout
        layout.addWidget(self.scroll_area)
        
        # Add gaze indicator to the central widget
        self.gaze_indicator = GazeIndicator(self.centralWidget())
        self.gaze_indicator.setGeometry(self.centralWidget().rect())
        # We'll set the text label reference when we load the PDF
        self.gaze_indicator.set_scroll_area(self.scroll_area)
        self.gaze_indicator.raise_()
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if not self.is_fullscreen:
            # Store current window state
            self.normal_geometry = self.geometry()
            # Hide toolbar and enter fullscreen
            self.toolbar.hide()
            self.showFullScreen()
            self.is_fullscreen = True
            self.status_label.setText("Press Enter to exit fullscreen")
        else:
            # Exit fullscreen and restore toolbar
            self.showNormal()
            self.setGeometry(self.normal_geometry)
            self.toolbar.show()
            self.is_fullscreen = False
            self.status_label.setText("Eye tracker: Connected")
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events."""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Space:
            # Page down on space
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() + self.scroll_area.height() - 100
            )
        elif event.key() == Qt.Key.Key_Backspace:
            # Page up on backspace
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - self.scroll_area.height() + 100
            )
        else:
            super().keyPressEvent(event)
            
    def change_filter(self, filter_type: str):
        """Change the eye tracking filter."""
        try:
            if self.eye_tracker.set_filter(filter_type):
                self.status_label.setText(f"Filter: {filter_type}")
            else:
                QMessageBox.warning(self, "Error", "Failed to change filter. Please recalibrate.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to change filter: {e}")
            
    def change_font_size(self, size: int):
        """Change the font size of the text."""
        if not self.page_labels:
            return
            
        for label in self.page_labels:
            font = label.font()
            font.setPointSize(size)
            label.setFont(font)
            
        # Update bounding boxes after font size change
        self.calculate_page_positions()
        self.gaze_indicator.update_bboxes()
        
    def change_granularity(self, granularity: str):
        """Change the text detection granularity."""
        if hasattr(self, 'gaze_indicator'):
            self.gaze_indicator.set_granularity(granularity)
            self.status_label.setText(f"Granularity: {granularity}")
            
            # Update the UI to indicate the current granularity mode
            self.granularity_combo.setStyleSheet(f"background-color: {'#d1e7ff' if granularity == 'block' else '#d1ffd1'}")
            
            # Force bbox update
            self.gaze_indicator.update_bboxes()
        
    def load_calibration(self):
        """Try to load existing calibration model."""
        model_path = os.path.expanduser("~/.readgaze/gaze_model.pkl")
        if os.path.exists(model_path):
            if self.eye_tracker.load_model(model_path):
                self.status_label.setText("Eye tracker: Connected and calibrated")
                return
        self.status_label.setText("Eye tracker: Connected (needs calibration)")
        
    def save_calibration(self):
        """Save the current calibration model."""
        os.makedirs(os.path.expanduser("~/.readgaze"), exist_ok=True)
        model_path = os.path.expanduser("~/.readgaze/gaze_model.pkl")
        if self.eye_tracker.save_model(model_path):
            self.status_label.setText("Eye tracker: Connected and calibrated")
            
    def calibrate_eye_tracker(self):
        """Start the eye tracker calibration process."""
        if not self.eye_tracker.is_connected():
            QMessageBox.warning(self, "Error", "Eye tracker is not connected")
            return
        
        # Start calibration
        if self.eye_tracker.calibrate():
            self.save_calibration()
            QMessageBox.information(self, "Success", "Calibration completed successfully!")
        else:
            QMessageBox.warning(self, "Error", "Calibration failed. Please try again.")
            
    def open_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDF File",
            "",
            "PDF Files (*.pdf)"
        )
        
        if file_name:
            try:
                self.pdf_document = PDFDocument(file_name)
                self.current_page = 0
                
                # Clear any previous page labels
                for label in self.page_labels:
                    self.text_content_layout.removeWidget(label)
                    label.deleteLater()
                self.page_labels = []
                
                # Initialize page_line_times for all pages
                self.page_line_times = {}
                for page in range(self.pdf_document.get_total_pages()):
                    self.page_line_times[page] = {}
                
                # Load all pages
                self.load_all_pages()
                
                # Update page indicator
                self.page_label.setText(f"Page: {self.current_page + 1}/{self.pdf_document.get_total_pages()}")
                
                self.logger.info(f"Successfully opened PDF: {file_name}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open PDF: {str(e)}")
                self.logger.error(f"Failed to open PDF {file_name}: {str(e)}")
    
    def load_all_pages(self):
        """Load all pages of the PDF document at once."""
        if not self.pdf_document:
            return
        
        # Create labels for each page
        for page_num in range(self.pdf_document.get_total_pages()):
            text = self.pdf_document.get_page_text(page_num)
            if not text:
                text = f"[No text content found on page {page_num + 1}]"
                
            # Format the text with proper spacing and line breaks
            formatted_text = text.replace('\n', '<br>')
            formatted_text = f'''
                <div style="
                    font-family: Arial;
                    line-height: 1.4;
                    color: black;
                    text-align: left;
                    width: 100%;
                    padding: 40px 60px;
                    margin-bottom: 20px;
                    border-bottom: 3px solid #ccc;
                ">
                    {formatted_text}
                </div>
            '''
            
            # Create the label for this page
            label = QLabel()
            label.setWordWrap(False)
            label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            label.setFont(QFont("Arial", self.font_size_spin.value()))
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    padding: 40px 60px;
                    line-height: 1.4;
                    color: black;
                    margin-bottom: 40px;
                }
            """)
            label.setMinimumWidth(800)
            label.setText(formatted_text)
            
            # Add a property to store block mapping
            label.blockMap = []
            
            # Add to layout
            self.text_content_layout.addWidget(label)
            self.page_labels.append(label)
            
            # Set the reference to this page's label for the gaze indicator before getting positions
            self.gaze_indicator.set_text_label(label)
            
            # Get line positions for this page (now includes block numbers)
            line_positions = self.pdf_document.get_line_positions(page_num, "line")
            self.gaze_indicator.set_line_positions(page_num, line_positions, "line")
            
            # Also get block positions
            block_positions = self.pdf_document.get_line_positions(page_num, "block")
            self.gaze_indicator.set_line_positions(page_num, block_positions, "block")
        
        # Calculate page positions in the scroll area
        self.calculate_page_positions()
        
    def calculate_page_positions(self):
        """Calculate the Y positions of each page in the scroll area."""
        self.page_y_positions = []
        current_y = 0
        
        for i, label in enumerate(self.page_labels):
            # Get the actual height of the label with all its content
            height = label.sizeHint().height()
            y_start = current_y
            y_end = y_start + height
            self.page_y_positions.append((y_start, y_end))
            # Update current_y for the next page
            current_y = y_end + 40  # Add margin between pages
            
        # Log page positions for debugging
        self.logger.info(f"Page positions: {self.page_y_positions}")

    def on_scroll_changed(self, value):
        """Handle scroll position changes to update current page."""
        if not self.page_y_positions or not self.pdf_document:
            return
            
        # Find which page is most visible in the viewport
        viewport_top = value
        viewport_height = self.scroll_area.viewport().height()
        viewport_bottom = viewport_top + viewport_height
        
        # Calculate visibility percentage for each page
        max_visibility = 0
        most_visible_page = self.current_page  # Default to current
        
        for i, (y_start, y_end) in enumerate(self.page_y_positions):
            # Skip if page is completely outside viewport
            if y_end < viewport_top or y_start > viewport_bottom:
                continue
                
            # Calculate how much of the page is visible
            visible_top = max(viewport_top, y_start)
            visible_bottom = min(viewport_bottom, y_end)
            visible_height = visible_bottom - visible_top
            
            # Calculate as percentage of viewport height
            visibility_percent = (visible_height / viewport_height) * 100
            
            self.logger.debug(f"Page {i+1} visibility: {visibility_percent:.1f}%")
            
            # Update most visible page
            if visibility_percent > max_visibility:
                max_visibility = visibility_percent
                most_visible_page = i
        
        # Update current page if changed
        if most_visible_page != self.current_page:
            self.current_page = most_visible_page
            self.page_label.setText(f"Page: {most_visible_page + 1}/{self.pdf_document.get_total_pages()}")
            # Make page label briefly flash to show page change
            orig_style = self.page_label.styleSheet()
            self.page_label.setStyleSheet("font-weight: bold; min-width: 80px; color: #2060D0;")
            QTimer.singleShot(300, lambda: self.page_label.setStyleSheet(orig_style))
            # Update gaze indicator's reference to current page
            self.gaze_indicator.set_current_page(most_visible_page)
            
            # Ensure label has blockMap property before we set it as text_label
            if not hasattr(self.page_labels[most_visible_page], 'blockMap'):
                self.page_labels[most_visible_page].blockMap = []
                
            self.gaze_indicator.set_text_label(self.page_labels[most_visible_page])
            self.gaze_indicator.update_bboxes()
        
    def update_eye_position(self):
        """Update eye position and detect which line is being read."""
        if not (self.eye_tracker.is_connected() and self.pdf_document):
            return
            
        gaze_data = self.eye_tracker.get_gaze_point()
        if not gaze_data:
            self.logger.debug("No gaze data received")
            self._reset_line_tracking()
            return

        gaze_point, is_blinking = gaze_data
        self.logger.debug(f"Gaze point: {gaze_point}, Blinking: {is_blinking}")
        
        # If blinking, reset line tracking
        if is_blinking:
            self.logger.debug("Blink detected")
            self._reset_line_tracking()
            return

        # Update gaze indicator with absolute screen coordinates
        self.gaze_indicator.set_gaze_point(gaze_point[0], gaze_point[1])
        
        # Determine and handle line change based on gaze
        line_info = self.gaze_indicator.get_line_text_at_gaze()
        if line_info is not None:
            page_num, line_text = line_info
            self._handle_line_change(page_num, line_text)
        else:
            self._reset_line_tracking()
            
    def _reset_line_tracking(self):
        """Reset line tracking state."""
        # Treat as moving away from any line
        if self.current_line is not None and self.current_line_page is not None:
            self._handle_line_change(None, None)
            
    def _handle_line_change(self, page_num: Optional[int], new_line: Optional[str]):
        """Manage transitions between lines and accumulate reading times."""
        now = time.time()
        
        # If gaze moved to a different line or page
        if new_line != self.current_line or page_num != self.current_line_page:
            # Finalize timing for previous line
            if self.current_line is not None and self.current_line_page is not None and self.line_start_time is not None:
                duration = now - self.line_start_time
                if self.current_line_page not in self.page_line_times:
                    self.page_line_times[self.current_line_page] = {}
                self.page_line_times[self.current_line_page][self.current_line] = (
                    self.page_line_times.get(self.current_line_page, {}).get(self.current_line, 0) + duration
                )
                self.logger.info(
                    f"Accumulated time on line (Page {self.current_line_page+1}) '{self.current_line[:50]}...': "
                    f"{self.page_line_times[self.current_line_page][self.current_line]:.2f}s"
                )
            
            # Start timing for new line if any
            if new_line is not None and page_num is not None:
                self.line_start_time = now
                self.current_line = new_line
                self.current_line_page = page_num
                # Initialize the line time if needed
                if page_num not in self.page_line_times:
                    self.page_line_times[page_num] = {}
                if new_line not in self.page_line_times[page_num]:
                    self.page_line_times[page_num][new_line] = 0
            else:
                self.line_start_time = None
                self.current_line = None
                self.current_line_page = None
        
    def show_explanation(self):
        """Show AI explanation for the current line."""
        if not self.current_line or not self.pdf_document or self.current_line_page is None:
            return
            
        context_lines = self.pdf_document.get_context_lines(self.current_line_page, self.current_line)
        # Extract just the text from the (page, text) tuples
        context_text = [text for _, text in context_lines]
        
        # Get accumulated reading time for this line
        line_time = self.page_line_times.get(self.current_line_page, {}).get(self.current_line, 0)
        
        explanation = self.ai_assistant.get_explanation(
            self.current_line,
            context_text,
            line_time
        )
        
        self.explanation_label.setText(explanation)
        self.explanation_label.show()
        
    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        if self.eye_tracker:
            self.eye_tracker.disconnect()
        if self.pdf_document:
            self.pdf_document.close()
        event.accept() 