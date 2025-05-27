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
        self.page_bbox_line_indices = {}
        # Current page
        self.current_page = 0
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

    def set_line_positions(self, page_num: int, line_positions: list[tuple[float, str]]):
        """
        Receive PDFDocument.page_line_positions as a list of (y_pos, text),
        sorted by y, for a specific page.
        """
        # Sort positions by PDF y-coordinate
        self.page_line_positions[page_num] = sorted(line_positions, key=lambda lt: lt[0])
        
    def set_gaze_point(self, x: int, y: int):
        """Update gaze point and convert to local coordinates."""
        local_pt = self.mapFromGlobal(QPoint(x, y))
        self.gaze_point = (local_pt.x(), local_pt.y())
        self.update()
        
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
        # Estimate line height and create a kernel to merge characters into full-line blobs
        fm = QFontMetrics(self.text_label.font())
        line_h = fm.lineSpacing()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.text_label.width() // 2, max(1, int(line_h * 0.3)))
        )
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Find line-level contours
        contours, _ = cv2.findContours(
            morph,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and store bounding boxes
        screen_bboxes = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out noise (too small boxes), target line height
            if w > 20 and (line_h * 0.5) < h < (line_h * 1.5):
                # Convert to global screen coordinates
                global_pos = self.text_label.mapToGlobal(QPoint(x, y))
                screen_bboxes.append(QRect(
                    self.mapFromGlobal(global_pos),
                    QSize(w, h)
                ))

        # Merge vertical regions halfway between adjacent lines
        boxes = sorted(screen_bboxes, key=lambda r: r.y())
        n = len(boxes)
        if n > 1:
            # compute current tops and bottoms
            tops = [r.y() for r in boxes]
            bottoms = [r.y() + r.height() for r in boxes]
            # midpoints between adjacent boxes
            mids = [(bottoms[i] + tops[i+1]) // 2 for i in range(n-1)]
            merged = []
            for i, r in enumerate(boxes):
                x, y, w, h = r.x(), r.y(), r.width(), r.height()
                if i == 0:
                    # first line: extend from its top to midpoint with second line
                    new_top = r.y()
                    new_bot = mids[0]
                    merged.append(QRect(x, new_top, w, new_bot - new_top))
                elif i == n - 1:
                    new_top = mids[i-1]
                    new_bot = r.y() + r.height()
                    merged.append(QRect(x, new_top, w, new_bot - new_top))
                else:
                    # middle lines: between adjacent midpoints
                    new_top = mids[i-1]
                    new_bot = mids[i]
                    merged.append(QRect(x, new_top, w, new_bot - new_top))
            screen_bboxes = merged

        # Store the bounding boxes for the current page
        self.page_screen_bboxes[self.current_page] = screen_bboxes

        # Map each detected screen bbox to its PDF line index by vertical order for current page
        if self.current_page in self.page_line_positions:
            screen_bboxes.sort(key=lambda r: r.y())
            self.page_bbox_line_indices[self.current_page] = list(range(len(screen_bboxes)))

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
            rect_pen.setColor(QColor(0, 255, 0))
            painter.setPen(rect_pen)
            
            for bbox in self.page_screen_bboxes[self.current_page]:
                painter.drawRect(bbox)
            
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
        if page_num in self.page_line_positions and line_idx < len(self.page_line_positions[page_num]):
            return (page_num, self.page_line_positions[page_num][line_idx][1])
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
            
            # Add to layout
            self.text_content_layout.addWidget(label)
            self.page_labels.append(label)
            
            # Get line positions for this page
            line_positions = self.pdf_document.get_line_positions(page_num)
            self.gaze_indicator.set_line_positions(page_num, line_positions)
        
        # Calculate page positions in the scroll area
        self.calculate_page_positions()
        
        # Set the reference to the current page's label for the gaze indicator
        if self.page_labels:
            self.gaze_indicator.set_text_label(self.page_labels[0])
    
    def calculate_page_positions(self):
        """Calculate the Y positions of each page in the scroll area."""
        self.page_y_positions = []
        for i, label in enumerate(self.page_labels):
            if i > 0:
                # Get the position from the previous page
                prev_start, prev_end = self.page_y_positions[i-1]
                y_start = prev_end + 40  # Add margin between pages
            else:
                y_start = 0
                
            y_end = y_start + label.height()
            self.page_y_positions.append((y_start, y_end))
    
    def on_scroll_changed(self, value):
        """Handle scroll position changes to update current page."""
        if not self.page_y_positions:
            return
            
        # Find which page is most visible in the viewport
        viewport_top = value
        viewport_bottom = value + self.scroll_area.viewport().height()
        viewport_middle = (viewport_top + viewport_bottom) / 2
        
        # Find which page contains the middle of the viewport
        for i, (y_start, y_end) in enumerate(self.page_y_positions):
            if y_start <= viewport_middle <= y_end:
                if i != self.current_page:
                    self.current_page = i
                    self.page_label.setText(f"Page: {i + 1}/{self.pdf_document.get_total_pages()}")
                    # Make page label briefly flash to show page change
                    orig_style = self.page_label.styleSheet()
                    self.page_label.setStyleSheet("font-weight: bold; min-width: 80px; color: #2060D0;")
                    QTimer.singleShot(300, lambda: self.page_label.setStyleSheet(orig_style))
                    # Update gaze indicator's reference to current page
                    self.gaze_indicator.set_current_page(i)
                    self.gaze_indicator.set_text_label(self.page_labels[i])
                    self.gaze_indicator.update_bboxes()
                break
                
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