from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QScrollArea, QLabel, QPushButton, QFileDialog,
                             QMessageBox, QHBoxLayout, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PyQt6.QtGui import QFont, QTextDocument, QPainter, QColor, QPen, QKeyEvent, QPixmap, QImage, QFontMetrics
import fitz  # PyMuPDF
import os
import time
from ..eye_tracking.tracker import EyeTracker, FilterType
from ..pdf.document import PDFDocument
from ..ai.assistant import AIAssistant
import logging
import cv2
import numpy as np
import re
from typing import Optional

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
        # Store detected bounding boxes
        self.screen_bboxes = []
        # Store PDF line positions and bbox-to-line mapping
        self.line_positions = []
        self.bbox_line_indices = []
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

    def set_line_positions(self, line_positions: list[tuple[float, str]]):
        """
        Receive PDFDocument.line_positions as a list of (y_pos, text),
        sorted by y.
        """
        # Sort positions by PDF y-coordinate
        self.line_positions = sorted(line_positions, key=lambda lt: lt[0])
        
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
        self.screen_bboxes = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out noise (too small boxes), target line height
            if w > 20 and (line_h * 0.5) < h < (line_h * 1.5):
                # Convert to global screen coordinates
                global_pos = self.text_label.mapToGlobal(QPoint(x, y))
                self.screen_bboxes.append(QRect(
                    self.mapFromGlobal(global_pos),
                    QSize(w, h)
                ))

        # Merge vertical regions halfway between adjacent lines
        boxes = sorted(self.screen_bboxes, key=lambda r: r.y())
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
            self.screen_bboxes = merged

        # Map each detected screen bbox to its PDF line index by vertical order
        # Assumes one-to-one count with line_positions
        self.screen_bboxes.sort(key=lambda r: r.y())
        self.bbox_line_indices = list(range(len(self.screen_bboxes)))

        self.update()
        
    def paintEvent(self, event):
        """Draw detected text boxes and gaze point."""
        if not (self.screen_bboxes or self.gaze_point):
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw text boxes
        rect_pen = QPen()
        rect_pen.setWidth(2)
        rect_pen.setColor(QColor(0, 255, 0))
        painter.setPen(rect_pen)
        
        for bbox in self.screen_bboxes:
            painter.drawRect(bbox)
            
        # Draw gaze point
        if self.gaze_point:
            pen = QPen()
            pen.setWidth(10)
            pen.setColor(QColor(255, 0, 0))
            painter.setPen(pen)
            x, y = self.gaze_point
            painter.drawPoint(x, y)

    def get_line_index_at_gaze(self) -> Optional[int]:
        """Return the PDF line index corresponding to the current gaze point."""
        if not (self.gaze_point and self.screen_bboxes and self.bbox_line_indices):
            return None
        pt = QPoint(self.gaze_point[0], self.gaze_point[1])
        for idx, bbox in zip(self.bbox_line_indices, self.screen_bboxes):
            if bbox.contains(pt):
                return idx
        return None

    def get_line_text_at_gaze(self) -> Optional[str]:
        """Return the PDF line text at the current gaze point."""
        idx = self.get_line_index_at_gaze()
        if idx is not None and idx < len(self.line_positions):
            return self.line_positions[idx][1]
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
        
        # Reading state
        self.reading_start_time = None
        self.current_line = None
        self.line_start_time = None
        self.line_reading_time = 0
        self.line_times = {}  # Dictionary to store reading times for each line
        
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
        
        # Fullscreen button
        self.fullscreen_button = QPushButton("Fullscreen (Enter)")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        toolbar_layout.addWidget(self.fullscreen_button)
        
        # Status label
        self.status_label = QLabel("Eye tracker: Not connected")
        toolbar_layout.addWidget(self.status_label)
        
        toolbar_layout.addStretch()
        layout.addWidget(self.toolbar)
        
        # PDF viewer
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(50, 20, 50, 20)  # Add margins for better appearance
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # Center horizontally
        
        self.text_label = QLabel()
        self.text_label.setWordWrap(False)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.text_label.setFont(QFont("Arial", 24))
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 40px 60px;
                line-height: 1.4;
                color: black;
            }
        """)
        
        # Set minimum width for the text label to ensure proper sizing
        self.text_label.setMinimumWidth(800)
        
        self.content_layout.addWidget(self.text_label)
        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)
        
        # Add gaze indicator to the central widget instead of viewport
        self.gaze_indicator = GazeIndicator(self.centralWidget())
        self.gaze_indicator.setGeometry(self.centralWidget().rect())
        self.gaze_indicator.set_text_label(self.text_label)  # Set the text label reference
        self.gaze_indicator.set_scroll_area(self.scroll_area)
        self.gaze_indicator.raise_()  # Ensure the gaze indicator stays on top
        
        # Removed resize event patching for text lines
        
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
        font = self.text_label.font()
        font.setPointSize(size)
        self.text_label.setFont(font)
        
    def change_model(self, model_name: str):
        """Change the AI model being used."""
        try:
            self.ai_assistant.set_model(model_name)
            self.status_label.setText(f"AI Model: {model_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to change model: {e}")
            # Reset combo box to previous value
            self.model_combo.setCurrentText(self.ai_assistant.model_name)
        
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
                self.display_current_page()
                self.logger.info(f"Successfully opened PDF: {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open PDF: {str(e)}")
                self.logger.error(f"Failed to open PDF {file_name}: {str(e)}")
            
    def display_current_page(self):
        if not self.pdf_document:
            return
            
        text = self.pdf_document.get_page_text(self.current_page)
        if not text:
            self.logger.warning(f"No text content found on page {self.current_page}")
            self.text_label.setText("No text content found on this page.")
            return
            
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
            ">
                {formatted_text}
            </div>
        '''
        self.text_label.setText(formatted_text)
        # Inform gaze indicator of new PDF line positions
        self.gaze_indicator.set_line_positions(self.pdf_document.line_positions)
        self.logger.info(f"Displayed page {self.current_page}")
        
    def update_eye_position(self):
        """Update eye position and detect which line is being read."""
        if not self.eye_tracker.is_connected():
            self.logger.debug("Eye tracker not connected")
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
        new_line = self.gaze_indicator.get_line_text_at_gaze()
        self._handle_line_change(new_line)
            
    def _reset_line_tracking(self):
        """Reset line tracking state."""
        # Treat as moving away from any line
        self._handle_line_change(None)
            
    def _update_line_time(self):
        """Update reading time for the current line."""
        self.line_reading_time = time.time() - self.line_start_time
        if self.current_line not in self.line_times:
            self.line_times[self.current_line] = 0
        self.line_times[self.current_line] += self.line_reading_time
        self.logger.info(f"Time spent on line: '{self.current_line[:50]}...' = {self.line_reading_time:.2f}s (Total: {self.line_times[self.current_line]:.2f}s)")

    def _handle_line_change(self, new_line: Optional[str]):
        """Manage transitions between lines and accumulate reading times."""
        now = time.time()
        # If gaze moved to a different line
        if new_line != self.current_line:
            # Finalize timing for previous line
            if self.current_line is not None and self.line_start_time is not None:
                duration = now - self.line_start_time
                self.line_times[self.current_line] = self.line_times.get(self.current_line, 0) + duration
                self.logger.info(
                    f"Accumulated time on line '{self.current_line[:50]}...': "
                    f"{self.line_times[self.current_line]:.2f}s"
                )
            # Start timing for new line if any
            if new_line is not None:
                self.line_start_time = now
            else:
                self.line_start_time = None
            self.current_line = new_line
        
    def show_explanation(self):
        """Show AI explanation for the current line."""
        if not self.current_line or not self.pdf_document:
            return
            
        context_lines = self.pdf_document.get_context_lines(self.current_line)
        explanation = self.ai_assistant.get_explanation(
            self.current_line,
            context_lines,
            self.line_reading_time
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