import fitz  # PyMuPDF
import tkinter as tk
from PIL import Image, ImageTk
import time
from gaze_tracking_utils import GazeTracker
from face_model_array import face_model_all
import cv2
import numpy as np
import threading
import os

# Initialize root before anything else
root = tk.Tk()
root.title("PDF Reader with Gaze Tracking")

pdf_document = None
current_page = 0
gaze_tracker = GazeTracker(calibration_matrix_path="./calibration_matrix.yaml", model_path="./p00.ckpt")
tracking = False
tracking_thread = None
capture_thread = None
frame = None

# Shape of face_model_all = np.zeros((468, 3), dtype=np.float32)
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

monitor_mm, monitor_pixels = (800, 600), (1920, 1080)  # Replace with actual monitor dimensions
plane = np.eye(3).reshape(-1)

def extract_lines_from_pdf(pdf_path):
    global pdf_document
    pdf_document = fitz.open(pdf_path)
    lines = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("blocks")

        for block in text:
            if block[6] == 0:  # this is a text block
                lines.append(block)
    
    return lines

def update_canvas(page_image):
    photo = ImageTk.PhotoImage(page_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def on_vertical_scroll(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    if tracking:
        update_gaze_tracking()

def update_gaze_tracking():
    global current_line, new_lines_accessed, start_time, frame
    while tracking:
        # print("Updating gaze tracking...")  # Debugging print statement
        if frame is not None:
            point_on_screen = gaze_tracker.process_frame(frame, face_model, face_model_all, landmarks_ids, plane, monitor_mm, monitor_pixels)
            if point_on_screen:
                pdf_coordinates = convert_screen_to_pdf_coordinates(point_on_screen)
                update_line_access(pdf_coordinates)
            else:
                print("No gaze point detected on screen.")  # Debugging print statement
        time.sleep(1)  # Sleep for a second before the next update

def capture_webcam_frames():
    global frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Webcam Frame', frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def convert_screen_to_pdf_coordinates(point_on_screen):
    screen_x, screen_y = point_on_screen
    pdf_x = screen_x / monitor_pixels[0] * page_rect.width
    pdf_y = screen_y / monitor_pixels[1] * page_rect.height
    return pdf_x, pdf_y

def update_line_access(pdf_coordinates):
    global current_line, new_lines_accessed, start_time
    x, y = pdf_coordinates
    for i, line in enumerate(lines):
        x0, y0, x1, y1 = line[:4]
        if x0 <= x <= x1 and y0 <= y <= y1:
            if i != current_line:
                current_line = i
                new_lines_accessed += 1
            
            line_times[i] = line_times.get(i, 0) + (time.time() - start_time)
            start_time = time.time()

            print(f"Accessing line {i}: Time spent so far: {line_times[i]:.2f} seconds")  # Debugging print statement

            with open('reading_times.txt', 'a') as f:
                f.write(f"Line {i}: {line[4]} - Time spent: {line_times[i]:.2f} seconds\n")

            if new_lines_accessed % 3 == 0:
                for j, t in line_times.items():
                    if t > threshold_time and j not in hard_lines:
                        hard_lines.append(j)
                print("Hard to understand lines:", hard_lines)  # Debugging print statement
            break


def display_reading_times():
    reading_window = tk.Toplevel(root)
    reading_window.title("Reading Times")

    text = tk.Text(reading_window)
    text.pack(fill=tk.BOTH, expand=True)

    for line_num, time_spent in sorted(line_times.items()):
        text.insert(tk.END, f"Line {line_num}: {time_spent:.2f} seconds\n")

    text.config(state=tk.DISABLED)


def open_pdf():
    file_path = "The Setting Sun 6.pdf"  # Replace with your PDF file path
    if file_path:
        global lines, pdf_document, page_rect
        print(f"Opening PDF: {file_path}")  # Debugging print statement
        lines = extract_lines_from_pdf(file_path)
        page_rect = pdf_document[0].rect
        page_image = render_pdf_page(pdf_document[0])
        update_canvas(page_image)
        threading.Thread(target=render_and_update_canvas).start()
        print("PDF opened and displayed.")  # Debugging print statement

def render_and_update_canvas():
    page_image = render_pdf_page(pdf_document[0])
    update_canvas(page_image)

def render_pdf_page(page):
    zoom = 2  # zoom factor for better readability
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def start_tracking(event=None):
    global tracking, tracking_thread,start_time
    start_time = time.time()
    if not tracking:
        print("Starting gaze tracking...")  # Debugging print statement
        tracking = True
        tracking_thread = threading.Thread(target=update_gaze_tracking)
        tracking_thread.start()

def pause_tracking(event=None):
    global tracking
    print("Pausing gaze tracking...")  # Debugging print statement
    tracking = False

def write_final_reading_data():
    with open('final_reading_times.txt', 'w') as f:
        for line_num, time_spent in sorted(line_times.items()):
            f.write(f"Line {line_num}: {lines[line_num][4]} - Time spent: {time_spent:.2f} seconds\n")

def stop_tracking(event=None):
    global tracking
    print("Stopping gaze tracking and closing application...")  # Debugging print statement
    tracking = False
    write_final_reading_data()  # Write final reading data to file
    root.destroy()


def start_capture_thread():
    global capture_thread
    capture_thread = threading.Thread(target=capture_webcam_frames)
    capture_thread.daemon = True  # Ensures thread exits when main program exits
    capture_thread.start()

lines = []
line_times = {}
current_line = -1
new_lines_accessed = 0
threshold_time = 5
hard_lines = []
canvas_height = 800

canvas = tk.Canvas(root, width=1280, height=canvas_height)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

canvas.bind_all("<MouseWheel>", on_vertical_scroll)
root.bind('s', start_tracking)
root.bind('p', pause_tracking)
root.bind('<Escape>', stop_tracking)

menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_pdf)
file_menu.add_command(label="Reading Times", command=display_reading_times)

open_pdf()
start_capture_thread()

root.mainloop()
