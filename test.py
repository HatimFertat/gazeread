import fitz  # PyMuPDF
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import time
from utils import get_monitor_dimensions
from gaze_tracking_utils import GazeTracker
from face_model_array import face_model_all
import cv2
import numpy as np
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r'SymbolDatabase.GetPrototype\(\) is deprecated')
root = tk.Tk()
# root.attributes("-fullscreen", True)
root.title("PDF Reader with Gaze Tracking")

calibration_matrix_path="./calibration_matrix.yaml"
pdf_document = None
current_page = 0
gaze_tracker = GazeTracker(calibration_matrix_path, model_path="./p00.ckpt")
tracking = False
tracking_thread = None
capture_thread = None
frame = None

face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])



monitor_mm, monitor_pixels = get_monitor_dimensions()

def extract_lines_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_lines_data = []

    for page_number, page in enumerate(document):
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    bbox = line["bbox"]
                    text = line["spans"][0]["text"] if line["spans"] else ""
                    line_data = [page_number + 1, bbox[0], bbox[1], bbox[2], bbox[3], text]
                    all_lines_data.append(line_data)
    return all_lines_data

def update_canvas(page_image):
    photo = ImageTk.PhotoImage(page_image)
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    image_width = photo.width()
    image_height = photo.height()
    x_offset = (canvas_width - image_width) // 4
    y_offset = (canvas_height - image_height) // 4
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
    canvas.image = photo
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def on_vertical_scroll(event):
    # pause_tracking()
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    if tracking:
        update_gaze_tracking()
    update_visible_lines()

def update_visible_lines():
    canvas_y = canvas.canvasy(0)
    visible_height = canvas.winfo_height()
    visible_lines = []
    for i, line in enumerate(lines):
        [page_num, x0, y0, x1, y1, text] = line
        if canvas_y <= y0 <= canvas_y + visible_height or canvas_y <= y1 <= canvas_y + visible_height:
            visible_lines.append(line)
    return visible_lines

def update_gaze_tracking():
    global current_line, new_lines_accessed, start_time, frame, gaze_points_computed
    while tracking:
        if frame is not None:
            point_on_screen = gaze_tracker.process_frame(frame, face_model, face_model_all, landmarks_ids)
            if point_on_screen:
                gaze_points_computed += 1
                draw_gaze_point(point_on_screen)
                pdf_coordinates = convert_screen_to_pdf_coordinates(point_on_screen, page_rect, monitor_pixels)
                print(point_on_screen, "\t\t\t", pdf_coordinates)
                visible_lines = update_visible_lines()
                update_line_access(pdf_coordinates, visible_lines)
        time.sleep(0.05)

def update_line_access(pdf_coordinates, visible_lines):
    global current_line, new_lines_accessed, start_time
    x, y = pdf_coordinates
    for i, line in enumerate(visible_lines):
        [page_num, x0, y0, x1, y1, text] = line
        if x0 <= x <= x1 and y0 <= y <= y1:
            if i != current_line:
                current_line = i
                new_lines_accessed += 1

            time_spent = time.time() - start_time
            if time_spent >= 0.5:
                line_times[i] = line_times.get(i, 0) + time_spent
                start_time = time.time()

                with open('reading_times.txt', 'a') as f:
                    f.write(f"Line {i}: '{text}' - Time spent: {line_times[i]:.2f} seconds\n")

                if new_lines_accessed % 3 == 0:
                    for j, t in line_times.items():
                        if t > threshold_time and j not in hard_lines:
                            hard_lines.append(j)
            break


def capture_webcam_frames():
    global frame
    cap = cv2.VideoCapture(0)
    fps = 60
    delay = 1 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time.sleep(delay)
    
    cap.release()


def convert_screen_to_pdf_coordinates(point_on_screen, page_rect, monitor_pixels):
    screen_x, screen_y = point_on_screen
    pdf_x = screen_x / monitor_pixels[0] * page_rect.width
    pdf_y = screen_y / monitor_pixels[1] * page_rect.height
    return pdf_x, pdf_y


def draw_gaze_point(point_on_screen):
    screen_x, screen_y = point_on_screen
    canvas.create_oval(screen_x - 5, screen_y - 5, screen_x + 5, screen_y + 5, outline="blue", width=2)

def display_reading_times():
    reading_window = tk.Toplevel(root)
    reading_window.title("Reading Times")

    text = tk.Text(reading_window)
    text.pack(fill=tk.BOTH, expand=True)

    for line_num, time_spent in sorted(line_times.items()):
        text.insert(tk.END, f"Line {line_num}: {time_spent:.2f} seconds\n")

    text.config(state=tk.DISABLED)

def open_pdf():
    file_path = "The Setting Sun 6.pdf"
    if file_path:
        global lines, pdf_document, page_rect
        lines = extract_lines_from_pdf(file_path)
        pdf_document = fitz.open(file_path)
        page_rect = pdf_document[0].rect
        page_image = render_pdf_page_with_lines(pdf_document[0], lines)
        update_canvas(page_image)
        threading.Thread(target=render_and_update_canvas).start()

def render_and_update_canvas():
    page_image = render_pdf_page_with_lines(pdf_document[0], lines)
    update_canvas(page_image)

def render_pdf_page_with_lines(page, lines):
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    draw = ImageDraw.Draw(img)
    for line in lines:
        [page_num, x0, y0, x1, y1, text] = line
        draw.rectangle([x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom], outline="red", width=2)

    return img

def start_tracking(event=None):
    global tracking, tracking_thread, start_time, gaze_points_computed
    start_time = time.time()
    gaze_points_computed = 0
    if not tracking:
        print("Starting gaze tracking...")  # Debugging print statement
        tracking = True
        tracking_thread = threading.Thread(target=update_gaze_tracking)
        tracking_thread.start()
        # threading.Thread(target=compute_gaze_points_per_minute).start()

def pause_tracking(event=None):
    global tracking
    tracking = False

def write_final_reading_data():
    with open('final_reading_times.txt', 'w') as f:
        for line_num, time_spent in sorted(line_times.items()):
            f.write(f"Line {line_num}: {lines[line_num][5]} - Time spent: {time_spent:.2f} seconds\n")

def stop_tracking(event=None):
    global tracking
    print("Stopping gaze tracking and closing application...")  # Debugging print statement
    tracking = False
    write_final_reading_data()
    root.destroy()

def start_capture_thread():
    global capture_thread
    capture_thread = threading.Thread(target=capture_webcam_frames)
    capture_thread.daemon = True
    capture_thread.start()

def compute_gaze_points_per_minute():
    global gaze_points_computed
    while tracking:
        time.sleep(60)
        print(f"Gaze points per minute: {gaze_points_computed}")
        gaze_points_computed = 0

lines = []
line_times = {}
current_line = -1
new_lines_accessed = 0
threshold_time = 5
hard_lines = []
canvas_height = 800
gaze_points_computed = 0

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
