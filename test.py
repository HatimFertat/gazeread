import fitz  # PyMuPDF
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import time
from utils import get_monitor_dimensions
from gaze_tracking_utils import GazeTracker
import cv2
from webcam import WebcamSource
import numpy as np
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r'SymbolDatabase.GetPrototype\(\) is deprecated')

def update_canvas_dimensions(event):
    canvas.config(width=root.winfo_width(), height=root.winfo_height())
    canvas.configure(yscrollcommand=scrollbar.set)

root = tk.Tk()
root.attributes("-fullscreen", True)
root.title("PDF Reader with Gaze Tracking")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Configure the canvas to match screen dimensions
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack(fill=tk.BOTH, expand=True)

calibration_matrix_path = "./calibration_matrix.yaml"
pdf_document = None
current_page = 0
gaze_tracker = GazeTracker(calibration_matrix_path, model_path="./p00.ckpt", method='affine')
tracking = False
tracking_thread = None
capture_thread = None
frame = None

face_model_all = np.load("face_model.npy")
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

webcam_source = WebcamSource(width=1280, height=720, fps=25, buffer_size=10)

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
    global image_width, image_height
    photo = ImageTk.PhotoImage(page_image)
    canvas_width = screen_width
    canvas_height = screen_height
    image_width = photo.width()
    image_height = photo.height()
    x_offset = (canvas_width - image_width) // 2  # Center the image horizontally
    y_offset = (canvas_height - image_height) // 2  # Center the image vertically
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
    canvas.image = photo
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

    draw_grid(canvas)
    # canvas.create_oval(5, 5, 15, 15, fill="blue", outline="blue") #point at (0,0) origin of the screen


def on_vertical_scroll(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.after(10, update_grid)  # Slight delay for smoother performance
    if tracking:
        update_visible_lines()

def update_grid():
    canvas.delete("grid")  # Remove old grid
    draw_grid(canvas)  # Draw new grid

def update_visible_lines():
    canvas_y = canvas.canvasy(0)
    visible_height = canvas.winfo_height()
    visible_lines = []
    for i, line in enumerate(lines):
        [page_num, x0, y0, x1, y1, text] = line
        # Convert PDF coordinates to canvas coordinates
        canvas_y0 = y0 - canvas_y
        canvas_y1 = y1 - canvas_y
        if 0 <= canvas_y0 <= visible_height or 0 <= canvas_y1 <= visible_height:
            visible_lines.append((i, x0, canvas_y0, x1, canvas_y1, text))
    return visible_lines

def update_gaze_tracking():
    global current_line, new_lines_accessed, start_time, frame, gaze_points_computed
    while tracking:
        if frame is not None:
            point_on_screen = gaze_tracker.process_frame(frame, face_model, face_model_all, landmarks_ids)
            gaze_points_computed += 1
            draw_gaze_point(point_on_screen)
            
            visible_lines = update_visible_lines()
            update_line_access(point_on_screen, visible_lines)
            
            print(point_on_screen, "\t\t\t")
        time.sleep(0.05)


def update_line_access(screen_coordinates, visible_lines):
    global current_line, new_lines_accessed, start_time
    screen_x, screen_y = screen_coordinates
    for i, (line_index, x0, y0, x1, y1, text) in enumerate(visible_lines):
        if x0 <= screen_x <= x1 and y0 <= screen_y <= y1:
            if line_index != current_line:
                current_line = line_index
                new_lines_accessed += 1

            time_spent = time.time() - start_time
            if time_spent >= 0.5: #don't include lines that we looked at for less than 0.5s
                line_times[line_index] = line_times.get(line_index, 0) + time_spent
                start_time = time.time()

                with open('reading_times.txt', 'a') as f:
                    f.write(f"Line {line_index}: '{text}' - Time spent: {line_times[line_index]:.2f} seconds\n")

                if new_lines_accessed % 3 == 0:
                    for j, t in line_times.items():
                        if t > threshold_time and j not in hard_lines:
                            hard_lines.append(j)
            break

def capture_webcam_frames():
    global frame
    while True:
        try:
            frame = next(webcam_source)
        except StopIteration:
            break

def convert_screen_to_pdf_coordinates(point_on_screen, page_rect, monitor_pixels, canvas_width, canvas_height, pdf_width, pdf_height):
    screen_x, screen_y = point_on_screen
    # Calculate the offset if the PDF is centered within the canvas
    canvas_offset_x = (canvas_width - pdf_width) // 2
    canvas_offset_y = (canvas_height - pdf_height) // 2
    
    # Adjust screen coordinates by subtracting the offsets
    adjusted_x = (screen_x - canvas_offset_x) / pdf_width * page_rect.width
    adjusted_y = (screen_y - canvas_offset_y) / pdf_height * page_rect.height
    
    return adjusted_x, adjusted_y

def draw_gaze_point(point_on_screen):
    x, y = point_on_screen
    canvas.delete("gaze_point")
    canvas.create_oval(x-7, y-7, x+7, y+7, fill="red", tags="gaze_point")

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
        print("Starting gaze tracking...")
        tracking = True
        tracking_thread = threading.Thread(target=update_gaze_tracking)
        tracking_thread.start()

def pause_tracking(event=None):
    global tracking
    tracking = False

def write_final_reading_data():
    with open('final_reading_times.txt', 'w') as f:
        for line_num, time_spent in sorted(line_times.items()):
            f.write(f"Line {line_num}: {lines[line_num][5]} - Time spent: {time_spent:.2f} seconds\n")

def stop_tracking(event=None):
    global tracking
    print("Stopping gaze tracking and closing application...")
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

def draw_grid(canvas):
    canvas.delete("grid_line")  # Clear existing grid lines
    interval = 50  # Change as needed for more or less granularity
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    # Vertical lines
    for x in range(0, width, interval):
        canvas.create_line(x, 0, x, height, fill='gray', tags="grid_line")
    # Horizontal lines
    for y in range(0, height, interval):
        canvas.create_line(0, y, width, y, fill='gray', tags="grid_line")

# Initialization of variables
lines = []
line_times = {}
current_line = -1
new_lines_accessed = 0
threshold_time = 5
hard_lines = []
gaze_points_computed = 0

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

def on_scrollbar_move(*args):
    update_grid()
    if tracking:
        update_visible_lines()

scrollbar.config(command=lambda *args: [canvas.yview(*args), on_scrollbar_move()])

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
