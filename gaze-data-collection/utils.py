import sys
import time
from datetime import datetime
from enum import Enum
import random

import cv2
import numpy as np
from typing import Tuple, Union

from webcam import WebcamSource

def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
        from screeninfo import get_monitors
        
        default_screen = get_monitors()[0].__dict__

        h_mm = default_screen['height_mm']
        w_mm = default_screen['width_mm']

        h_pixels = default_screen['height']
        w_pixels = default_screen['width']

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

class TargetOrientation(Enum):
    UP = ord('z')
    DOWN = ord('s')
    LEFT = ord('q')
    RIGHT = ord('d')

def create_image(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=1., orientation=TargetOrientation.RIGHT, target='E') -> Tuple[np.ndarray, float, bool]:
    """
    Create image to display on screen.

    :param monitor_pixels: monitor dimensions in pixels
    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param orientation: orientation of the target
    :param target: char to write on image
    :return: created image, new smaller circle_scale and bool that indicated if it is the last frame in the animation
    """
    width, height = monitor_pixels
    if orientation == TargetOrientation.LEFT or orientation == TargetOrientation.RIGHT:
        img = np.zeros((height, width, 3), np.float32)

        if orientation == TargetOrientation.LEFT:
            center = (width - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.LEFT:
            img = cv2.flip(img, 1)
    else:  # TargetOrientation.UP or TargetOrientation.DOWN
        img = np.zeros((width, height, 3), np.float32)
        center = (center[1], center[0])

        if orientation == TargetOrientation.UP:
            center = (height - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.UP:
            img = cv2.flip(img, 1)

        img = img.transpose((1, 0, 2))

    return img / 255, circle_scale * 0.85, end_animation_loop

def write_text_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Write target on image and check if last frame of the animation.

    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param img: image to write data on
    :param target: char to write
    :return: True if last frame of the animation
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)
    cv2.circle(img, center, int(text_size[0] * 5 * circle_scale), (32, 32, 32), -1)
    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    end_animation_loop = circle_scale < random.uniform(0.1, 0.5)
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)

    return end_animation_loop

def get_grid_positions(monitor_pixels: Tuple[int, int], rows: int = 5, cols: int = 5) -> list:
    """
    Get grid positions on monitor including corners, starting and ending at the screen edge vertically
    and at the effective width's edge horizontally.

    :param monitor_pixels: monitor dimensions in pixels
    :param rows: number of rows in the grid
    :param cols: number of columns in the grid
    :return: list of tuple coordinates for grid positions including corners
    """
    width, height = monitor_pixels

    # Calculate the effective width after ignoring 20% on each side
    effective_width = width * 0.6
    x_start = width * 0.2  # Start at 20% of the width
    x_spacing = effective_width // (cols - 1)  # Ensure it starts and ends at the effective width's edge
    y_spacing = height // (rows - 1)  # Ensure it starts and ends at the screen edge

    positions = [(int(x_start + x * x_spacing), int(y * y_spacing)) for x in range(cols) for y in range(rows)]

    # Adding corners (ensuring they are exactly at the edges)
    corners = [
        (0, 0), (width - 1, 0),
        (0, height - 1), (width - 1, height - 1)
    ]
    positions.extend(corners)
    random.shuffle(positions)
    return positions


def show_point_on_screen(window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource, index: int) -> Tuple[str, Tuple[int, int], float, int]:
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    grid_positions = get_grid_positions(monitor_pixels, 8,5)
    if index == 0: print(grid_positions)
    
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None

    while not end_animation_loop:
        center = grid_positions[index]
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(10):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('b'):
                cv2.destroyAllWindows()
                sys.exit()
    
    if end_animation_loop:
        index = (index + 1) % len(grid_positions)
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            key = cv2.waitKey(42) & 0xFF
            if key == orientation.value:
                print("correct")
                source.clear_frame_buffer()
                cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
                time_till_capture = time.time() - start_time_color_change
                break

        cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
        cv2.waitKey(500)
        end_animation_loop = False
        circle_scale = 1.  # Reset circle scale for next point

    return f'{file_name}.jpg', center, time_till_capture, index
