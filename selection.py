from PIL import ImageDraw, ImageEnhance
from PIL import Image, ImageTk
import cv2
from image_effect import display_image


def handle_mouse_down(event):
    global rect_start
    rect_start = (event.x, event.y)

def handle_mouse_drag(event, img_label, image):
    global rect_start
    if rect_start:
        img_copy = image.resize((600, 600))
        img_draw = ImageDraw.Draw(img_copy)
        img_draw.rectangle([rect_start, (event.x, event.y)], outline="red", width=2)
        tk_image = ImageTk.PhotoImage(img_copy)
        img_label.config(image=tk_image)

def handle_mouse_up(event, image, img_label):
    global rect_start, rect_end, selected_area
    rect_end = (event.x, event.y)
    selected_area = calculate_selection(image)

def calculate_selection(image):
    global rect_start, rect_end
    if rect_start and rect_end:
        x1, y1 = rect_start
        x2, y2 = rect_end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        orig_width, orig_height = image.size
        scale_x = orig_width / 600
        scale_y = orig_height / 600

        return int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

def adjust_selection_brightness(image, img_label, brightness_factor, selected_area):
    if selected_area:
        x1, y1, x2, y2 = selected_area
        cropped = image.crop((x1, y1, x2, y2))
        enhancer = ImageEnhance.Brightness(cropped)
        cropped = enhancer.enhance(brightness_factor)
        image.paste(cropped, (x1, y1, x2, y2))
        display_image(image, img_label)
