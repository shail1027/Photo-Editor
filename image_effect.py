from PIL import ImageEnhance, Image
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog

def apply_filter(image, img_label, filter_type):
    if image:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(img_cv)

        if filter_type == "warm":
            r = cv2.add(r, 50)
        elif filter_type == "cool":
            b = cv2.add(b, 50)
        
        img_cv = cv2.merge((b, g, r))
        image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        display_image(image, img_label)

def adjust_brightness(image, img_label, brightness_factor):
    if image:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        display_image(image, img_label)

def create_collage(image, img_label):
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_paths:
        images = [Image.open(fp).resize((200, 200)) for fp in file_paths[:6]]
        collage = Image.new("RGB", (600, 400), "white")
        positions = [(0, 0), (200, 0), (400, 0), (0, 200), (200, 200), (400, 200)]
        for img, pos in zip(images, positions):
            collage.paste(img, pos)
        display_image(collage, img_label)

def display_image(image, img_label):
    resized = image.resize((600, 600))
    tk_image = ImageTk.PhotoImage(resized)
    img_label.config(image=tk_image)
