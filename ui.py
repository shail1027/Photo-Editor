import tkinter as tk
from tkinter import filedialog
import cv2
from image_effect import apply_filter, adjust_brightness, create_collage
from selection import handle_mouse_down, handle_mouse_drag, handle_mouse_up, adjust_selection_brightness

image = None
tk_image = None
img_label = None
start = None
end = None
select_ROI = None

def ui(root):
    global img_label
    left_frame = tk.Frame(root, width=200, height=600, bg="lightgray")
    left_frame.place(x=0, y=0)

    img_label = tk.Label(root, bg="gray")
    img_label.place(x=200, y=0, width=600, height=600)

    # 버튼과 슬라이더 추가
    tk.Button(left_frame, text="Open Image", command=open_image).pack(pady=10)
    tk.Button(left_frame, text="Save Image", command=save_image).pack(pady=10)
    tk.Button(left_frame, text="Create Collage", command=lambda: create_collage(image, img_label)).pack(pady=10)
    tk.Button(left_frame, text="Apply Warm Filter", command=lambda: apply_filter(image, img_label, "warm")).pack(pady=10)
    tk.Button(left_frame, text="Apply Cool Filter", command=lambda: apply_filter(image, img_label, "cool")).pack(pady=10)
    
    brightness_slider = tk.Scale(left_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Brightness")
    brightness_slider.pack(pady=10)
    tk.Button(left_frame, text="Adjust Brightness", command=lambda: adjust_brightness(image, img_label, brightness_slider.get())).pack(pady=10)
    tk.Button(left_frame, text="Adjust Brightness (Selection)", command=lambda: adjust_selection_brightness(image, img_label, brightness_slider.get(), selected_area)).pack(pady=10)

    # 마우스 이벤트 바인딩
    img_label.bind("<ButtonPress-1>", lambda event: handle_mouse_down(event))
    img_label.bind("<B1-Motion>", lambda event: handle_mouse_drag(event, img_label, image))
    img_label.bind("<ButtonRelease-1>", lambda event: handle_mouse_up(event, image, img_label))

def open_image():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        display_image(image)

def save_image():
    global image
    if image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, image)

def display_image(img):
    global tk_image, img_label
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 600))
        tk_image = tk.PhotoImage(data=cv2.imencode('.ppm', img)[1].tobytes())
        img_label.config(image=tk_image)
