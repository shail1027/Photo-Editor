import tkinter as tk
from tkinter import filedialog  # 파일 열기/저장을 위한 모듈
from PIL import Image, ImageTk  # 이미지 처리를 위한 Pillow 라이브러리
import cv2
import image_effect

label = None  # 이미지가 표시될 Label


def ui(root):  # UI를 구성하는 함수
    global label

    # 왼쪽 사이드 UI
    frame = tk.Frame(root, width=200, height=600)
    frame.place(x=0, y=0)

    # 오른쪽 사이드 UI
    right_frame = tk.Frame(root, width=200, height=600)
    right_frame.place(x=1000, y=0)

    # 이미지 표시 영역
    label = tk.Label(root, bg="gray")
    label.place(x=200, y=0, width=800, height=800)

    # 왼쪽 UI 버튼
    # 파일 열기&저장 버튼
    tk.Button(frame, text="파일 열기", command=open_img).pack(pady=5)
    tk.Button(frame, text="저장하기", command=save_img).pack(pady=5)

    # 대비 조절 슬라이더&버튼
    contrast_slider = tk.Scale(
        frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="대비"
    )
    contrast_slider.pack(pady=5)
    tk.Button(
        frame, text="대비 조절", command=lambda: adjust_contrast(contrast_slider.get())
    ).pack(pady=5)

    # 채도 조절 슬라이더&버튼
    saturation_slider = tk.Scale(
        frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="채도"
    )
    saturation_slider.pack(pady=5)
    tk.Button(
        frame,
        text="채도 조절",
        command=lambda: adjust_saturation(saturation_slider.get()),
    ).pack(pady=5)

    # 밝기 조절 슬라이더&버튼
    slider = tk.Scale(
        frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="밝기"
    )
    slider.pack(pady=5)
    tk.Button(
        frame, text="밝기 조절", command=lambda: adjust_brightness(slider.get())
    ).pack(pady=5)

    # 색조 조절 슬라이더&버튼
    hue_slider = tk.Scale(
        frame, from_=-90, to=90, resolution=10, orient=tk.HORIZONTAL, label="색조"
    )
    hue_slider.pack(pady=5)
    tk.Button(
        frame, text="색조 조절", command=lambda: adjust_hue(hue_slider.get())
    ).pack(pady=5)

    # 사진을 흑백으로 바꾸기(흑백화) 버튼
    tk.Button(frame, text="흑백화", command=apply_gray_filter).pack(pady=5)

    # 사진의 윤곽선을 추출하기(외곽선 추출) 버튼
    tk.Button(frame, text="윤곽선 추출", command=apply_edge_filter).pack(pady=5)

    # 블러 효과 버튼
    tk.Button(frame, text="블러 효과", command=apply_blur_filter).pack(pady=5)

    # 선명 효과(=샤프닝 필터) 버튼
    tk.Button(frame, text="선명 효과", command=apply_sharpen_filter).pack(pady=5)

    # 오른쪽 UI 버튼
    # 레트로 필터 버튼
    tk.Button(right_frame, text="레트로 필터", command=apply_retro_filter).pack(pady=5)

    # 빈티지 필터 버튼
    tk.Button(right_frame, text="빈티지 필터", command=apply_vintage_filter).pack(
        pady=5
    )

    # 윤곽선 강조(윤곽선 스케치 느낌의 필터) 버튼
    tk.Button(right_frame, text="윤곽선 강조", command=apply_edgeS_filter).pack(pady=5)

    # 원본으로 되될리기 버튼
    tk.Button(right_frame, text="원본으로 되돌리기", command=restore_original).pack(
        pady=5
    )


def open_img():  # 이미지 열기
    path = filedialog.askopenfilename()  # 파일 선택창을 띄워 이미지를 선택하도록 함
    img = cv2.imread(path)  # OpenCV로 이미지 불러오기
    image_effect.set_image(img)  # 이미지 상태 저장
    display_image()  # 이미지를 띄움


def save_img():  # 이미지 저장
    img = image_effect.get_image()  # 현재 이미지 가져오기
    path = filedialog.asksaveasfilename(
        defaultextension=".png"
    )  # 기본 확장자를 png로 설정
    img_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # OpenCV는 기본적으로 BGR이기때문에 RGB로 변환
    Image.fromarray(img_rgb).save(path)  # Pillow의 Image.fromarray를 통해 이미지 저장


def apply_retro_filter():  # 레트로 필터를 적용하는 함수
    image_effect.retro_filter()  # 필터 적용
    display_image()  # 결과 표시


def apply_gray_filter():  # 흑백화 필터를 적용하는 함수
    image_effect.convert_to_grayscale()
    display_image()


def apply_edge_filter():  # 윤곽선 추출 필터를 적용하는 함수
    image_effect.edge_detection()
    display_image()


def apply_sharpen_filter():  # 선명 효과 필터를 적용하는 함수
    image_effect.sharpen_filter()
    display_image()


def apply_blur_filter():  # 블러 효과를 적용하는 함수
    image_effect.blur_filter()
    display_image()


def apply_vintage_filter():  # 빈티지 필터를 적용하는 함수
    image_effect.vintage_filter()
    display_image()


def adjust_brightness(arg):  # 밝기 조절을 적용하는 함수
    image_effect.adjust_brightness(arg)  # 밝기 조정
    display_image()  # 결과 표시


def apply_edgeS_filter():  # 윤곽선 강조(윤곽선 스케치) 효과를 적용하는 함수
    image_effect.edge_emphasize()
    display_image()


def adjust_contrast(arg):  # 대비 조절을 적용하는 함수
    image_effect.adjust_contrast(arg)
    display_image()


def adjust_saturation(arg):  # 채도 조절을 적용하는 함수
    image_effect.adjust_saturation(arg)
    display_image()


def adjust_hue(arg):  # 색조 조절을 적용하는 함수
    image_effect.adjust_hue(arg)
    display_image()


def restore_original():  # 원본으로 되돌리는 함수
    image_effect.original()
    display_image()


def display_image():  # 현재 이미지를 표시하는 함수
    global label
    img = image_effect.get_image()  # 현재 이미지 가져오기
    img_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # OpenCV는 BGR이 기본이므로 RGB로 변환
    img_pil = Image.fromarray(img_rgb)  # Pillow 이미지 객체로 변환
    img_resized = img_pil.resize((600, 600))  # 이미지 크기를 600x600으로 리사이즈
    tk_img = ImageTk.PhotoImage(
        img_resized
    )  # Tkinter에서 이미지를 표시할 수 있도록 변환

    # label에 이미지를 연결
    # tkinter에서 이미지를 표시하려면 PhotoImage객체를 이용해야하기 때문에 Pillow 이미지 객체를 tkinter가 이해할 수 있는 형식으로 변화하는 과정이 필요하다.
    label.config(image=tk_img)
    label.image = tk_img
