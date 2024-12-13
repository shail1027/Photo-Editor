import tkinter as tk
from tkinter import filedialog, colorchooser  # 파일 열기/저장을 위한 모듈
from PIL import Image, ImageTk  # 이미지 처리를 위한 Pillow 라이브러리
import cv2
import image_effect
import numpy as np

label = None  # 이미지가 표시될 Label
start_point = None  # 선택 시작점
end_point = None  # 선택 끝점점
tk_img = None  # tkinter에서 사용할 이미지 객체

brush_color = (0, 0, 255) # 메이크업 필터 브러시 색
brush_size = 20 # 메이크업 필터 브러시 사이즈 

blur_brush = 20 # 블러효과 브러시 사이즈


def ui(root):
    """UI를 구성하는 함수수"""

    global label, blur_brush

    # 왼쪽 UI
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
    brush_size_slider_blur = tk.Scale(
        frame, from_=1, to=100, orient=tk.HORIZONTAL, label="브러쉬 크기"
    )
    brush_size_slider_blur.pack(pady=5)
    brush_size_slider_blur.bind(
        "<Motion>", lambda event: blur_brush_size(brush_size_slider_blur.get())
    )

    # 블러 효과 활성화 버튼
    tk.Button(frame, text="블러 효과", command=apply_blur_filter).pack(pady=5)

    # 선명 효과(=샤프닝 필터) 버튼
    tk.Button(frame, text="선명 효과", command=apply_sharpen_filter).pack(pady=5)

    # 레트로 필터 버튼
    tk.Button(right_frame, text="레트로 필터", command=apply_retro_filter).pack(pady=5)

    tk.Button(right_frame, text="픽셀 유동화", command=liquify_apply).pack(pady=5)

    tk.Button(right_frame, text="무채색 필터", command=apply_mono_filter).pack(pady=5)

    brush_size_slider = tk.Scale(
        right_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="브러쉬 크기"
    )
    brush_size_slider.pack(pady=5)
    brush_size_slider.bind(
        "<Motion>", lambda event: brush_size_change(brush_size_slider.get())
    )
    makeup_button = tk.Button(
        right_frame, text="Makeup", command=apply_makeup
    )
    makeup_button.pack(pady=5)

    color_button = tk.Button(right_frame, text="색상 선택", command=choose_color)
    color_button.pack(pady=5)

    # 잡티제거 필터
    tk.Button(right_frame, text="페퍼 필터", command=apply_nosie_remove).pack(
        pady=5
    )

    # y2k 필터
    tk.Button(right_frame, text="y2k 필터", command=apply_y2k_filter).pack(pady=5)

    # 윤곽선 강조(윤곽선 스케치 느낌의 필터) 버튼
    tk.Button(right_frame, text="윤곽선 강조", command=apply_edgeS_filter).pack(pady=5)

    # 원본으로 되될리기 버튼
    tk.Button(right_frame, text="원본으로 되돌리기", command=restore_original).pack(
        pady=5
    )

    # Ctrl Z, Ctrl Y (되돌리기, 다시 실행행)
    tk.Button(right_frame, text="되돌리기", command=undo).pack(pady=5)
    tk.Button(right_frame, text="다시 실행", command=redo).pack(pady=5)


# ------ 파일 관련 함수 ------#
def open_img():
    """이미지 열기 함수"""
    global tk_img
    path = filedialog.askopenfilename()  # 파일 선택창
    if not path:
        print("파일이 선택되지 않았습니다.")
        return

    try:
        # Pillow를 사용해 한글 경로 처리 및 이미지 읽기
        img_pil = Image.open(path)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환

        # 이미지 상태 저장 및 표시
        image_effect.init_image(img)
        display_image()
    except Exception as e:
        print(f"Error loading image: {e}")


def save_img():
    """이미지 저장 함수"""
    img = image_effect.get_image()  # 현재 이미지 가져오기
    path = filedialog.asksaveasfilename(
        defaultextension=".png"
    )  # 기본 확장자를 png로 설정
    img_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # OpenCV는 기본적으로 BGR이기때문에 RGB로 변환
    Image.fromarray(img_rgb).save(path)  # Pillow의 Image.fromarray를 통해 이미지 저장


# ------ 대비 조정 ------#
def adjust_contrast(arg):
    """대비 조정 적용"""
    record_and_apply(image_effect.adjust_contrast, arg)


# ------ 채도 조정 ------#
def adjust_saturation(arg):
    """채도 조정 적용"""
    record_and_apply(image_effect.adjust_saturation, arg)


# ------ 색조 조정 ------#
def adjust_hue(arg):
    """색조 조정 적용"""
    record_and_apply(image_effect.adjust_hue, arg)


# ------ 흑백화 ------#
def apply_gray_filter():
    """이미지 흑백화 적용"""
    record_and_apply(image_effect.convert_to_grayscale)


# ------ 윤곽선 추출------$
def apply_edge_filter():
    """윤곽선 추출 적용"""
    record_and_apply(image_effect.edge_detection)


# ------ 블러 적용 ------#
def apply_blur_adjustment(event):
    """마우스로 특정 부분에 블러 적용"""

    global start_point, resized_image, blur_brush

    if resized_image is None:
        print("Error: Resized 이미지가 없습니다.")
        return

    try:
        # 드래그 끝점
        end_point = (event.x, event.y)

        # 블러링 필터 적용
        record_and_apply(
            image_effect.apply_blur,
            resized_image,
            start_point,
            end_point,
            blur_brush,
        )

        # 시작점 갱신
        start_point = end_point

    except Exception as e:
        print(f"Error applying blur adjustment: {e}")


def apply_blur_filter():
    """블러 효과 적용 활성화"""
    global label
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", apply_blur_adjustment)
    label.bind("<ButtonRelease-1>", on_mouse_release)


def blur_brush_size(value):
    """블러 브러쉬 크기 조절"""
    global blur_brush
    blur_brush = int(value)


# ------ 선명 효과 적용 ------#
def apply_sharpen_filter():
    """선명 효과 적용"""
    record_and_apply(image_effect.sharpen_filter)


# ------ 레트로 느낌 필터 적용 ------#
def apply_retro_filter():
    """레트로 필터 적용"""
    record_and_apply(image_effect.retro_filter)


# ------ 잡티 제거 효과 적용 ------#
def apply_noise_remove_image(event):
    """마우스로 이미지에서 잡티 제거를 적용"""

    global resized_image

    if resized_image is None:
        print("Error: Resized 이미지가 없습니다.")
        return

    # 클릭 좌표 가져오기
    x, y = event.x, event.y

    # 잡티 제거 적용
    record_and_apply(image_effect.remove_noise, resized_image, (x, y))


def apply_nosie_remove():
    """잡티 제거 모드 활성화"""

    global label

    # 마우스 이벤트 바인딩
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", apply_salt_pepper_removal)
    label.bind("<ButtonRelease-1>", on_mouse_release)


# ------ y2k필터 적용 ------#
def apply_y2k_filter():
    """y2k필터 적용"""
    record_and_apply(image_effect.y2k_filter)


# ------ 무채색 필터 적용 ------#
def apply_mono_filter():
    """무채색 느낌의 보정 필터 적용"""
    record_and_apply(image_effect.mono_filter)


# ------ 윤곽선 강조 필터 적용 ------#
def apply_edgeS_filter():
    """윤곽선 강조 필터 적용"""
    record_and_apply(image_effect.edge_emphasize)


# ------ 원본으로 되돌리기 ------#
def restore_original():
    """보정된 이미지를 원본으로 되돌림"""
    record_and_apply(image_effect.original)


# ------ Ctrl+z, Ctrl+y ------#
def record_and_apply(effect_function, *args):
    """
    작업 기록을 자동으로 저장하고, 효과를 적용
    :param effect_function: 적용할 효과 함수
    :param args: 효과 함수에 전달할 추가 인자
    """
    # 현재 상태를 undo 스택에 저장
    image_effect.push_to_undo(image_effect.get_image())
    # 효과 함수 호출
    effect_function(*args)
    # 결과 갱신
    display_image()


def undo():
    """되돌리기(Ctrl+z)"""

    global label
    prev_img = image_effect.pop_from_undo()
    if prev_img is not None:
        image_effect.push_to_redo(
            image_effect.get_image()
        )  # 현재 상태를 redo 스택에 저장
        image_effect.set_image(prev_img)  # 이전 상태로 복원
        display_image()  # 이미지 갱신
    else:
        print("되돌릴 작업이 없습니다")


def redo():
    """다시 실행(Ctrl+y)"""

    global label
    next_img = image_effect.pop_from_redo()
    if next_img is not None:
        image_effect.push_to_undo(
            image_effect.get_image()
        )  # 현재 상태를 undo 스택에 저장
        image_effect.set_image(next_img)  # 다음 상태로 복원
        display_image()  # 이미지 갱신
    else:
        print("다시 실행할 작업이 없습니다!")


# ------ 픽셀 유동화 실행 및 적용------#
def liquify_apply():
    """픽셀 유동화 활성화"""
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", apply_liquify)
    label.bind("<ButtonRelease-1>", on_mouse_release)


def apply_liquify(event):
    """마우스 드래그를 통해 Resized 이미지에서 픽셀 유동화 적용"""
    global start_point
    if resized_image is None:
        print("Error: Resized 이미지가 없습니다.")
        return

    if start_point is None:
        return

    # Tkinter 이벤트 좌표를 Resized 이미지 좌표로 그대로 사용
    end_point = (event.x, event.y)
    start_point_mapped = start_point

    # record_and_apply를 통해 liquify_pixels 실행
    record_and_apply(
        image_effect.liquify_pixels, resized_image, start_point_mapped, end_point
    )

    # 새로운 시작점 업데이트
    start_point = end_point


# ------ 메이크업 브러쉬 실행 및 적용------#
def apply_color_adjustment(event):
    """메이크업업 기능을 적용하는 함수"""
    global brush_color, brush_size, start_point
    # 드래그된 좌표를 사용하여 apply_makeup 호출
    if resized_image is not None:
        end_point = (event.x, event.y)
        record_and_apply(
            image_effect.apply_makeup,
            resized_image,
            start_point,
            end_point,
            brush_color,  # 색상
            brush_size,  # 브러쉬 크기
        )
        start_point = end_point


def apply_makeup():
    """색상 강조 모드 활성화"""

    global label
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", apply_color_adjustment)
    label.bind("<ButtonRelease-1>", on_mouse_release)


def choose_color():
    """색상 선택을 위한 색상 선택기 호출"""
    global brush_color
    color_code = colorchooser.askcolor(title="Select color")[0]
    if color_code:
        # RGB -> BGR로 변환
        brush_color = tuple(
            map(int, (color_code[2], color_code[1], color_code[0]))
        )  # (B, G, R)


def brush_size_change(value):
    """브러쉬 크기 적용"""
    global brush_size
    brush_size = int(value)


# ------ 마우스 이벤트 처리 ------#
def on_mouse_press(event):
    """마우스 클릭 시 시작점 설정"""
    global start_point
    start_point = (event.x, event.y)


def on_mouse_release(event):
    """마우스 뗄 시 선택 초기화"""
    global start_point
    start_point = None


# ------ 이미지 보이기 ------#
def display_image():
    """현재 이미지를 Label에 표시 (800x800)"""
    global label, tk_img, resized_image, displayed_width, displayed_height
    img = image_effect.get_image()

    if img is None:
        print("Error: No image to display.")
        return

    # 원본 비율 유지하면서 최대 800x800 크기로 축소
    max_size = 800
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)  # 축소 비율 계산
    displayed_width, displayed_height = int(w * scale), int(h * scale)

    # 이미지 크기 조정 및 저장
    resized_image = cv2.resize(
        img, (displayed_width, displayed_height), interpolation=cv2.INTER_AREA
    )

    # OpenCV 이미지를 Tkinter에서 사용할 수 있도록 변환
    img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(img_pil)

    # Label에 이미지 업데이트
    label.config(image=tk_img, bg="gray")
    label.image = tk_img  # 참조 유지
