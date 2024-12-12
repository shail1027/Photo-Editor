import tkinter as tk
from tkinter import filedialog  # 파일 열기/저장을 위한 모듈
from PIL import Image, ImageTk, ImageDraw  # 이미지 처리를 위한 Pillow 라이브러리
import cv2
import image_effect
import numpy as np

label = None  # 이미지가 표시될 Label
start_point = None  # 선택 시작점
end_point = None
tk_img = None  #
start_point = None

scale = None


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

    tk.Button(right_frame, text="픽셀 유동화", command=liquify_tool).pack(pady=5)

    # 빈티지 필터 버튼
    # tk.Button(right_frame, text="빈티지 필터", command=apply_vintage_filter).pack(
    #     pady=5
    # )

    # 윤곽선 강조(윤곽선 스케치 느낌의 필터) 버튼
    tk.Button(right_frame, text="윤곽선 강조", command=apply_edgeS_filter).pack(pady=5)

    # 원본으로 되될리기 버튼
    tk.Button(right_frame, text="원본으로 되돌리기", command=restore_original).pack(
        pady=5
    )

    # tk.Button(right_frame, text="선택 영역 선택", command=selection_tool).pack(pady = 5)
    tk.Button(right_frame, text="되돌리기", command=undo).pack(pady=5)
    tk.Button(right_frame, text="다시 실행", command=redo).pack(pady=5)
    # tk.Button(right_frame, text="올가미 툴", command=lasso_tool).pack(pady=5)


def open_img():
    """이미지 열기 (한글 경로 지원 및 여백 포함)"""
    global tk_img
    path = filedialog.askopenfilename()  # 파일 선택창
    if not path:
        print("No file selected.")
        return

    try:
        # Pillow를 사용해 한글 경로 처리 및 이미지 읽기
        img_pil = Image.open(path)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환

        # 이미지 상태 저장 및 표시
        image_effect.set_image(img)
        display_image()
    except Exception as e:
        print(f"Error loading image: {e}")


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
    record_and_apply(image_effect.retro_filter)


def apply_gray_filter():  # 흑백화 필터를 적용하는 함수
    record_and_apply(image_effect.convert_to_grayscale)


def apply_edge_filter():  # 윤곽선 추출 필터를 적용하는 함수
    record_and_apply(image_effect.edge_detection)


def apply_sharpen_filter():  # 선명 효과 필터를 적용하는 함수
    record_and_apply(image_effect.sharpen_filter)


def apply_blur_filter():  # 블러 효과를 적용하는 함수
    record_and_apply(image_effect.blur_filter)


# def apply_vintage_filter():  # 빈티지 필터를 적용하는 함수
#     record_and_apply(image_effect.adjust_contrast, arg)


def adjust_brightness(arg):  # 밝기 조절을 적용하는 함수
    record_and_apply(image_effect.adjust_brightness, arg)


def record_and_apply(effect_function, *args):
    """
    작업 기록을 자동으로 저장하고, 효과를 적용
    :param effect_function: 적용할 효과 함수 (image_effect.py의 함수)
    :param args: 효과 함수에 전달할 추가 인자
    """
    # 현재 상태를 undo 스택에 저장
    image_effect.push_to_undo(image_effect.get_image())
    # 효과 함수 호출
    effect_function(*args)
    # 결과 갱신
    display_image()


def apply_edgeS_filter():  # 윤곽선 강조(윤곽선 스케치) 효과를 적용하는 함수
    record_and_apply(image_effect.edge_emphasize)


def adjust_contrast(arg):
    """대비 조정"""
    record_and_apply(image_effect.adjust_contrast, arg)


def adjust_saturation(arg):
    """채도 조정"""
    record_and_apply(image_effect.adjust_saturation, arg)


def adjust_hue(arg):  # 색조 조절을 적용하는 함수
    record_and_apply(image_effect.adjust_hue, arg)


def restore_original():  # 원본으로 되돌리는 함수
    record_and_apply(image_effect.original)


def undo():
    """되돌리기 버튼 동작"""
    global label
    prev_img = image_effect.pop_from_undo()
    if prev_img is not None:
        image_effect.push_to_redo(
            image_effect.get_image()
        )  # 현재 상태를 redo 스택에 저장
        image_effect.set_image(prev_img)  # 이전 상태로 복원
        display_image()  # 이미지 갱신
    else:
        print("되돌릴 작업이 없습니다!")


def redo():
    """다시 실행 버튼 동작"""
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


# 유동화 UI 로직
def liquify_tool():
    """픽셀 유동화 도구 활성화"""
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", apply_liquify)
    label.bind("<ButtonRelease-1>", on_mouse_release)


def on_mouse_press(event):
    """마우스 클릭 시 시작점 설정"""
    global start_point
    start_point = (event.x, event.y)
    print("strpt", start_point)


def apply_liquify(event):
    """마우스 드래그를 통해 Resized 이미지에서 픽셀 유동화 적용"""
    global start_point, resized_image
    if resized_image is None:
        print("Error: Resized 이미지가 없습니다.")
        return

    if start_point is None:
        return

    # Tkinter 이벤트 좌표를 Resized 이미지 좌표로 그대로 사용
    end_point = (event.x, event.y)
    start_point_mapped = start_point

    # 디버깅 메시지
    print(f"Start: {start_point_mapped}, End: {end_point}")

    # Resized 이미지에서 픽셀 유동화 적용
    modified_img = image_effect.liquify_pixels(
        resized_image, start_point_mapped, end_point
    )

    # 이미지가 변경되었는지 확인
    # if np.array_equal(resized_image, modified_img):
    #     print("Image not changed!")
    # else:
    #     print("Image successfully changed!")

    # Resized 이미지 업데이트
    resized_image = modified_img
    display_modified_image()  # 변경된 이미지를 표시
    start_point = end_point  # 끝점을 새로운 시작점으로 설정


def display_modified_image():
    """변경된 Resized 이미지를 Label에 표시"""
    global label, resized_image, tk_img

    if resized_image is None:
        print("Error: No resized image to display.")
        return

    # OpenCV 이미지를 Tkinter에서 사용할 수 있도록 변환
    img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(img_pil)

    # Label에 이미지 업데이트
    label.config(image=tk_img, bg="gray")
    label.image = tk_img  # 참조 유지


def on_mouse_release(event):
    """마우스 릴리스 시 선택 초기화"""
    global start_point
    start_point = None


'''
def selection_tool():
    """선택 도구 활성화"""
    global label

    # Label에 마우스 이벤트 바인딩
    label.bind("<ButtonPress-1>", on_mouse_press)
    label.bind("<B1-Motion>", on_mouse_drag)
    label.bind("<ButtonRelease-1>", on_mouse_release)

def on_mouse_press(event):
    """마우스 클릭 시 드래그 시작점 설정"""
    global start_point
    start_point = (event.x, event.y)  # Label 좌표 그대로 사용


def on_mouse_drag(event):
    """마우스 드래그 중 선택 영역 표시"""
    global start_point

    if not start_point:
        return

    # 현재 이미지를 OpenCV 형식으로 가져옴
    img = get_image().copy()

    # 현재 드래그 위치에 사각형 그리기
    x1, y1 = convert_coords(*start_point)
    x2, y2 = convert_coords(event.x, event.y)

    if None in (x1, y1, x2, y2):
        return

    # 선택 영역 표시
    temp_img = img.copy()
    cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    update_label(temp_img)


def on_mouse_release(event):
    """마우스 드래그 종료 시 선택 영역 확정"""
    global start_point

    # 선택 영역 좌표 저장
    x1, y1 = convert_coords(*start_point)
    x2, y2 = convert_coords(event.x, event.y)

    if None in (x1, y1, x2, y2):
        print("Invalid selection area")
        return

    # 선택 영역 좌표 출력
    print(f"선택 영역: ({x1}, {y1}) -> ({x2}, {y2})")
    set_selection((x1, y1, x2, y2))
    start_point = None  # 선택 초기화

def draw_rectangle(img, coords):
    """Tkinter Label의 이미지 위에 사각형을 그리기"""
    x1, y1, x2, y2 = coords
    img_draw = Image.fromarray(np.array(img))  # 이미지를 PIL 이미지로 변환
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
    return ImageTk.PhotoImage(img_draw)

def convert_coords(x, y):
    """Label 좌표를 원본 이미지 좌표로 변환"""
    img = get_image()
    if img is None:
        return None, None

    h, w = img.shape[:2]  # 원본 이미지 크기

    # Label 좌표에서 축소된 이미지 좌표로 변환
    x_within_image = x - x_offset
    y_within_image = y - y_offset

    # 축소된 이미지 안에 포함되지 않으면 좌표 무효화
    if x_within_image < 0 or y_within_image < 0 or x_within_image > displayed_width or y_within_image > displayed_height:
        return None, None

    # 축소된 이미지 좌표를 원본 이미지 좌표로 변환
    real_x = int(x_within_image * (w / displayed_width))
    real_y = int(y_within_image * (h / displayed_height))
    return real_x, real_y

def update_label(img):
    """Label에 OpenCV 이미지를 업데이트"""
    global label, tk_img

    # OpenCV 이미지를 Pillow로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(img_pil)

    # Label에 이미지 설정
    label.config(image=tk_img)
    label.image = tk_img  # 참조 유지
'''

# def display_image():
#     """현재 이미지를 Label에 표시 (600x600에 맞춰 원본 비율 유지)"""
#     global label, tk_img, displayed_width, displayed_height, x_offset, y_offset, scale  # scale 추가
#     img = image_effect.get_image()

#     if img is None:
#         print("Error: No image to display.")
#         return

#     # 원본 비율 유지하면서 최대 600x600 크기로 축소
#     max_size = 800
#     h, w = img.shape[:2]
#     scale = min(max_size / w, max_size / h)  # 축소 비율 계산
#     displayed_width, displayed_height = int(w * scale), int(h * scale)

#     # 이미지 크기 조정
#     resized_img = cv2.resize(img, (displayed_width, displayed_height), interpolation=cv2.INTER_AREA)

#     # Label에서 여백 계산 (중앙 정렬)
#     x_offset = (max_size - displayed_width) // 2
#     y_offset = (max_size - displayed_height) // 2


#     # OpenCV 이미지를 Tkinter에서 사용할 수 있도록 변환
#     img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_rgb)
#     tk_img = ImageTk.PhotoImage(img_pil)

#     # Label에 이미지 업데이트
#     label.config(image=tk_img, bg="gray")
#     label.image = tk_img  # 참조 유지


def display_image():
    """현재 이미지를 Label에 표시 (800x800에 맞춰 원본 비율 유지)"""
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
