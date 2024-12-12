import cv2
import numpy as np
from PIL import Image, ImageTk

# 전역 변수
current_image = None  # 현재 이미지
original_image = None  # 원본 이미지
selection = None

undo_stack = []
redo_stack = []

def push_to_undo(img):
    """현재 상태를 undo 스택에 저장"""
    global undo_stack
    if img is not None:
        undo_stack.append(img.copy())

def pop_from_undo():
    """undo 스택에서 상태를 가져옴"""
    global undo_stack
    if undo_stack:
        return undo_stack.pop()
    return None

def push_to_redo(img):
    """현재 상태를 redo 스택에 저장"""
    global redo_stack
    if img is not None:
        redo_stack.append(img.copy())

def pop_from_redo():
    """redo 스택에서 상태를 가져옴"""
    global redo_stack
    if redo_stack:
        return redo_stack.pop()
    return None


def set_image(img):  # 현재 이미지& 원본 이미지 설정
    global current_image, original_image
    current_image = img
    original_image = img.copy()  # 원본 이미지 저장


def get_image():  # 현재 이미지 반환
    return current_image


def set_selection(cord):
    global selection
    selection = cord
    

def apply_to_selection_or_full(effect_func):
    global current_image, selection
    if current_image is None:
        return

    if selection:
        x1, y1, x2, y2 = selection
        roi = current_image[y1:y2, x1:x2]  # 선택 영역 가져오기
        roi = effect_func(roi)  # 선택 영역에 효과 적용
        current_image[y1:y2, x1:x2] = roi  # 원본 이미지에 반영
    else:
        current_image = effect_func(current_image)  # 전체 이미지에 효과 적용
        
        
def retro_filter():
    def effect(image):
        b, g, r = cv2.split(image)
        r = cv2.add(r, 50)
        g = cv2.add(g, 30)
        return cv2.merge((b, g, r))

    apply_to_selection_or_full(effect)


def vintage_filter():  # 빈티지 필터
    def effect(image) :
        
        global current_image
        # 필름노이즈 추가 및 색감조정
        b, g, r = cv2.split(current_image)  # BRG 채널 분리
        r = cv2.add(r, 40)
        g = cv2.add(g, 20)
        b = cv2.add(b, 10)
        current_image = cv2.merge((b, g, r))  # 채널 병합

        # 대비 감소 및 밝기 감소
        current_image = cv2.convertScaleAbs(current_image, alpha=0.8, beta=-10)

        # 따뜻한 갈색 톤 오버레이
        overlay = np.full_like(current_image, (20, 10, 0))  # 약간의 갈색 필터
        current_image = cv2.addWeighted(current_image, 0.9, overlay, 0.1, 0)

        # 필름 노이즈 텍스쳐 추가
        noise_texture = cv2.imread(
            "FilmNoise.png", cv2.IMREAD_COLOR
        )  # 노이즈 텍스처 크기 조정
        noise_texture = cv2.resize(
            noise_texture, (current_image.shape[1], current_image.shape[0])
        )
        # 텍스쳐 합성
        current_image = cv2.addWeighted(current_image, 0.95, noise_texture, 0.3, 0)
        return current_image
    apply_to_selection_or_full(effect)


def adjust_brightness(n):  # 밝기 조정
    global current_image
    temp = current_image + n
    temp = np.clip(temp, 0, 255).astype(np.uint8) 
    current_image = temp

def adjust_contrast(n):  # 대비 조정
    global current_image
    current_image = cv2.convertScaleAbs(current_image, alpha=n, beta=0)


def adjust_saturation(n):  # 채도 조정
    global current_image
    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], n)
    current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_hue(hue_shift):
    """색조 조정 함수 (예외 처리 포함)"""
    global current_image

    if current_image is None:
        print("Error: No image to adjust.")
        return

    try:
        # hue_shift 값 범위 검증
        if not isinstance(hue_shift, (int, float)):
            raise ValueError("hue_shift must be an integer or a float.")

        # 이미지 HSV 변환
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

        # Hue 범위 검증 및 조정
        hue_shift = hue_shift % 180  # Hue 범위는 0 ~ 179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # HSV를 BGR로 변환하여 이미지 업데이트
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        print(f"Hue adjusted by {hue_shift} degrees.")
    
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error adjusting hue: {e}")

def convert_to_grayscale(): # 흑백화
    global current_image
    if current_image is None:
        print("Error: No image to convert.")
        return

    # 밝기(Grayscale)를 계산
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Grayscale을 3채널로 변환하여 색상 정보 유지
    current_image = cv2.merge([gray, gray, gray])


def sharpen_filter(): # 선명 효과(샤프닝)
    global current_image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    current_image = cv2.filter2D(current_image, -1, kernel)


def edge_detection(): # 윤곽선 추출
    global current_image

    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200) 
    current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 


def blur_filter(): # 블러 효과
    global current_image
    kernel_size = 5
    current_image = cv2.GaussianBlur(current_image, (kernel_size, kernel_size), 0)


def edge_emphasize(): # 윤곽선 강조(스케치 효과)
    edge = 20 # 밝기 차이의 기준 값
    global current_image
    height, width, _ = current_image.shape # 현재 이미지의 높이, 너비
    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY) # 밝기 정보만 사용할거기때문에 흑백으로 변환
    temp = np.copy(current_image) # 기존 이미지 복사

    for col in range(height): # 이미지의 픽셀을 순회하며 윤곽선을 계산
        for row in range(width):
            mono = int(gray_image[col, row])

            # 오른쪽과 아래쪽 픽셀을 비교하기때문에 이미지의 경계를 확인
            if row + 1 < width and col + 1 < height: # 만약 경계에 걸리지 않는다면
                right = int(gray_image[col, row + 1]) # 오른쪽 픽셀의 밝기 값
                bottom = int(gray_image[col + 1, row]) # 아래 픽셀의 밝기 값

                diff_right = abs(mono - right) # 현재 픽셀과 오른쪽 픽셀의 밝기 차
                diff_bottom = abs(mono - bottom) # 현재 픽셀과 아래 픽셀의 밝기 차

                if diff_right > edge: # 만약 기준값(edge)보다 크다면 윤곽선으로 간주한다
                        temp[col, row] = [diff_right] * 3
                elif diff_bottom > edge:
                        temp[col, row] = [diff_bottom] * 3
                else: # 윤곽선이 아닐 경우 원래 픽셀 값을 유지
                        temp[col, row] = current_image[col, row]
            else: # 경계 픽셀은 원본 값을 유지
                    temp[col, row] = current_image[col, row]

    current_image = temp  # 결과 이미지를 저장


def original(): # 원본 이미지로 되돌리기
    global current_image, original_image
    current_image = original_image
    
    
def liquify_pixels(img, start_point, end_point, strength=10, radius=20):
    """픽셀 유동화 로직"""
    global current_image
    h, w = img.shape[:2]
    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]

    output = img.copy()
    for y in range(max(0, start_point[1] - radius), min(h, start_point[1] + radius)):
        for x in range(max(0, start_point[0] - radius), min(w, start_point[0] + radius)):
            distance = np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2)
            if distance < radius:
                ratio = (radius - distance) / radius
                new_x = int(x + dx * ratio * strength / 100)
                new_y = int(y + dy * ratio * strength / 100)

                new_x = np.clip(new_x, 0, w - 1)
                new_y = np.clip(new_y, 0, h - 1)

                output[y, x] = img[new_y, new_x]
                
    current_image = output
