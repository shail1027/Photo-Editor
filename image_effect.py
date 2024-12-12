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
        
        
def retro_filter():  # 레트로 필터
    global current_image
    # 누리끼리한 2010년대 필터 느낌
    b, g, r = cv2.split(current_image)  # BRG 채널 분리
    r = cv2.add(r, 50)
    g = cv2.add(g, 30)
    current_image = cv2.merge((b, g, r))  # 채널 병합


def adjust_contrast(n, beta=0):
    """대비 조정 함수 (예외 처리 포함)"""
    global current_image

    if current_image is None:
        print("Error: No image to adjust.")
        return

    try:
        # 입력 값 검증
        if not isinstance(n, (int, float)):
            raise ValueError("Contrast multiplier (alpha) must be an integer or a float.")
        if not isinstance(beta, (int, float)):
            raise ValueError("Brightness offset (beta) must be an integer or a float.")
        if n <= 0:
            raise ValueError("Contrast multiplier (alpha) must be greater than 0.")

        # 대비 조정
        temp = cv2.convertScaleAbs(current_image, alpha=n, beta=beta)

        # 값이 0~255 범위를 벗어나지 않도록 자동 처리
        current_image = np.clip(temp, 0, 255).astype(np.uint8)
        print(f"Contrast adjusted with alpha={n}, beta={beta}.")

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error adjusting contrast: {e}")



def adjust_saturation(n):
    """채도 조정 함수 (예외 처리 포함)"""
    global current_image

    if current_image is None:
        print("Error: No image to adjust.")
        return

    try:
        # n 값 검증
        if not isinstance(n, (int, float)):
            raise ValueError("Saturation multiplier must be an integer or a float.")
        if n < 0:
            raise ValueError("Saturation multiplier cannot be negative.")

        # 이미지 HSV 변환
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

        # 채도 조정
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], n)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # 채도 값은 0~255 범위 유지

        # HSV를 BGR로 변환하여 이미지 업데이트
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        print(f"Saturation adjusted with multiplier {n}.")

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Error adjusting saturation: {e}")


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


def custom_filter():
    """
    사용자 지정 필터 적용:
    휘도 -20, 하이라이트 -40, 대비 +60, 채도 -30, 색 선명도 +15, 필터 (모노 or 느와르) +40
    """
    global current_image

    if current_image is None:
        print("Error: No image to apply the custom filter.")
        return

    try:
        # Step 1: 휘도 조정 (-20)
        brightness_adjustment = -20
        temp = current_image.astype(np.int16) + brightness_adjustment
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        current_image = temp

        # Step 2: 하이라이트 조정 (-40)
        highlight_adjustment = -40
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) + highlight_adjustment, 0, 255).astype(np.uint8)
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 3: 대비 조정 (+60)
        contrast_adjustment = 1.6  # 대비 조정 값 (alpha)
        beta = 0  # 밝기 보정 값
        current_image = cv2.convertScaleAbs(current_image, alpha=contrast_adjustment, beta=beta)

        # Step 4: 채도 조정 (-30)
        saturation_adjustment = 0.7  # 채도 조정 비율
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_adjustment, 0, 255).astype(np.uint8)
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 5: 색 선명도 (+15)
        sharpen_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        current_image = cv2.filter2D(current_image, -1, sharpen_kernel)

    except Exception as e:
        print(f"Error applying custom filter: {e}")


def temp_filter():
    """
    통합 필터 함수:
    1. 대비 +60
    2. 채도 -20
    3. 모션 블러 적용
    """
    global current_image

    if current_image is None:
        print("Error: No image to apply filter.")
        return

    try:
        # Step 1: 대비 +60
        contrast = 1.6  # 대비 값 (1.0 = 기본값, 1.6 = 60% 증가)
        beta = 0  # 밝기 보정 값
        current_image = cv2.convertScaleAbs(current_image, alpha=contrast, beta=beta)
        print(f"Step 1: Contrast increased by 60% (alpha={contrast}, beta={beta}).")

        # Step 2: 채도 -20
        saturation_adjustment = 0.8  # 채도 비율 (1.0 = 기본값, 0.8 = 20% 감소)
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_adjustment, 0, 255).astype(np.uint8)
        current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        print(f"Step 2: Saturation decreased by 20% (scale={saturation_adjustment}).")

        # Step 3: 모션 블러 적용
        kernel_size = 7  # 모션 블러 커널 크기
        motion_blur_kernel = np.zeros((kernel_size, kernel_size))
        np.fill_diagonal(motion_blur_kernel, 1)  # 대각선 방향으로 값 설정
        motion_blur_kernel /= kernel_size  # 정규화
        current_image = cv2.filter2D(current_image, -1, motion_blur_kernel)
        print(f"Step 3: Motion blur applied with kernel size {kernel_size}.")

        print("All filters applied successfully.")

    except Exception as e:
        print(f"Error applying filters: {e}")


def remove_salt_pepper(image, center):
    """
    Resized 이미지에서 Salt-and-Pepper 잡티를 제거
    :param image: 처리할 Resized 이미지
    :param center: 클릭한 중심 좌표 (x, y)
    """
    global current_image
    try:
        # 중심 좌표와 커널 크기 설정
        x, y = center
        kernel_size = 7  # 커널 크기
        half_k = kernel_size // 2

        # 이미지 크기 가져오기
        h, w = image.shape[:2]

        # ROI 경계 설정
        x_start = max(0, x - half_k)
        y_start = max(0, y - half_k)
        x_end = min(w, x + half_k + 1)
        y_end = min(h, y + half_k + 1)

        # ROI 추출
        roi = image[y_start:y_end, x_start:x_end]

        # 중간값 필터 적용
        filtered_roi = cv2.medianBlur(roi, kernel_size)

        # 필터링된 결과를 Resized 이미지에 반영
        image[y_start:y_end, x_start:x_end] = filtered_roi
        print(f"Applied median filter to region: ({x_start}, {y_start}) to ({x_end}, {y_end})")

        current_image =  image

    except Exception as e:
        print(f"Error in remove_salt_pepper: {e}")
        current_image =  image





def apply_makeup(image, start_point, end_point, color='red', intensity=0.5):
    """
    특정 영역의 색상을 강조 (브러시 스타일)
    :param image: 입력 이미지
    :param start_point: 시작 좌표
    :param end_point: 끝 좌표
    :param color: 적용할 색상 ('red', 'pink', etc.)
    :param intensity: 색상 강도 (0 ~ 1)
    :return: 색상 강조된 이미지
    """
    global current_image
    try:
        # 중심 좌표 설정
        x, y = end_point
        radius = 10  # 브러시 반지름

        # ROI 경계 설정
        x_start, x_end = max(0, x - radius), min(image.shape[1], x + radius)
        y_start, y_end = max(0, y - radius), min(image.shape[0], y + radius)

        # ROI 추출
        roi = image[y_start:y_end, x_start:x_end]

        # 색상 강조 (HSV 색상 공간으로 변환)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color == 'red':
            hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180  # Hue 증가 (빨간색 강조)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(255 * intensity), 0, 255)  # 채도 증가

        # 블렌딩 처리
        adjusted_roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        adjusted_roi = cv2.GaussianBlur(adjusted_roi, (5, 5), sigmaX=2)

        # 원본 이미지에 반영
        image[y_start:y_end, x_start:x_end] = adjusted_roi

        current_image =  image

    except Exception as e:
        print(f"Error in apply_makeup: {e}")
        return None
