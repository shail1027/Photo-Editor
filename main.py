from ui import ui
import tkinter as tk

"""
2023216083 이예빈
영상처리-기말고사 프로젝트 1차 제출

프로젝트의 메인 실행 파일입니다.
"""

# Tkinter로 메인 윈도우 생성
root = tk.Tk()
root.title("영상처리-기말프로젝트")  # 윈도우 타이틀
root.geometry("1100x800")  # 윈도우 크기
ui(root)  # ui.py파일의 ui함수 호출
root.mainloop()
