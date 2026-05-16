import math
import time
import cv2
import mss
import numpy as np
import pyautogui

pyautogui.PAUSE = 0


class DinoBox:

    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height


def find_dinosaur():
    time.sleep(3)

    template = cv2.imread("dino.png", cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    h, w = template.shape[:2]

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        scr = sct.grab(monitor)
        screen_img = np.array(scr)
        gray_screen = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2GRAY)
        result = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= 0.75:
            dino_x = max_loc[0] + monitor["left"]
            dino_y = max_loc[1] + monitor["top"]
            return DinoBox(dino_x, dino_y, w, h)
        return None


def main():
    dino_box = find_dinosaur()
    if dino_box is None:
        return

    GAME_BOX = {
        "top": int(dino_box.top - 65),
        "left": int(dino_box.left),
        "width": 650,
        "height": int(dino_box.height + 80),
    }

    BASE_X_START = int(dino_box.width + 75)
    TRIGGER_WIDTH = 45
    Y_TOP = 30
    Y_BOTTOM = in
    t(GAME_BOX["height"] - 60)
    PIXEL_THRESHOLD = 4

    pyautogui.press("space")
    start_time = time.time()
    last_jump_time = 0

    with mss.mss() as sct:
        while True:
            scr = sct.grab(GAME_BOX)
            img = np.array(scr)

            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            edges = cv2.Canny(gray, threshold1=100, threshold2=200)

            elapsed_time = time.time() - start_time
            speed_offset = int(18 * math.log(elapsed_time + 1))
            speed_offset = min(speed_offset, 150)

            x_start = BASE_X_START + speed_offset
            x_end = x_start + TRIGGER_WIDTH

            roi = edges[Y_TOP:Y_BOTTOM, x_start:x_end]
            white_pixels = np.sum(roi > 0)

            current_time = time.time()

            if white_pixels >= PIXEL_THRESHOLD and (
                current_time - last_jump_time > 0.12
            ):
                pyautogui.press("space")
                last_jump_time = current_time


if __name__ == "__main__":
    main()
