# Mouse Movements Imports
import pyautogui

from constants import *


class Mouse:
    def __init__(self):
        self.screenWidth, self.screenHeight = pyautogui.size()

    def move(self, finger_tip):
        scaleFactor = 0.8
        deadZone = 0.1

        finger_tip_x = finger_tip.x
        finger_tip_y = finger_tip.y

        # Apply dead zone
        if abs(finger_tip_x - 0.5) < deadZone:
            finger_tip_x = 0.5
        if abs(finger_tip_y - 0.5) < deadZone:
            finger_tip_y = 0.5

        # Apply scaling factor and calculate screen coordinates
        pointer_x = int((finger_tip_x - 0.5) * scaleFactor * self.screenWidth + self.screenWidth / 2)
        pointer_y = int((finger_tip_y - 0.5) * scaleFactor * self.screenHeight + self.screenHeight / 2)

        # Ensure the pointer coordinates are within screen bounds
        pointer_x = max(0, min(self.screenWidth, pointer_x))
        pointer_y = max(0, min(self.screenHeight, pointer_y))

        pyautogui.moveTo(pointer_x, pointer_y)

    def scroll(self, distance: float):
        scrollValue = round(distance * SCROLL_OFFSET)

        # Scroll up/down according to the direction
        pyautogui.scroll(scrollValue)

    def click(self):
        pyautogui.click()
