import pyautogui

def move_mouse(x, y):
    """
    Move the mouse to the specified coordinates (x, y).
    """
    pyautogui.moveTo(x, y)

def click_mouse():
    """
    Perform a mouse click at the current cursor location.
    """
    pyautogui.click()

def scroll_mouse(scroll_amount):
    """
    Scroll the mouse 'scroll_amount' clicks; positive for up, negative for down.
    """
    pyautogui.scroll(scroll_amount)
