import cv2
import numpy as np

def process(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate(deque):
    frame_stack = np.array(deque)
    return np.transpose(frame_stack, (1, 2, 0))
