import datetime

import pandas as pd
import numpy as np
import cv2
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def landmark_xy(hn_landmarks, image):
    height, width, _ = image.shape
    hand_points = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                   'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                   'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    coord_dict = {}
    for point in hand_points:
        try:
            x = int(hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].x * width)
            y = int(hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].y * height)
        # z = hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].z * height
        except:
            x = np.nan
            y = np.nan
        coord_dict[point + 'x'] = x
        coord_dict[point + 'y'] = y

    return coord_dict


def bbox_landmarks(hn_landmark, image):
    padding = 20
    crop_copy = image.copy()

    x = [landmark.x for landmark in hn_landmark.landmark]
    y = [landmark.y for landmark in hn_landmark.landmark]

    coords = [min(x) * image.shape[1], max(x) * image.shape[1], min(y) * image.shape[0], max(y) * image.shape[0]]
    center = np.array([np.mean(x) * image.shape[1], np.mean(y) * image.shape[0]]).astype('int32')

    dist = [center[0] - coords[0], coords[1] - center[0], center[1] - coords[2], coords[3] - center[1]]
    bb_dim = int(max(dist) + padding)

    start_r = center[1] - bb_dim
    start_c = center[0] - bb_dim
    end_r = center[1] + bb_dim
    end_c = center[0] + bb_dim
    
    if start_r != 0 or start_c != 0:
        crop = crop_copy[start_r:end_r, start_c:end_c]
    elif start_r < 0 and start_c < 0:
        crop = crop_copy[:end_r, :end_c]

    cv2.circle(image, tuple(center), 10, (255, 0, 0), 2)  # for checking the center
    cv2.rectangle(image, (center[0] - bb_dim, center[1] - bb_dim), (center[0] + bb_dim, center[1] + bb_dim),
                  (255, 0, 0), 2)

    return crop


def frame_folder(frame, user_name, gesture, train=True):
    PATH = 'images/'
    TIME = str(time.time()) + '_' + user_name + '.png'
    if train:
        path = PATH + 'train/' + gesture + '/' + TIME
        cv2.imwrite(path, frame)

    else:
        path = PATH + 'validation/' + gesture + '/' + TIME
        cv2.imwrite(path, frame)
    print(path)

