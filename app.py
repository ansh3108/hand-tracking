import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands
capture = cv2.VideoCapture(0)

audio_devices = AudioUtilities.GetSpeakers()
interface = audio_devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
audio_volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_limits = audio_volume.GetVolumeRange()
volume_min = volume_limits[0]
volume_max = volume_limits[1]

with hands_module.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    previous_distance = None
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Frame read error. Exiting...")
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]
                thumb_x = thumb_tip.x * frame.shape[1]
                thumb_y = thumb_tip.y * frame.shape[0]
                index_x = index_tip.x * frame.shape[1]
                index_y = index_tip.y * frame.shape[0]
                
                distance = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
                
                if previous_distance is not None:
                    volume_change = (distance - previous_distance) * 0.5
                    current_volume = audio_volume.GetMasterVolumeLevel()
                    new_volume = max(min(current_volume + volume_change, volume_max), volume_min)
                    audio_volume.SetMasterVolumeLevel(new_volume, None)
                
                previous_distance = distance
                
                drawing_utils.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)
        
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
