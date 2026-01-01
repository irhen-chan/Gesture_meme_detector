"""
Gesture detection module using MediaPipe
Handles facial and hand gesture recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class GestureThresholds:
    """Detection thresholds configuration."""
    mouth_open_ratio: float = 0.5
    eye_closed_ratio: float = 0.2
    tongue_out_ratio: float = 0.3
    eyebrow_raise_distance: float = 0.08
    peace_sign_confidence: float = 0.7
    flex_angle_threshold: float = 50.0

class GestureDetector:
    """
    Gesture detection using MediaPipe.
    Detects facial expressions and hand gestures.
    """
    
    def __init__(self, config):
        """Initialize detector with MediaPipe models."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.thresholds = GestureThresholds()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.landmarks_indices = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'mouth_outer': [61, 291, 39, 269, 0, 17],
            'mouth_inner': [78, 308, 13, 14, 312, 82],
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [300, 293, 334, 296, 336],
            'nose_tip': [1, 2, 4, 5, 6],
            'tongue': [12, 15, 16, 17, 18]
        }
        
        self.logger.info("Gesture detector initialized")
    
    def detect_gestures(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """Detect gestures in frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        detected_gesture = None
        
        # Check for combo gesture first
        if (face_results.multi_face_landmarks and 
            hand_results.multi_hand_landmarks and 
            len(hand_results.multi_hand_landmarks) >= 2):
            
            face_landmarks = face_results.multi_face_landmarks[0]
            
            if (self._is_tongue_out(face_landmarks, frame.shape) and 
                self._are_hands_up(hand_results.multi_hand_landmarks, frame.shape)):
                detected_gesture = "freaky_combo"
                
                if self.config.SHOW_LANDMARKS:
                    self._draw_face_landmarks(frame, face_landmarks)
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self._draw_hand_landmarks(frame, hand_landmarks)
        
        # Check flex pose
        if not detected_gesture and pose_results.pose_landmarks:
            if self._is_flex_pose(pose_results.pose_landmarks):
                detected_gesture = "flex_pose"
                if self.config.SHOW_LANDMARKS:
                    self._draw_pose_landmarks(frame, pose_results.pose_landmarks)
        
        # Check individual gestures
        if not detected_gesture:
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                if self.config.SHOW_LANDMARKS:
                    self._draw_face_landmarks(frame, face_landmarks)
                
                if self._is_tongue_out(face_landmarks, frame.shape):
                    detected_gesture = "tongue_out"
                elif self._are_eyes_closed(face_landmarks):
                    detected_gesture = "eyes_closed"
                elif self._is_mouth_open(face_landmarks):
                    detected_gesture = "mouth_open"
                elif self._are_eyebrows_raised(face_landmarks, frame.shape):
                    detected_gesture = "eyebrows_raised"
            
            if not detected_gesture and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if self.config.SHOW_LANDMARKS:
                        self._draw_hand_landmarks(frame, hand_landmarks)
                    
                    if self._is_peace_sign(hand_landmarks):
                        detected_gesture = "peace_sign"
                        break
        
        self._add_status_text(frame, detected_gesture)
        
        return detected_gesture, frame
    
    def _is_flex_pose(self, landmarks) -> bool:
        """Detect flexing bicep pose (side chest)."""
        try:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            def calculate_angle(a, b, c):
                """Calculate angle between three points."""
                ba = np.array([a.x - b.x, a.y - b.y])
                bc = np.array([c.x - b.x, c.y - b.y])
                
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                return np.degrees(angle)
            
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            
            is_left_flexed = 30 <= left_angle <= 90
            is_right_flexed = 30 <= right_angle <= 90
            
            
            left_flex_up = left_wrist.y < left_elbow.y
            right_flex_up = right_wrist.y < right_elbow.y
            
            return (is_left_flexed and left_flex_up) or (is_right_flexed and right_flex_up)
            
        except Exception:
            return False
    
    def _is_tongue_out(self, landmarks, frame_shape) -> bool:
        """Detect tongue out."""
        h, w = frame_shape[:2]
        
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        mouth_left = landmarks.landmark[61]
        mouth_right = landmarks.landmark[291]
        
        mouth_vertical = abs(upper_lip.y - lower_lip.y) * h
        mouth_horizontal = abs(mouth_left.x - mouth_right.x) * w
        
        if mouth_horizontal > 0:
            mouth_ratio = mouth_vertical / mouth_horizontal
            return mouth_ratio > self.thresholds.tongue_out_ratio
        
        return False
    
    def _are_eyes_closed(self, landmarks) -> bool:
        """Detect closed eyes."""
        def eye_aspect_ratio(eye_landmarks, all_landmarks):
            v1 = abs(all_landmarks.landmark[eye_landmarks[1]].y - 
                    all_landmarks.landmark[eye_landmarks[5]].y)
            v2 = abs(all_landmarks.landmark[eye_landmarks[2]].y - 
                    all_landmarks.landmark[eye_landmarks[4]].y)
            
            h = abs(all_landmarks.landmark[eye_landmarks[0]].x - 
                   all_landmarks.landmark[eye_landmarks[3]].x)
            
            if h > 0:
                ear = (v1 + v2) / (2.0 * h)
                return ear
            return 1.0
        
        left_ear = eye_aspect_ratio(self.landmarks_indices['left_eye'], landmarks)
        right_ear = eye_aspect_ratio(self.landmarks_indices['right_eye'], landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear < self.thresholds.eye_closed_ratio
    
    def _is_mouth_open(self, landmarks) -> bool:
        """Detect open mouth."""
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        mouth_left = landmarks.landmark[61]
        mouth_right = landmarks.landmark[291]
        
        mouth_height = abs(upper_lip.y - lower_lip.y)
        mouth_width = abs(mouth_left.x - mouth_right.x)
        
        if mouth_width > 0:
            ratio = mouth_height / mouth_width
            return ratio > self.thresholds.mouth_open_ratio
        
        return False
    
    def _are_eyebrows_raised(self, landmarks, frame_shape) -> bool:
        """Detect raised eyebrows."""
        h = frame_shape[0]
        
        left_eyebrow_y = np.mean([landmarks.landmark[i].y 
                                  for i in self.landmarks_indices['left_eyebrow']])
        right_eyebrow_y = np.mean([landmarks.landmark[i].y 
                                   for i in self.landmarks_indices['right_eyebrow']])
        
        left_eye_y = np.mean([landmarks.landmark[i].y 
                              for i in self.landmarks_indices['left_eye']])
        right_eye_y = np.mean([landmarks.landmark[i].y 
                               for i in self.landmarks_indices['right_eye']])
        
        left_distance = (left_eye_y - left_eyebrow_y)
        right_distance = (right_eye_y - right_eyebrow_y)
        avg_distance = (left_distance + right_distance) / 2.0
        
        return avg_distance > self.thresholds.eyebrow_raise_distance
    
    def _is_peace_sign(self, landmarks) -> bool:
        """Detect peace sign."""
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
            self.mp_hands.HandLandmark.THUMB_TIP
        ]
        
        finger_bases = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP,
            self.mp_hands.HandLandmark.THUMB_CMC
        ]
        
        extended_fingers = []
        for tip, base in zip(finger_tips, finger_bases):
            tip_y = landmarks.landmark[tip].y
            base_y = landmarks.landmark[base].y
            extended_fingers.append(tip_y < base_y)
        
        is_peace = (extended_fingers[0] and extended_fingers[1] and 
                   not extended_fingers[2] and not extended_fingers[3])
        
        return is_peace
    
    def _are_hands_up(self, hand_landmarks_list, frame_shape) -> bool:
        """Detect raised hands."""
        if not hand_landmarks_list or len(hand_landmarks_list) < 2:
            return False
        
        h = frame_shape[0]
        hands_up_count = 0
        
        for hand_landmarks in hand_landmarks_list:
            wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
            
            if wrist_y < 0.4:
                hands_up_count += 1
        
        return hands_up_count >= 2
    
    def _draw_face_landmarks(self, frame, landmarks):
        """Draw face mesh."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    def _draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
    
    def _draw_pose_landmarks(self, frame, landmarks):
        """Draw pose landmarks."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing_styles.get_default_pose_landmarks_style(),
            self.mp_drawing_styles.get_default_pose_connections_style()
        )
    
    def _add_status_text(self, frame, gesture):
        """Add detection status to frame with improved UI."""
        height, width = frame.shape[:2]
        
        # Create gradient overlay at top
        overlay = frame.copy()
        gradient_height = 60
        for y in range(gradient_height):
            alpha = (gradient_height - y) / gradient_height * 0.7
            cv2.rectangle(overlay, (0, y), (width, y+1), (20, 20, 20), -1)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if gesture:
            display_text = gesture.replace('_', ' ').upper()
            
            # Status indicator
            cv2.circle(frame, (30, 30), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"DETECTED: {display_text}", 
                       (50, 35), cv2.FONT_HERSHEY_DUPLEX, 
                       0.6, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (30, 30), 8, (100, 100, 100), -1)
            cv2.putText(frame, "Ready for gesture...", 
                       (50, 35), cv2.FONT_HERSHEY_DUPLEX, 
                       0.6, (200, 200, 200), 1)
        
        # FPS counter in corner
        cv2.putText(frame, f"FPS: {int(cv2.getTickFrequency() / 1000)}", 
                   (width - 80, 35), cv2.FONT_HERSHEY_DUPLEX, 
                   0.5, (150, 150, 150), 1)
    
    def cleanup(self):
        """Clean up resources."""
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
        self.logger.info("Gesture detector cleaned up")
