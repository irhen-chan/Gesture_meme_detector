"""
Gesture Meme Detector
Real-time gesture detection with meme reactions
Author: Snehar
"""

import cv2
import sys
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.gesture_detector import GestureDetector
from src.reaction_manager import ReactionManager
from src.config import Config
from src.utils import setup_logging, check_dependencies

def main():
    """Main application loop."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Gesture Meme Detector...")
    
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them first.")
        return
    
    try:
        config = Config()
        gesture_detector = GestureDetector(config)
        reaction_manager = ReactionManager(config)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Please check if it's connected.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    logger.info("Webcam opened successfully")
    
    print("\n" + "="*60)
    print("           🎮 GESTURE MEME DETECTOR STARTED 🎮")
    print("="*60)
    print("\n📸 Available Gestures:")
    print("  • Tongue Out        😛")
    print("  • Eyes Closed       😴") 
    print("  • Mouth Open        😮")
    print("  • Eyebrows Raised   🤨")
    print("  • Peace Sign        ✌️")
    print("  • Flex Pose         💪")
    print("  • COMBO: Hands Up + Tongue = FREAKY! 🤪")
    print("\n[Q] Quit  [R] Reset  [H] Hide/Show Landmarks")
    print("="*60 + "\n")
    
    current_gesture = None
    last_gesture_time = time.time()
    gesture_hold_time = 0.3
    show_landmarks = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            gesture_detector.config.SHOW_LANDMARKS = show_landmarks
            detected_gesture, processed_frame = gesture_detector.detect_gestures(frame)
            
            current_time = time.time()
            if detected_gesture != current_gesture:
                if current_time - last_gesture_time > gesture_hold_time:
                    if detected_gesture:
                        logger.info(f"Gesture detected: {detected_gesture}")
                        reaction_manager.show_reaction(detected_gesture)
                    else:
                        reaction_manager.clear_reaction()
                    
                    current_gesture = detected_gesture
                    last_gesture_time = current_time
            
            cv2.imshow("Gesture Detector", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit command received")
                break
            elif key == ord('r') or key == ord('R'):
                reaction_manager.clear_reaction()
                print("✨ Reset reaction window")
            elif key == ord('h') or key == ord('H'):
                show_landmarks = not show_landmarks
                status = "ON" if show_landmarks else "OFF"
                print(f"👁️ Landmarks: {status}")
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        logger.info("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        reaction_manager.cleanup()
        logger.info("Application closed successfully")
        print("\n✨ Thanks for using Gesture Meme Detector!")

if __name__ == "__main__":
    main()
