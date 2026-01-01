"""
Configuration settings
"""

import os
from pathlib import Path

class Config:
    """Application configuration."""
    
    def __init__(self):
        """Initialize configuration."""
        
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.ASSETS_DIR = self.PROJECT_ROOT / "assets" / "memes"
        self.OUTPUT_DIR = self.PROJECT_ROOT / "output"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        self._create_directories()
        
        # Camera settings
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 1280
        self.CAMERA_FPS = 30
        self.CAMERA_INDEX = 0
        
        # Display settings
        self.SHOW_LANDMARKS = False
        self.REACTION_WINDOW_WIDTH = 1280
        self.REACTION_WINDOW_HEIGHT = 720
        
        # MediaPipe settings
        self.MP_MIN_DETECTION_CONFIDENCE = 0.5
        self.MP_MIN_TRACKING_CONFIDENCE = 0.5
        self.MP_MAX_NUM_FACES = 1
        self.MP_MAX_NUM_HANDS = 2
        self.MP_REFINE_LANDMARKS = True
        
        # Detection thresholds
        self.MOUTH_OPEN_RATIO = 0.5
        self.EYE_CLOSED_RATIO = 0.2
        self.TONGUE_OUT_RATIO = 0.3
        self.EYEBROW_RAISE_DISTANCE = 0.08
        self.PEACE_SIGN_CONFIDENCE = 0.7
        self.FLEX_ANGLE_THRESHOLD = 50.0
        
        # Timing settings
        self.GESTURE_HOLD_TIME = 0.3
        self.REACTION_MIN_DURATION = 1.0
        
        # Logging settings
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.LOG_FILE = self.LOGS_DIR / 'gesture_detector.log'
        
        # Performance settings
        self.ENABLE_GPU = False
        self.FRAME_SKIP = 1
        
        # Debug settings
        self.DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
        self.SAVE_DEBUG_FRAMES = False
        self.DEBUG_OUTPUT_DIR = self.OUTPUT_DIR / 'debug'
        
        # Gesture descriptions
        self.GESTURE_NAMES = {
            'tongue_out': 'Tongue Out',
            'eyes_closed': 'Eyes Closed',
            'mouth_open': 'Mouth Open',
            'eyebrows_raised': 'Eyebrows Raised',
            'peace_sign': 'Peace Sign',
            'flex_pose': 'Flex Pose',
            'freaky_combo': 'Freaky Combo'
        }
        
        # UI Colors (BGR format)
        self.COLORS = {
            'primary': (255, 100, 0),
            'success': (0, 255, 0),
            'warning': (0, 165, 255),
            'danger': (0, 0, 255),
            'info': (255, 255, 0),
            'text': (255, 255, 255),
            'background': (30, 30, 30)
        }
        
        # Window names
        self.MAIN_WINDOW_NAME = "Gesture Detector"
        self.REACTION_WINDOW_NAME = "Meme Reaction"
        
        # Features
        self.FEATURES = {
            'auto_save_reactions': False,
            'gesture_history': True,
            'gif_animations': True,
            'face_blur': False,
            'background_removal': False,
            'multi_face': False,
            'gesture_combos': True
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.ASSETS_DIR,
            self.OUTPUT_DIR,
            self.LOGS_DIR,
            self.OUTPUT_DIR / 'screenshots',
            self.OUTPUT_DIR / 'debug'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_gesture_threshold(self, gesture_type: str) -> float:
        """Get threshold for gesture type."""
        thresholds = {
            'mouth_open': self.MOUTH_OPEN_RATIO,
            'eye_closed': self.EYE_CLOSED_RATIO,
            'tongue_out': self.TONGUE_OUT_RATIO,
            'eyebrow_raise': self.EYEBROW_RAISE_DISTANCE,
            'peace_sign': self.PEACE_SIGN_CONFIDENCE,
            'flex_pose': self.FLEX_ANGLE_THRESHOLD
        }
        
        return thresholds.get(gesture_type, 0.5)
    
    def update_from_env(self):
        """Update config from environment variables."""
        env_mappings = {
            'CAMERA_INDEX': ('CAMERA_INDEX', int),
            'DEBUG_MODE': ('DEBUG', lambda x: x.lower() == 'true'),
            'SHOW_LANDMARKS': ('SHOW_LANDMARKS', lambda x: x.lower() == 'true'),
            'LOG_LEVEL': ('LOG_LEVEL', str)
        }
        
        for attr, (env_var, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                except (ValueError, TypeError):
                    pass
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

config = Config()
