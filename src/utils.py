"""
Utility functions
"""

import logging
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import json

def setup_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Configure logging."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a') if log_file else logging.NullHandler()
        ]
    )
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")

def check_dependencies() -> bool:
    """Check required dependencies."""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'imageio': 'imageio'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}\n")
        return False
    
    return True

def check_camera(camera_index: int = 0) -> bool:
    """Check camera availability."""
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception as e:
        logging.error(f"Camera check failed: {e}")
        return False

def resize_with_aspect_ratio(image: np.ndarray, 
                            width: Optional[int] = None, 
                            height: Optional[int] = None) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def save_screenshot(frame: np.ndarray, 
                   output_dir: Path, 
                   prefix: str = "screenshot") -> Optional[Path]:
    """Save screenshot with timestamp."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        logging.info(f"Screenshot saved: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Failed to save screenshot: {e}")
        return None

def add_text_overlay(frame: np.ndarray, 
                    text: str, 
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.7,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2,
                    background: bool = True) -> np.ndarray:
    """Add text overlay to frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    if background:
        padding = 5
        cv2.rectangle(frame, 
                     (x - padding, y - text_size[1] - padding),
                     (x + text_size[0] + padding, y + padding),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, 
                     (x - padding, y - text_size[1] - padding),
                     (x + text_size[0] + padding, y + padding),
                     color, 1)
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame

def create_gradient_background(width: int, 
                              height: int,
                              start_color: Tuple[int, int, int] = (50, 50, 50),
                              end_color: Tuple[int, int, int] = (100, 100, 100),
                              direction: str = 'vertical') -> np.ndarray:
    """Create gradient background."""
    image = np.zeros((height, width, 3), np.uint8)
    
    if direction == 'vertical':
        for y in range(height):
            ratio = y / height
            color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) 
                         for i in range(3))
            image[y, :] = color
    else:
        for x in range(width):
            ratio = x / width
            color = tuple(int(start_color[i] * (1 - ratio) + end_color[i] * ratio) 
                         for i in range(3))
            image[:, x] = color
    
    return image

def load_json_config(config_path: Path) -> dict:
    """Load JSON configuration."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}

def save_json_config(config: dict, config_path: Path):
    """Save configuration to JSON."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4, default=str)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

def format_time_duration(seconds: float) -> str:
    """Format duration to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def get_system_info() -> dict:
    """Get system information."""
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__,
        'processor': platform.processor(),
        'machine': platform.machine()
    }
    
    camera_count = 0
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_count += 1
            cap.release()
        else:
            break
    
    info['available_cameras'] = camera_count
    
    return info

class FPSCounter:
    """FPS counter for performance monitoring."""
    
    def __init__(self, average_over: int = 30):
        """Initialize FPS counter."""
        self.average_over = average_over
        self.frame_times = []
        self.last_time = None
    
    def update(self) -> float:
        """Update and return current FPS."""
        import time
        
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > self.average_over:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        
        return 0.0
    
    def draw_fps(self, frame: np.ndarray, position: Tuple[int, int] = (10, 30)):
        """Draw FPS on frame."""
        fps = self.update()
        fps_text = f"FPS: {fps:.1f}"
        
        if fps >= 25:
            color = (0, 255, 0)
        elif fps >= 15:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        add_text_overlay(frame, fps_text, position, color=color, background=True)

def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🎭  GESTURE MEME DETECTOR  🎭                   ║
    ║                                                               ║
    ║         Real-time Gesture Recognition with Memes             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

__all__ = [
    'setup_logging',
    'check_dependencies',
    'check_camera',
    'resize_with_aspect_ratio',
    'save_screenshot',
    'add_text_overlay',
    'create_gradient_background',
    'load_json_config',
    'save_json_config',
    'format_time_duration',
    'get_system_info',
    'FPSCounter',
    'print_banner'
]
