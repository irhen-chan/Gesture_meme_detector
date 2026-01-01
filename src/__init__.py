"""
Gesture Meme Detector
Core modules for gesture detection and meme reactions
"""

from .gesture_detector import GestureDetector
from .reaction_manager import ReactionManager
from .config import Config, config
from .utils import (
    setup_logging,
    check_dependencies,
    check_camera,
    save_screenshot,
    FPSCounter,
    print_banner
)

__version__ = "2.0.0"
__author__ = "Snehar"

__all__ = [
    'GestureDetector',
    'ReactionManager',
    'Config',
    'config',
    'setup_logging',
    'check_dependencies',
    'check_camera',
    'save_screenshot',
    'FPSCounter',
    'print_banner'
]
