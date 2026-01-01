"""
Reaction manager for displaying memes
Handles meme/GIF display in reaction window
"""

import cv2
import numpy as np
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict
import imageio

class ReactionManager:
    """Manages meme reactions display."""
    
    def __init__(self, config):
        """Initialize reaction manager."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.window_name = "Meme Reaction"
        self.current_reaction = None
        self.reaction_cache = {}
        self.gif_frames = {}
        self.gif_current_frame = {}
        self.gif_threads = {}
        
        self.reaction_width = 1000
        self.reaction_height = 1000
        
        self._load_reactions()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.reaction_width, self.reaction_height)
        
        self.clear_reaction()
        
        self.logger.info("Reaction manager initialized")
    
    def _load_reactions(self):
        """Load reaction media files."""
        assets_dir = Path(self.config.ASSETS_DIR)
        
        reaction_files = {
            'tongue_out': 'tongue_out.gif',
            'eyes_closed': 'eyes_closed.gif',
            'mouth_open': 'mouth_open.gif',
            'eyebrows_raised': 'eyebrows_raised.gif',
            'peace_sign': 'peace_sign.gif',
            'flex_pose': 'flex_pose.png',
            'freaky_combo': 'freaky_combo.jpg'
        }
        
        alternative_formats = ['.gif', '.png', '.jpg', '.jpeg', '.webp']
        
        for gesture, filename in reaction_files.items():
            file_path = assets_dir / filename
            
            if not file_path.exists():
                for ext in alternative_formats:
                    alt_path = assets_dir / f"{gesture}{ext}"
                    if alt_path.exists():
                        file_path = alt_path
                        break
            
            if file_path.exists():
                try:
                    if file_path.suffix.lower() == '.gif':
                        self._load_gif(gesture, file_path)
                    else:
                        self._load_image(gesture, file_path)
                    
                    self.logger.info(f"Loaded reaction for {gesture}: {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
                    self._create_placeholder(gesture)
            else:
                self.logger.warning(f"Reaction file not found for {gesture}")
                self._create_placeholder(gesture)
    
    def _load_gif(self, gesture: str, file_path: Path):
        """Load GIF frames."""
        try:
            gif = imageio.get_reader(file_path)
            frames = []
            
            for frame in gif:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, 
                                          (self.reaction_width, self.reaction_height))
                frames.append(frame_resized)
            
            if frames:
                self.gif_frames[gesture] = frames
                self.gif_current_frame[gesture] = 0
                self.logger.info(f"Loaded {len(frames)} frames for {gesture}")
            
        except Exception as e:
            self.logger.error(f"Error loading GIF {file_path}: {e}")
            self._create_placeholder(gesture)
    
    def _load_image(self, gesture: str, file_path: Path):
        """Load static image."""
        try:
            image = cv2.imread(str(file_path))
            if image is not None:
                image_resized = cv2.resize(image, 
                                          (self.reaction_width, self.reaction_height))
                self.reaction_cache[gesture] = image_resized
            else:
                raise ValueError(f"Could not read image: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading image {file_path}: {e}")
            self._create_placeholder(gesture)
    
    def _create_placeholder(self, gesture: str):
        """Create placeholder meme with modern UI."""
        placeholders = {
            'tongue_out': {
                'emoji': '😛',
                'text': 'TONGUE OUT',
                'color': (147, 20, 255),
                'gradient': [(60, 20, 80), (147, 20, 255)]
            },
            'eyes_closed': {
                'emoji': '😴',
                'text': 'SLEEPY',
                'color': (255, 100, 0),
                'gradient': [(80, 40, 20), (255, 100, 0)]
            },
            'mouth_open': {
                'emoji': '😮',
                'text': 'SHOCKED',
                'color': (0, 165, 255),
                'gradient': [(20, 60, 80), (0, 165, 255)]
            },
            'eyebrows_raised': {
                'emoji': '🤨',
                'text': 'SUSPICIOUS',
                'color': (0, 255, 255),
                'gradient': [(20, 80, 80), (0, 255, 255)]
            },
            'peace_sign': {
                'emoji': '✌️',
                'text': 'PEACE',
                'color': (0, 255, 0),
                'gradient': [(20, 60, 20), (0, 255, 0)]
            },
            'flex_pose': {
                'emoji': '💪',
                'text': 'FLEX',
                'color': (255, 0, 150),
                'gradient': [(80, 20, 60), (255, 0, 150)]
            },
            'freaky_combo': {
                'emoji': '🤪',
                'text': 'FREAKY',
                'color': (255, 0, 255),
                'gradient': [(80, 20, 80), (255, 0, 255)]
            }
        }
        
        info = placeholders.get(gesture, {
            'emoji': '🎭',
            'text': gesture.upper(),
            'color': (255, 255, 255),
            'gradient': [(40, 40, 40), (100, 100, 100)]
        })
        
        placeholder = np.zeros((self.reaction_height, self.reaction_width, 3), np.uint8)
        
        # Gradient background
        for y in range(self.reaction_height):
            t = y / self.reaction_height
            color = tuple(int(info['gradient'][0][i] * (1 - t) + 
                            info['gradient'][1][i] * t) for i in range(3))
            placeholder[y, :] = color
        
        # Modern geometric pattern
        for i in range(5):
            x = np.random.randint(100, 500)
            y = np.random.randint(100, 500)
            size = np.random.randint(30, 80)
            alpha = 0.3
            overlay = placeholder.copy()
            cv2.rectangle(overlay, (x, y), (x + size, y + size), info['color'], -1)
            cv2.addWeighted(overlay, alpha, placeholder, 1 - alpha, 0, placeholder)
        
        # Main text with modern style
        font = cv2.FONT_HERSHEY_DUPLEX
        text = info['text']
        text_size = cv2.getTextSize(text, font, 2.5, 3)[0]
        text_x = (self.reaction_width - text_size[0]) // 2
        text_y = self.reaction_height // 2
        
        # Glow effect
        for i in range(3):
            cv2.putText(placeholder, text, (text_x, text_y), 
                       font, 2.5, info['color'], 6 - i*2)
        cv2.putText(placeholder, text, (text_x, text_y), 
                   font, 2.5, (255, 255, 255), 2)
        
        # Emoji representation
        emoji_text = f"[{info['emoji']}]"
        emoji_size = cv2.getTextSize(emoji_text, font, 1.2, 2)[0]
        emoji_x = (self.reaction_width - emoji_size[0]) // 2
        emoji_y = text_y + 80
        
        cv2.putText(placeholder, emoji_text, (emoji_x, emoji_y), 
                   font, 1.2, (200, 200, 200), 2)
        
        # Modern border
        border_thickness = 2
        cv2.rectangle(placeholder, (0, 0), 
                     (self.reaction_width-1, self.reaction_height-1), 
                     info['color'], border_thickness)
        
        self.reaction_cache[gesture] = placeholder
    
    def show_reaction(self, gesture: str):
        """Display reaction for detected gesture."""
        if self.current_reaction == gesture:
            return
        
        self._stop_gif_animation()
        
        self.current_reaction = gesture
        
        if gesture in self.gif_frames:
            self._start_gif_animation(gesture)
        elif gesture in self.reaction_cache:
            cv2.imshow(self.window_name, self.reaction_cache[gesture])
        else:
            self.logger.warning(f"No reaction found for gesture: {gesture}")
            self.clear_reaction()
    
    def _start_gif_animation(self, gesture: str):
        """Start GIF animation thread."""
        def animate():
            frames = self.gif_frames[gesture]
            frame_count = len(frames)
            frame_index = 0
            
            while gesture in self.gif_threads and self.gif_threads[gesture]:
                cv2.imshow(self.window_name, frames[frame_index])
                frame_index = (frame_index + 1) % frame_count
                time.sleep(0.033)
        
        self.gif_threads[gesture] = True
        thread = threading.Thread(target=animate, daemon=True)
        thread.start()
    
    def _stop_gif_animation(self):
        """Stop running GIF animations."""
        for gesture in list(self.gif_threads.keys()):
            self.gif_threads[gesture] = False
        
        self.gif_threads.clear()
        time.sleep(0.05)
    
    def clear_reaction(self):
        """Clear reaction window with modern UI."""
        self.current_reaction = None
        self._stop_gif_animation()
        
        default_screen = np.zeros((self.reaction_height, self.reaction_width, 3), np.uint8)
        
        # Dark gradient background
        for y in range(self.reaction_height):
            color_value = int(20 + (y / self.reaction_height) * 30)
            default_screen[y, :] = (color_value, color_value, color_value)
        
        # Animated dots effect
        current_time = int(time.time() * 2) % 4
        dots = "." * current_time
        
        # Modern typography
        font = cv2.FONT_HERSHEY_DUPLEX
        text1 = "Waiting for yo dum ass" + dots
        text2 = "!"
        
        text1_size = cv2.getTextSize(text1, font, 0.8, 1)[0]
        text2_size = cv2.getTextSize(text2, font, 0.6, 1)[0]
        
        text1_x = (self.reaction_width - text1_size[0]) // 2
        text1_y = self.reaction_height // 2 - 20
        
        text2_x = (self.reaction_width - text2_size[0]) // 2
        text2_y = self.reaction_height // 2 + 20
        
        cv2.putText(default_screen, text1, (text1_x, text1_y), 
                   font, 0.8, (100, 100, 100), 1)
        cv2.putText(default_screen, text2, (text2_x, text2_y), 
                   font, 0.6, (80, 80, 80), 1)
        
        # Subtle border
        cv2.rectangle(default_screen, (0, 0), 
                     (self.reaction_width-1, self.reaction_height-1), 
                     (50, 50, 50), 1)
        
        cv2.imshow(self.window_name, default_screen)
    
    def cleanup(self):
        """Clean up resources."""
        self._stop_gif_animation()
        cv2.destroyWindow(self.window_name)
        self.logger.info("Reaction manager cleaned up")
