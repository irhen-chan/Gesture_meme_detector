"""
Placeholder meme generator
Creates test memes for gestures
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def create_meme_placeholders():
    """Create placeholder memes for all gestures."""
    
    memes_dir = Path("assets/memes")
    memes_dir.mkdir(parents=True, exist_ok=True)
    
    memes = {
        'tongue_out': {
            'text': 'TONGUE',
            'subtitle': 'Being Silly',
            'color': (147, 20, 255),
            'bg_gradient': [(255, 200, 255), (200, 100, 200)]
        },
        'eyes_closed': {
            'text': 'SLEEPY',
            'subtitle': 'Zzz Time',
            'color': (255, 100, 0),
            'bg_gradient': [(255, 230, 200), (200, 180, 150)]
        },
        'mouth_open': {
            'text': 'SHOCKED',
            'subtitle': 'Mind Blown',
            'color': (0, 165, 255),
            'bg_gradient': [(200, 230, 255), (150, 180, 200)]
        },
        'eyebrows_raised': {
            'text': 'SUS',
            'subtitle': 'Very Suspicious',
            'color': (0, 255, 255),
            'bg_gradient': [(200, 255, 255), (150, 200, 200)]
        },
        'peace_sign': {
            'text': 'PEACE',
            'subtitle': 'Vibing',
            'color': (0, 255, 0),
            'bg_gradient': [(200, 255, 200), (150, 200, 150)]
        },
        'flex_pose': {
            'text': 'FLEX',
            'subtitle': 'Strong',
            'color': (255, 0, 150),
            'bg_gradient': [(255, 200, 230), (200, 100, 180)]
        },
        'freaky_combo': {
            'text': 'FREAKY',
            'subtitle': 'Maximum Chaos',
            'color': (255, 0, 255),
            'bg_gradient': [(255, 200, 255), (200, 100, 200)]
        }
    }
    
    print("\nCreating placeholder memes...")
    print("-" * 40)
    
    for gesture, info in memes.items():
        img = np.zeros((600, 600, 3), np.uint8)
        
        # Gradient background
        for y in range(600):
            gradient_factor = y / 600
            color = tuple(int(info['bg_gradient'][0][i] * (1 - gradient_factor) + 
                            info['bg_gradient'][1][i] * gradient_factor) for i in range(3))
            img[y, :] = color
        
        # Decorative elements
        for _ in range(10):
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 550)
            radius = np.random.randint(10, 40)
            color = tuple(np.random.randint(100, 255) for _ in range(3))
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(img, (x, y), radius + 2, info['color'], 2)
        
        # Main text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = info['text']
        text_size = cv2.getTextSize(text, font, 3, 5)[0]
        text_x = (600 - text_size[0]) // 2
        text_y = 300
        
        cv2.putText(img, text, (text_x + 4, text_y + 4), 
                   font, 3, (0, 0, 0), 5)
        cv2.putText(img, text, (text_x, text_y), 
                   font, 3, (255, 255, 255), 5)
        
        # Subtitle
        subtitle = info['subtitle']
        subtitle_size = cv2.getTextSize(subtitle, font, 1.2, 2)[0]
        subtitle_x = (600 - subtitle_size[0]) // 2
        subtitle_y = text_y + 80
        
        cv2.putText(img, subtitle, (subtitle_x, subtitle_y), 
                   font, 1.2, (255, 255, 255), 2)
        
        # Border
        cv2.rectangle(img, (20, 20), (580, 580), info['color'], 5)
        cv2.rectangle(img, (10, 10), (590, 590), (255, 255, 255), 2)
        
        # Corner decorations
        corner_size = 50
        cv2.line(img, (10, 10), (10 + corner_size, 10), info['color'], 3)
        cv2.line(img, (10, 10), (10, 10 + corner_size), info['color'], 3)
        cv2.line(img, (590, 10), (590 - corner_size, 10), info['color'], 3)
        cv2.line(img, (590, 10), (590, 10 + corner_size), info['color'], 3)
        cv2.line(img, (10, 590), (10 + corner_size, 590), info['color'], 3)
        cv2.line(img, (10, 590), (10, 590 - corner_size), info['color'], 3)
        cv2.line(img, (590, 590), (590 - corner_size, 590), info['color'], 3)
        cv2.line(img, (590, 590), (590, 590 - corner_size), info['color'], 3)
        
        filename = memes_dir / f"{gesture}.png"
        cv2.imwrite(str(filename), img)
        print(f"Created: {filename}")
    
    print("-" * 40)
    print("All placeholder memes created!")
    
    return True

def main():
    """Main function."""
    print("\n" + "="*50)
    print("   MEME PLACEHOLDER GENERATOR")
    print("="*50)
    
    try:
        create_meme_placeholders()
        
        print("\n" + "="*50)
        print("Setup complete!")
        print("\nYou can now run: python main.py")
        print("\nTo add real memes:")
        print("1. Find funny GIFs/images")
        print("2. Name them: gesture_name.gif")
        print("3. Place in assets/memes/")
        print("="*50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you're in the project directory!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
