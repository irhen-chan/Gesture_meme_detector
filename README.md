# Gesture_meme_detector
So I built this thing that watches you through your webcam and throws memes at you when you make faces. Yeah, it's as ridiculous as it sounds.

What is this?
Remember those Instagram filters that react to your face? This is like that but with memes. Stick your tongue out, get a meme. Close your eyes, another meme. Flex your muscles like you're at the gym, boom - Hello Kitty flexing appears.
Built this using MediaPipe for the face/hand tracking and OpenCV for the video stuff. No sounds yet, but feel free to add them. 
The Gestures It Detects

- Tongue out → nailong
- Eyes closed → lebron crying
- Mouth open → shocked reaction
- Eyebrows raised → the rock
- Peace sign → chill mode
- Flex pose → jacked hello kitty (my personal favorite)
- Combo move: Hands up + Tongue → absolute freaky

# Setup

### Virtual environment (always a good idea)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

## Install stuff
pip install -r requirements.txt

## Run it
python main.py
Note: MediaPipe doesn't play nice with Python 3.13 yet, so stick with 3.10-3.12.
## How to Use

Once it's running:
- **Q** = quit (obviously)
- **R** = reset the meme window
- **H** = show/hide the tracking dots on your face (kinda creepy but cool)

Just make faces at your camera and watch the memes pop up. It's surprisingly addictive.

## Want Different Memes?

Drop any image or GIF in `assets/memes/` and name it after the gesture:
- `tongue_out.gif`
- `eyes_closed.png`
- `flex_pose.jpg`
- etc.

The app will use whatever you put there. 

## Project Structure

```
gesture-meme-detector/
├── main.py                 # starts everything
├── src/
│   ├── gesture_detector.py # the brain that detects gestures
│   ├── reaction_manager.py # shows the memes
│   ├── config.py          # tweak settings here
│   └── utils.py           # helper stuff
├── assets/
│   └── memes/            # your meme collection goes here
└── requirements.txt       # dependencies
```

## Tech Stack

- **MediaPipe** - Google's ML library that does the heavy lifting
- **OpenCV** - for camera stuff
- **NumPy** - because Python
- **ImageIO** - for handling GIFs

Works pretty smoothly at 30+ FPS on my laptop. Uses about 200MB RAM, so nothing crazy.

## Adding New Gestures

If you want to add more gestures, check out `src/gesture_detector.py`. The flex pose detection is a good example - it basically checks if your arm is bent at the right angle. You could probably add a dab detector or something equally ridiculous.

## Known Issues

- Sometimes loses tracking if you move too fast (working on it)
- The flex detection might need you to adjust the angle threshold
- Works better with good lighting

## Why I Made This

Started as a "let me learn MediaPipe" project. 

## Contributing

Feel free to add more gestures, better memes, or fix my questionable code. PRs welcome!

## License

MIT - do whatever you want with it

---

Made by Snehar | If you use this, drop me a star ⭐
