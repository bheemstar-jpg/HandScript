# HandScript - Live ASL Alphabet Predictor

HandScript is a real-time American Sign Language (ASL) alphabet recognition system that uses your webcam to predict ASL hand gestures. It utilizes MediaPipe for hand tracking and a trained machine learning model to classify hand signs into letters A-Z.

## Features
- 🖐️ Real-time ASL alphabet detection using a webcam
- 🧠 Machine learning-based predictions
- 🎥 Uses MediaPipe for efficient hand tracking
- 📏 Normalizes hand landmark positions for better accuracy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/waditariq/HandScript.git
   cd HandScript
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the following files are in the same directory:
   - `asl_prediction.py` (your ASL prediction script)
   - `model.p` (the trained machine learning model)

2. Run the ASL predictor script:
   ```bash
   python asl_prediction.py
   ```

3. Allow access to your webcam. The program will start detecting ASL hand gestures and display predictions on the screen.

## Controls
- Press `ESC` to toggle the menu.
- Press `Q` (when the menu is shown) to exit.

## Notes
- The model is pre-trained to recognize ASL alphabet gestures (A-Z).
- Ensure proper lighting and a clear hand view for better accuracy.

## Contributing
Feel free to fork the repository, improve the model, or add new features! Open a pull request with any enhancements.

## License
This project is open-source under the MIT License.

---
Roan Malec

