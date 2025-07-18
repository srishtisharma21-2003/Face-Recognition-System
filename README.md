# Face Recognition System | Python + OpenCV + KNN

## Features
- Face capture using webcam
- Train on local image dataset
- Recognize faces in real-time using GUI or CLI
- Toggle between light/dark UI

## Usage

```bash
pip install -r requirements.txt

python collect_data.py     # Capture face images
python training.py         # Train & save encodings
python gui_app.py          # Launch GUI
python recognizer.py       # Use CLI-based recognition
