# Deepfake Software (YOLOv8 + Django)

Deepfake Software is a Django 5 application that wraps a fine-tuned YOLOv8 model to flag synthetic faces in still images, pre-recorded videos, or a live webcam feed. The dashboard lets authenticated users upload media, runs the detector on the server, and streams back annotated results with class names (`real`/`fake`) and confidence scores.

## Features
- Email/password registration with custom password validation and session-based dashboards.
- Image pipeline: uploads land in `media/uploaded_images/`, YOLOv8 adds bounding boxes, and the annotated result is stored in `media/detected_images/`.
- Video pipeline: each frame is processed, labeled, and re-assembled at the original FPS into `media/detected_videos/…_detected.mp4`.
- Live camera mode for real-time inference via OpenCV.
- Training artifacts (`Dataset/train2`) for the fine-tuned detector, including weights, curves, and confusion matrices to audit model quality.

## Project Layout
```
deepfake_software/
├── app/                   # Django app with views, auth helpers, YOLO inference
├── deepfake_software/     # Project settings, urls, wsgi/asgi
├── templates/             # UI (index, auth, dashboard)
├── static/                # Bootstrap theme, JS, demo assets
├── media/                 # Runtime uploads and detection outputs (gitignored)
└── Dataset/train2/        # YOLO training outputs (args.yaml, weights, metrics)
```

## Prerequisites
- Python 3.10+ (tested with 3.11)
- pip 23+
- Git
- Optional: CUDA-capable GPU + PyTorch with CUDA for faster inference/training

## Quickstart
```bash
git clone https://github.com/<your-org>/deepfake_software.git
cd deepfake_software
python -m venv .venv
.venv\Scripts\activate   # on Windows; use `source .venv/bin/activate` on macOS/Linux
pip install --upgrade pip wheel
pip install django==5.0.1 ultralytics opencv-python imageio numpy pillow python-dotenv
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
Visit http://127.0.0.1:8000, register/login, then open the dashboard to upload media.

## Configuring the Detector
- The YOLO weights path defaults to `Dataset/train2/weights/best.pt` (see `app/views.py`). Replace the file with your own checkpoint or update the path.
- Media folders (`media/uploaded_images`, `media/detected_images`, `media/uploaded_videos`, `media/detected_videos`) must exist and be writable by the Django process.
- For production, move secrets into environment variables (e.g. via `.env`) and set:
  ```
  SECRET_KEY=<your secret>
  DEBUG=False
  ALLOWED_HOSTS=your.domain.com
  ```
  Then load them in `deepfake_software/settings.py` before deploying.

## Running YOLO Training
The repository includes the Ultralytics training outputs under `Dataset/train2/`. To retrain:
```bash
yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml imgsz=640 epochs=50 project=Dataset/train2
```
Update the `model = YOLO('Dataset/train2/weights/best.pt')` line once training finishes.

## Testing
Run Django’s test suite (currently placeholders in `app/tests.py`):
```bash
python manage.py test
```
Add tests for any new views, forms, or APIs before submitting changes.

## Roadmap Ideas
- Add REST APIs for programmatic submissions.
- Ship a `requirements.txt` / `pyproject.toml` and Dockerfile for reproducible deploys.
- Persist detection history per user and display trends on the dashboard.
- Add GPU/CPU inference switches and background job queues for long videos.

## License
Choose a license (MIT, Apache-2.0, etc.) before publishing the project publicly, then update this section accordingly.

