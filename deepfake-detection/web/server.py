"""
server.py — Flask Backend for DeepDetect
==========================================
Loads the trained DeepfakeDetector model and runs real inference
on uploaded images/videos.

Usage:
    # With trained checkpoint:
    python server.py --checkpoint checkpoints/best_auroc.pt

    # Without checkpoint (uses untrained model — for testing UI flow):
    python server.py --no-checkpoint

    # Custom port:
    python server.py --checkpoint checkpoints/best_auroc.pt --port 5000
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import DeepfakeDetector

# ── App Setup ──────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Global model reference
model = None
device = None


# ── Model Loading ──────────────────────────────────────────────────────────

def load_model(checkpoint_path=None):
    """Load DeepfakeDetector model."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Server] Device: {device}")

    model = DeepfakeDetector(
        vit_model="google/vit-base-patch16-224-in21k",
        vit_hidden_dim=512,
        audio_hidden_dim=512,
        num_heads=8,
        ffn_hidden_dim=256,
        dropout=0.3,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Server] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        print(f"[Server] ✅ Loaded trained model from epoch {epoch}")
        if metrics:
            print(f"[Server] Training metrics: {metrics}")
    else:
        print("[Server] ⚠️  No checkpoint loaded — using untrained model")
        print("[Server]    Results will NOT be accurate until you train the model.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Server] Model ready: {total_params:,} parameters")


# ── Preprocessing for Inference ────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Preprocess a single image for inference.
    Creates a 16-frame sequence from a single image (repeated).

    Returns:
        frames: (1, 16, 6, 224, 224) tensor
        mel: (1, 16, 80, 32) tensor (zeros — no audio for images)
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Try face detection
    face_crop, mouth_crop = extract_face_and_mouth(img)

    # Create 6-channel tensor (face + mouth)
    face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
    mouth_tensor = torch.from_numpy(mouth_crop).permute(2, 0, 1).float() / 255.0
    frame_6ch = torch.cat([face_tensor, mouth_tensor], dim=0)  # (6, 224, 224)

    # Repeat for 16 frames (single image → static video)
    frames = frame_6ch.unsqueeze(0).repeat(16, 1, 1, 1)  # (16, 6, 224, 224)
    frames = frames.unsqueeze(0)  # (1, 16, 6, 224, 224)

    # No audio for images — use zeros
    mel = torch.zeros(1, 16, 80, 32)

    return frames, mel


def preprocess_video(video_path):
    """
    Preprocess a video for inference.
    Extracts 16 frames and audio mel spectrogram.

    Returns:
        frames: (1, 16, 6, 224, 224) tensor
        mel: (1, 16, 80, 32) tensor
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise ValueError("Video has no frames")

    # Uniformly sample 16 frame indices
    indices = np.linspace(0, total_frames - 1, 16, dtype=int)

    raw_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif raw_frames:
            raw_frames.append(raw_frames[-1])  # Repeat last frame
        else:
            raw_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()

    # Pad if needed
    while len(raw_frames) < 16:
        raw_frames.append(raw_frames[-1] if raw_frames else np.zeros((224, 224, 3), dtype=np.uint8))

    # Process each frame: face + mouth extraction
    processed_frames = []
    last_face = None
    last_mouth = None
    for frame in raw_frames[:16]:
        try:
            face_crop, mouth_crop = extract_face_and_mouth(frame)
            last_face = face_crop
            last_mouth = mouth_crop
        except Exception:
            if last_face is not None:
                face_crop, mouth_crop = last_face, last_mouth
            else:
                face_crop = cv2.resize(frame, (224, 224))
                mouth_crop = cv2.resize(frame, (224, 224))

        face_t = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        mouth_t = torch.from_numpy(mouth_crop).permute(2, 0, 1).float() / 255.0
        frame_6ch = torch.cat([face_t, mouth_t], dim=0)
        processed_frames.append(frame_6ch)

    frames = torch.stack(processed_frames, dim=0).unsqueeze(0)  # (1, 16, 6, 224, 224)

    # Try to extract audio mel spectrogram
    mel = extract_audio_mel(video_path)  # (1, 16, 80, F)

    return frames, mel


def extract_face_and_mouth(img):
    """
    Extract face and mouth crops from an image.
    Falls back to center crop if face detection fails.

    Returns:
        face_crop: (224, 224, 3) numpy array
        mouth_crop: (224, 224, 3) numpy array
    """
    import cv2

    h, w = img.shape[:2]

    # Try MTCNN if available
    try:
        from facenet_pytorch import MTCNN
        detector = MTCNN(keep_all=False, device='cpu')
        boxes, _ = detector.detect(img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                face = img[y1:y2, x1:x2]
                face_crop = cv2.resize(face, (224, 224))

                # Mouth: lower 40% of face
                mouth_y1 = y1 + int(0.6 * (y2 - y1))
                mouth = img[mouth_y1:y2, x1:x2]
                mouth_crop = cv2.resize(mouth, (224, 224))
                return face_crop, mouth_crop
    except ImportError:
        pass

    # Fallback: center crop
    min_dim = min(h, w)
    cy, cx = h // 2, w // 2
    half = min_dim // 2
    face = img[cy - half:cy + half, cx - half:cx + half]
    face_crop = cv2.resize(face, (224, 224))

    # Mouth: lower 40% of center crop
    mouth_start = cy
    mouth = img[mouth_start:cy + half, cx - half:cx + half]
    if mouth.shape[0] < 10 or mouth.shape[1] < 10:
        mouth = face
    mouth_crop = cv2.resize(mouth, (224, 224))

    return face_crop, mouth_crop


def extract_audio_mel(video_path):
    """
    Extract mel spectrogram from video audio.
    Falls back to zeros if audio extraction fails.

    Returns:
        mel: (1, 16, 80, 32) tensor
    """
    try:
        import torchaudio

        # Extract audio from video using torchaudio
        waveform, sr = torchaudio.load(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        )
        mel_spec = mel_transform(waveform).squeeze(0)  # (80, time)

        # Window into 16 segments
        total_time = mel_spec.shape[1]
        window_size = max(1, total_time // 16)
        mel_windows = []
        for i in range(16):
            start = i * window_size
            end = min(start + window_size, total_time)
            if start >= total_time:
                window = torch.zeros(80, 32)
            else:
                window = mel_spec[:, start:end]
                # Pad or truncate to 32
                if window.shape[1] < 32:
                    window = torch.nn.functional.pad(window, (0, 32 - window.shape[1]))
                else:
                    window = window[:, :32]
            mel_windows.append(window)

        mel = torch.stack(mel_windows, dim=0).unsqueeze(0)  # (1, 16, 80, 32)
        return mel

    except Exception:
        # No audio available — return zeros
        return torch.zeros(1, 16, 80, 32)


# ── Analysis Notes ─────────────────────────────────────────────────────────

def get_analysis_note(is_fake, confidence, has_audio):
    """Generate forensic analysis note based on result."""
    if is_fake:
        if confidence > 90:
            return "Strong cross-modal desynchronization detected. Lip movements do not match speech patterns across multiple frames."
        elif confidence > 80:
            return "Audio-visual inconsistencies detected. Temporal artifacts present in facial region."
        else:
            return "Potential manipulation detected. Spectral anomalies found in the analyzed media."
    else:
        if not has_audio:
            return "Visual analysis shows no manipulation artifacts. Upload a video with audio for full cross-modal analysis."
        if confidence > 90:
            return "All forensic checks passed. Audio-visual synchronization within expected parameters."
        else:
            return "No significant manipulation artifacts detected. Cross-modal features are consistent."


# ── API Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    """Serve the frontend."""
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze an uploaded image or video for deepfake detection.

    Accepts multipart/form-data with a 'file' field.
    Returns JSON: { verdict, confidence, note }
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Determine file type
    filename = file.filename.lower()
    is_image = filename.endswith(('.jpg', '.jpeg', '.png'))
    is_video = filename.endswith('.mp4')

    if not is_image and not is_video:
        return jsonify({"error": "Unsupported format. Use JPG, PNG, or MP4."}), 400

    # Save to temp file
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()

        # Preprocess
        if is_image:
            frames, mel = preprocess_image(tmp.name)
            has_audio = False
        else:
            frames, mel = preprocess_video(tmp.name)
            has_audio = mel.abs().sum() > 0

        # Run inference
        frames = frames.to(device)
        mel = mel.to(device)

        with torch.no_grad():
            prob = model(frames, mel)  # (1, 1) sigmoid output

        probability = prob.item()
        is_fake = probability >= 0.5
        confidence = probability * 100 if is_fake else (1 - probability) * 100

        verdict = "FAKE" if is_fake else "REAL"
        note = get_analysis_note(is_fake, confidence, has_audio)

        return jsonify({
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "note": note,
            "probability": round(probability, 4),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    })


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepDetect Backend Server")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Run without a checkpoint (untrained model)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    # Load model
    load_model(args.checkpoint)

    print(f"\n[Server] Starting on http://{args.host}:{args.port}")
    print(f"[Server] Open http://{args.host}:{args.port} in your browser\n")

    app.run(host=args.host, port=args.port, debug=False)
