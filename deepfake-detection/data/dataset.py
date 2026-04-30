from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

from data.preprocessing import build_face_extractor, compute_mel_spectrogram, crop_faces


@dataclass
class Record:
    video_path: Path
    audio_path: Path
    label: int


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = "train",
        image_size: int = 224,
        clip_len: int = 16,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        win_length: int = 400,
        max_audio_seconds: int = 4,
        use_face_crop: bool = True,
        device: Optional[str] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = self._load_records(self.manifest_path, split)
        self.image_size = image_size
        self.clip_len = clip_len
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_audio_seconds = max_audio_seconds
        self.use_face_crop = use_face_crop
        self.device = device

        self.image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.face_extractor = (
            build_face_extractor(image_size=image_size, device=device) if use_face_crop else None
        )

    def _load_records(self, manifest_path: Path, split: Optional[str]) -> List[Record]:
        records: List[Record] = []
        with manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if split and "split" in row and row["split"] != split:
                    continue
                video_path = Path(row["video_path"])
                if not video_path.is_absolute():
                    video_path = self.root / video_path

                audio_path_value = row.get("audio_path") or row["video_path"]
                audio_path = Path(audio_path_value)
                if not audio_path.is_absolute():
                    audio_path = self.root / audio_path

                label = int(row["label"])
                records.append(Record(video_path=video_path, audio_path=audio_path, label=label))
        return records

    def _read_video_frames(self, video_path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.clip_len
        indices = np.linspace(0, max(frame_count - 1, 0), num=self.clip_len, dtype=int)

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(f"No frames read from {video_path}")

        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        return frames

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        max_len = int(self.sample_rate * self.max_audio_seconds)
        if waveform.size(1) > max_len:
            waveform = waveform[:, :max_len]

        return waveform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        frames = self._read_video_frames(record.video_path)
        if self.use_face_crop:
            video_tensor = crop_faces(
                frames,
                image_size=self.image_size,
                device=self.device,
                extractor=self.face_extractor,
            )
        else:
            video_tensor = torch.stack([self.image_transform(frame) for frame in frames], dim=0)

        waveform = self._load_audio(record.audio_path)
        mel = compute_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        label = torch.tensor(record.label, dtype=torch.float32)
        return {"video": video_tensor, "audio": mel, "label": label}
