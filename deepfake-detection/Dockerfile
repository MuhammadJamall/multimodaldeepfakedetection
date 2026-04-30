FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/deepfake-detection

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "training.train", "--config", "configs/default.yaml"]
