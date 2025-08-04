FROM ubuntu:24.04

# Set working directory
WORKDIR /app

# Install Python, pip, dan tools untuk build & venv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Buat virtual environment
RUN python3 -m venv /opt/venv

# Aktifkan virtual environment dan tambahkan ke PATH
ENV PATH="/opt/venv/bin:/app:$PATH"

# Salin file requirements dan install ke virtual environment
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek
COPY . .

# Buat direktori persistent
RUN mkdir -p \
    persistent/eval_history \
    persistent/metadata \
    persistent/preview_history \
    persistent/roi_result \
    persistent/uploaded

# Buka port (opsional)
EXPOSE 8080

# Set working directory dan jalankan aplikasi
WORKDIR /app
CMD ["python", "main.py"]
