# Use Python 3.10 Slim
FROM python:3.10-slim

# 1. Install System Dependencies
# FIX: Replaced 'libgl1-mesa-glx' (obsolete) with 'libgl1' (new standard)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Work Directory
WORKDIR /app

# 3. Install Python Dependencies
COPY requirements.txt .

# Install PyTorch CPU version (Keeps image size small)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 4. --- AUTOMATIC FIX FOR BASICSR CRASH ---
# This command finds the broken file inside the container and edits the bad line automatically.
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py

# 5. Copy Project Files
COPY . .

# 6. Expose Ports (8001=API, 8091=Dashboard)
EXPOSE 8001
EXPOSE 8091

# 7. Run the App
CMD ["python", "run.py"]