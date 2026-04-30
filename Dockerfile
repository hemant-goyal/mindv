FROM python:3.11-slim

LABEL maintainer="Manjyot"
LABEL description="minDeepVariant: minimal deep learning variant caller"

# System dependencies for pysam
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy configs
COPY configs/ configs/

# Default: show help
ENTRYPOINT ["mindeepvariant"]
CMD ["--help"]

# Example usage:
# docker build -t mindeepvariant .
# docker run mindeepvariant train --output /data/weights.pth
# docker run -v /path/to/data:/data mindeepvariant scan \
#     --bam_dir /data/bams --ref /data/ref.fna \
#     --panel /app/configs/leprae.json --outdir /data/results
