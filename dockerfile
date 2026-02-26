# ---------------------------------------------------------
# StopGuardian Dockerfile (Fast, Stable, SUMO + PyTorch)
# ---------------------------------------------------------

# Use official SUMO image (already includes SUMO + tools)
FROM dlrts/sumo:latest

# Install Python + pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set SUMO environment (already set, but we ensure it)
ENV SUMO_HOME=/usr/share/sumo
ENV PATH="$SUMO_HOME/bin:$PATH"

# ---------------------------------------------------------
# Install Python dependencies FIRST (cached)
# ---------------------------------------------------------

# Create requirements file inside the image
# This allows Docker to cache pip installs unless requirements change
COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# ---------------------------------------------------------
# Set working directory
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Copy project files LAST (fast rebuilds)
# ---------------------------------------------------------
COPY . /app

# Ensure results folder exists
RUN mkdir -p /app/results

# ---------------------------------------------------------
# Default command
# ---------------------------------------------------------
CMD ["python3", "baselines/fixed_time_metrics.py"]
