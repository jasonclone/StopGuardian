# StopGuardian Dockerfile (Ubuntu 22.04 + SUMO from source + Python 3.10 + PyTorch)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.10 python3.10-distutils python3.10-dev \
    curl git build-essential cmake \
    libxerces-c-dev libfox-1.6-dev libgdal-dev \
    libproj-dev libgl2ps-dev libeigen3-dev \
    libqt5svg5-dev qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install pip for Python 3.10
# ---------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

# ---------------------------------------------------------
# Build SUMO from source (latest stable)
# ---------------------------------------------------------
RUN git clone --recursive https://github.com/eclipse/sumo.git /opt/sumo && \
    cd /opt/sumo && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

ENV SUMO_HOME=/usr/local/share/sumo
ENV PATH="$SUMO_HOME/bin:$PATH"

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------
WORKDIR /app
COPY . /app

RUN mkdir -p /app/results

CMD ["python3", "baselines/b1.py"]
