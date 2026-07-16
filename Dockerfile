# Dockerfile for ConvertLeica-Docker
# Base image with Python 3.13
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# The current OpenCV headless wheel links libxcb on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements-docker.txt /app/
COPY main.py \
     leica_converter.py \
     ci_leica_converters_helpers.py \
     ci_leica_converters_single_lif.py \
     ci_leica_converters_ometiff.py \
     ci_leica_converters_ometiff_rgb.py \
     ci_leica_converters_omezarr.py \
     ReadLeicaLIF.py \
     ReadLeicaLOF.py \
     ReadLeicaXLEF.py \
     ParseLeicaImageXML.py \
     ParseLeicaImageXMLLite.py \
     /app/
COPY cideconvolve_io /app/cideconvolve_io

# Create and activate virtual environment, install dependencies
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements-docker.txt \
    && python -c "import leica_converter; import ci_leica_converters_omezarr"

# Ensure venv is used for all future commands
ENV PATH="/opt/venv/bin:$PATH"

# Expose convert_leica as a CLI
ENTRYPOINT ["python", "main.py"]

# docker build --no-cache -t convertleica-docker .   
# docker build -t convertleica-docker .   

# WSL Example usage:

# sudo mkdir -p /mnt/data
# sudo mount -t drvfs L:/Archief/active/cellular_imaging/OMERO_test/ValidateDocker /mnt/data

# docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB.lif --image_uuid 710afbc4-24d7-11f0-bebf-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
