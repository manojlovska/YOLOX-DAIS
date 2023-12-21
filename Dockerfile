FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get -y install gcc g++ ffmpeg libsm6 libxext6
COPY ./ /install/
RUN pip install --no-cache-dir /install/. && cd .. && rm -r install
WORKDIR /code