FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-aiohttp python3-pip
RUN pip3 install tflite-runtime

RUN useradd pass_complexity -u 10000 -m

WORKDIR /home/pass_complexity/demo
CMD ["python3", "worker.py"]
