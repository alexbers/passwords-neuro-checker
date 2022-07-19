FROM ubuntu:22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy python3-pip
RUN pip3 install tensorflow aiohttp

RUN useradd pass_complexity -u 10000 -m

WORKDIR /home/pass_complexity/
CMD ["python3", "worker.py"]
