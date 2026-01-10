FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp
COPY installer.sh .
COPY requirements.txt .

RUN apt-get update && apt-get install sudo bash -y

RUN bash installer.sh
RUN pip3 install -r requirements.txt --break-system-packages