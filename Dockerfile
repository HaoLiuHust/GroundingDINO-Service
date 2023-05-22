FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

#for Chinese User, uncomment this line
# COPY sources.list /etc/apt/sources.list

RUN apt update && \
     apt install openjdk-17-jdk -y

RUN apt install git -y

#install python packages
COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt --no-cache -i https://repo.huaweicloud.com/repository/pypi/simple/