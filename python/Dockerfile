FROM python:3.7


RUN apt update 

# for OpenCV
RUN apt-get install libgl1-mesa-dev
COPY requirement.txt ./requirement.txt
RUN pip3 install -r requirement.txt
