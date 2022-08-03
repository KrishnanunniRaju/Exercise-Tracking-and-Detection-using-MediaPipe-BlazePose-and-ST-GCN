FROM python:3.10-slim-buster

RUN python -m pip install --upgrade pip

RUN pip3 install requests paho-mqtt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /ResearchProject

WORKDIR /ResearchProject

COPY . /ResearchProject

RUN pip3 install -r /ResearchProject/requirements.txt

EXPOSE 3000

CMD [ "python3", "/ResearchProject/main.py" ]
