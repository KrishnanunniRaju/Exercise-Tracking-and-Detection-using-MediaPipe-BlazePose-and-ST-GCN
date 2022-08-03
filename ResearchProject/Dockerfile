FROM ubuntu:20.04

RUN apt-get update -qq   && apt-get install -y -qq python3.10
RUN apt-get install -y python3-pip python-dev python3.10-dev && python3.10 -m pip install pip --upgrade

RUN pip3 install --no-cache --upgrade pip setuptools

RUN mkdir /ResearchProject

WORKDIR /ResearchProject

COPY . /ResearchProject

RUN pip3 install -r /ResearchProject/requirements.txt

EXPOSE 3000

CMD [ "python3", "/ResearchProject/main.py" ]
