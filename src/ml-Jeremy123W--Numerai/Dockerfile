FROM python:3.5-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -yq update && \
    apt-get -yq install build-essential libopenblas-dev

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements*.txt

ADD run.py /code/run.py

ADD scripts/* /code/

CMD ["/usr/local/bin/python3", "/code/run.py"]
