FROM python:3-alpine3.6

RUN apk add --update build-base openblas-dev

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements*.txt

ADD scripts/* /code/

CMD ["/usr/local/bin/python3", "/code/run.py"]
