FROM python:3-alpine3.6

RUN apk add --update build-base openblas-dev linux-headers swig

ADD requirements*.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements1.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements2.txt && \
    rm /tmp/requirements*.txt

ADD run.py /code/run.py

CMD ["/usr/local/bin/python3", "/code/run.py"]
