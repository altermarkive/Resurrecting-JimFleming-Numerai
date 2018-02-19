FROM python:3.6-alpine3.7

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ADD *.py /code/

ENV PYTHONPATH=/code/

CMD ["luigi", "--module", "explorer", "Exploring", "--local-scheduler", "--workers", "10"]
