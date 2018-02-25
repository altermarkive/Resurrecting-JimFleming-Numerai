FROM ubuntu:xenial-20171114

RUN apt-get -yq update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -yq install curl python3 python3-pip python3-dev python3-tk build-essential git libopenblas-dev libblas-dev libatlas-base-dev

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
RUN pip3 install runipy==0.1.3 && \
    sed -i "s/'svg'\,/'svg',\ 'text\/vnd\.plotly\.v1\+html'\:\ 'html'\,/g" /usr/local/lib/python3.5/dist-packages/runipy/notebook_runner.py
RUN pip install plotly==2.2.3
ADD Makefile.fastFM /Makefile.fastFM
RUN git clone --recursive https://github.com/ibayer/fastFM.git && \
    mv /Makefile.fastFM /fastFM/Makefile && \
    cd /fastFM && \
    pip3 install -r ./requirements.txt && \
    make && \
    pip3 install .
RUN git clone https://github.com/altermarkive/SOMPY.git && \
    cd /SOMPY && \
    python3 setup.py install

ADD *.py /code/
ADD models /code/models
ADD notebooks /code/notebooks
ADD bh_tsne /code/bh_tsne

RUN cd /code/bh_tsne && g++ sptree.cpp tsne.cpp -o bh_tsne -O2 2> /dev/null

CMD ["/usr/bin/python3", "/code/run.py"]
