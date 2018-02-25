FROM ubuntu:xenial-20171114

RUN apt-get -yq update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -yq install build-essential default-jdk curl unzip libatlas3-base libopenblas-base

RUN useradd -s /bin/bash -m -d /home/user user
USER user
WORKDIR /home/user

RUN curl -L -O http://prdownloads.sourceforge.net/weka/weka-3-8-1.zip && \
    unzip weka-3-8-1.zip && \
    rm weka-3-8-1.zip
ENV CLASSPATH="${CLASSPATH}:/home/user/weka-3-8-1/weka.jar"

RUN java weka.core.WekaPackageManager -install-package netlibNativeLinux
ENV CLASSPATH="${CLASSPATH}:/home/user/wekafiles/packages/netlibNativeLinux/lib/jniloader-1.1.jar:/home/user/wekafiles/packages/netlibNativeLinux/lib/native_system-java-1.1.jar:/home/user/wekafiles/packages/netlibNativeLinux/lib/netlib-native_system-linux-x86_64-1.1-natives.jar:/home/user/wekafiles/packages/netlibNativeLinux/lib/native_ref-java-1.1.jar:/home/user/wekafiles/packages/netlibNativeLinux/lib/netlib-native_ref-linux-x86_64-1.1-natives.jar"

RUN java weka.core.WekaPackageManager -install-package Auto-WEKA
ENV CLASSPATH="${CLASSPATH}:/home/user/wekafiles/packages/Auto-WEKA/autoweka.jar"

ADD run.sh /code/run.sh

USER root

CMD ["/bin/bash", "/code/run.sh"]
