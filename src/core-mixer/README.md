# Numerai: Mixer

This repository accompanies that YouTube video: [https://youtu.be/NBB2AxI3gF0](https://youtu.be/NBB2AxI3gF0)

Assuming the data is supposed to be kept in current directory the mixer container can be run with the following command:

    docker run \
      -v $PWD:/data \
      -v /var/run/docker.sock:/var/run/docker.sock \
      r606020/numerai-mixer
