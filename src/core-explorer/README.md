# Numerai: Explorer

This repository accompanies that YouTube video: [https://youtu.be/x0wKIDTXG9c](https://youtu.be/x0wKIDTXG9c)

Assuming the data is supposed to be kept in current directory the explorer container can be run with the following command:

    docker run \
      -v $PWD:/data \
      -v /var/run/docker.sock:/var/run/docker.sock \
      r606020/numerai-explorer
