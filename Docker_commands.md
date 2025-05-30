## build docker image:
After cloning the repository, run:

```
docker build -t mmsegmentation docker/
```

Afterwards, you can run this command to launch the container:

```
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmsegmentation/data mmsegmentation
```

in my case:
```
docker run --rm --gpus all --shm-size=8g -it -v .:/mmsegmentation mmsegmentation
```
