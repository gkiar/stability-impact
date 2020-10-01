FROM gkiar/py37-alpine-base:latest

RUN apk --no-cache --update-cache --allow-untrusted \
        --repository http://dl-3.alpinelinux.org/alpine/edge/testing \
        add hdf5 hdf5-dev

RUN pip3 install nibabel dipy matplotlib onevox boutiques

RUN mkdir -p /opt
COPY ./dipy_tracking.py /opt/dipy_tracking.py

ENTRYPOINT ["python3", "/opt/dipy_tracking.py"]
