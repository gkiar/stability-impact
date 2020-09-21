FROM nvidia/cuda:8.0-runtime-ubuntu16.04 

ENV FSLDIR=/usr/local/fsl
RUN apt-get update -qqq && \
    apt-get upgrade -y && \
    apt-get install -y wget curl git file gcc libmng-dev bzip2 sudo perl python && \
    apt-get install -y python-pip python3-dev python3-pip &&\
    pip install nibabel &&\
    cd /tmp && \
    wget http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb && \
    dpkg -i libpng12-0_1.2.54-1ubuntu1_amd64.deb && \
    rm -rf libpng12-0_1.2.54-1ubuntu1_amd64.deb

RUN apt-get install -y python curl gcc

RUN curl -O https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    chmod 755 fslinstaller.py && \
    su -c "python ./fslinstaller.py -D -E -d /usr/share/fsl --fslversion 5.0.11"

RUN rm -rf fslinstaller.py

ENV FSLDIR=/usr/share/fsl
RUN curl -O https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/centos6/eddy_cuda8.0 && \
    mv eddy_cuda8.0 ${FSLDIR}/bin && \
    chmod 755 ${FSLDIR}/bin/eddy_cuda8.0  

RUN curl -O https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/centos6/eddy_openmp && \
    mv eddy_openmp ${FSLDIR}/bin && \
    chmod 755 ${FSLDIR}/bin/eddy_openmp 

RUN pip3 install duecredit \
                 pybids==0.7.0 \
                 nibabel \
                 numpy \
                 scipy \
                 boutiques \
                 indexed_gzip

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD ./ /opt/

ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH=/usr/lib/fsl:$FSLDIR/bin:$PATH
ENV FSLMULTIFILEQUIT=TRUE
ENV POSSUMDIR=/usr/share/fsl
ENV LD_LIBRARY_PATH=/usr/lib/fsl:$LD_LIBRARY_PATH
ENV FSLTCLSH=/usr/bin/tclsh
ENV FSLWISH=/usr/bin/wish
ENV FSLOUTPUTTYPE=NIFTI_GZ
