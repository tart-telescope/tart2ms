FROM kernsuite/base:5
ARG DEBIAN_FRONTEND=noninteractive

RUN docker-apt-install locales \
    meqtrees \
    casalite \
    python3-numpy \
    python3-astropy \
    python3-requests \
    python3-tz \
    python3-casacore \
    python3-pip

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

RUN pip3 install tart2ms

ENV LANG en_US.UTF-8 
