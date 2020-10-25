FROM pytorch/pytorch:latest

RUN apt -y update && apt -y install \
python3-tk

RUN python -m pip install -U pip
RUN python -m pip install -U matplotlib

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
