FROM --platform=linux/amd64  pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS surgvu25Cat1


#FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN apt update&&apt install -y openssh-server && apt install nano&&apt install -y python3 python3-pip&&apt-get update && apt-get install -y libgl1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user model /opt/app/model

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --chown=user:user inference.py /opt/app/


ENTRYPOINT ["python", "inference.py"]
