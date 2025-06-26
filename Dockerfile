# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    JAVA_HOME="/usr/bin/java"

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt/ \
    apt-get update && apt-get install -y --no-install-recommends \
        python3-opengl \
        openjdk-21-jdk \
        xvfb \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN --mount=type=cache,id=poetry-build,target=/root/.cache/poetry \
    pip install --no-cache-dir poetry==$POETRY_VERSION && \
    python -m pip install --upgrade pip

WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY lib/ lib/
COPY src/ src/

RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    poetry lock --no-update && \
    poetry install --no-interaction --no-ansi --only main

WORKDIR /app/lib/gym_microrts/
RUN sed -i 's/\r$//' build.sh && chmod +x build.sh && bash build.sh && \
    rm -rf build.sh && \
    rm -rf gym_microrts/microrts/data && \
    rm -rf gym_microrts/microrts/resources && \
    rm -rf gym_microrts/microrts/src && \
    rm -rf gym_microrts/microrts/tests && \
    rm -rf gym_microrts/microrts/utts && \
    rm -rf gym_microrts/microrts/build.xml

CMD ["python", "src/main.py", "--note", "docker_run"]