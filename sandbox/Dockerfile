FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN pip install --no-cache-dir pytest==8.3.5
RUN useradd --uid 10001 --create-home sandbox
USER 10001:10001
WORKDIR /workspace
CMD ["sleep", "infinity"]
