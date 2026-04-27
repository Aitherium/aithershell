# AitherShell Docker Image
# Build: docker build -t aithershell:1.0.0 .
# Run:   docker run -it aithershell:1.0.0

FROM python:3.10-slim

LABEL maintainer="Aitherium <alex@aitherium.com>"
LABEL description="AitherShell - The kernel shell for AitherOS"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY setup.py setup.cfg pyproject.toml MANIFEST.in README.md LICENSE ./
COPY aithershell ./aithershell
COPY examples ./examples
COPY tests ./tests

# Install AitherShell
RUN pip install --no-cache-dir -e .

# Create config directory
RUN mkdir -p /root/.aither

# Set default command
ENTRYPOINT ["aither"]
CMD ["--help"]

# Build args for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION
