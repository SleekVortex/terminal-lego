FROM debian:bookworm-slim

ARG OPENCODE_VERSION=1.17.13
ARG OPENCODE_TARGET=linux-x64
ENV PATH="/opt/terminal-lego/opencode/bin:/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    file \
    findutils \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    url="https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-${OPENCODE_TARGET}.tar.gz"; \
    curl -fL \
      --retry 8 \
      --retry-delay 5 \
      --retry-all-errors \
      --connect-timeout 20 \
      -o /tmp/opencode.tar.gz \
      "${url}"; \
    mkdir -p /opt/terminal-lego/opencode/bin; \
    tar -xzf /tmp/opencode.tar.gz -C /opt/terminal-lego/opencode/bin opencode; \
    rm -f /tmp/opencode.tar.gz; \
    chmod +x /opt/terminal-lego/opencode/bin/opencode; \
    /opt/terminal-lego/opencode/bin/opencode --version | tee /opt/terminal-lego/opencode/VERSION; \
    grep -F "${OPENCODE_VERSION}" /opt/terminal-lego/opencode/VERSION

CMD ["/opt/terminal-lego/opencode/bin/opencode", "--version"]
