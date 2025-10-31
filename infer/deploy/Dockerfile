FROM litellm/litellm:latest
LABEL org.opencontainers.image.source=https://github.com/londonaicentre/GenoLlama
COPY config.yml /app/config.yml
ENTRYPOINT ["litellm", "--config", "/app/config.yml"]