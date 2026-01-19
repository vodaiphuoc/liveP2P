FROM ghcr.io/ggml-org/llama.cpp:server

WORKDIR /app
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]
