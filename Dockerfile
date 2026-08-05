FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["fed-pi-train"]

