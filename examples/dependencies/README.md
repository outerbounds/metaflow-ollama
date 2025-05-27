## Build CPU image for Metaflow tasks
```bash
docker build -t ollama-metaflow-task:cpu -f Dockerfile.cpu  .
```

## Build GPU image for Metaflow tasks
```bash
docker build -t ollama-metaflow-task:gpu -f Dockerfile.gpu  .
```