# Build Instructions for Dockerfile.blackwell

## Basic build command:

```bash
docker build -f Dockerfile.blackwell -t openfold-3-blackwell:latest .
```

This will create a Docker image named `openfold-3-blackwell` with the `latest` tag.


## test Pytorch and CUDA

```bash
docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    openfold-3-blackwell:latest \
    python -c "import torch; print('CUDA:', torch.version.cuda); print('PyTorch:', torch.__version__)"
```

Should print something like:

```
CUDA: 13.1
PyTorch: 2.10.0a0+b4e4ee81d3.nv25.12
```

## test run_openfold inference example

```bash
docker run \
    --gpus all -it \
    --ipc=host \
    --ulimit memlock=-1 \
    -v /home/jandom/.openfold3:/root/.openfold3 \
    -v $(pwd)/output:/output \
    -w /output openfold-3-blackwell:latest \
    run_openfold predict \
    --query_json=/opt/openfold3/examples/example_inference_inputs/query_ubiquitin.json \
    --num_diffusion_samples=1 \
    --num_model_seeds=1 \
    --use_templates=false 
```