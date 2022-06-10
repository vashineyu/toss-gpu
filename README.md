# Toss-GPU

This package is used to toss gpu for a local machine when you don't want to check gpu everytime by `nvidia-smi`

### Installation
`pip toss-gpu`

### Usage
Parameters
* `--min-reqeust`: at least have #gpus
* `--max-request`: at most to have #gpus
* `--memory-lb`: only consider GPUs that fit memory-lower-bound (unit: MB)

```bash
# Automatically set gpu to environment variables.
read -r N_GPUS, GPU_IDS <<< `toss-gpu --min-request=1`

# Check variables are set
echo ${N_GPUS}, ${GPU_IDS}

```

* Additional scenarios
```bash
# At least give me one GPU, but I'd like to have 4 if it is possible.
read -r N_GPUS, GPU_IDS <<< `toss-gpu --min-request=1 --max-request=4`

mpirun -np ${N_GPUS} \
  --allow-run-as-root \
  -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -x CUDA_VISIBLE_DEVICES=${GPU_IDS} \
  --bind-to none \
  python ...
```
**Note** It is not determistic, you can up to 4 if available (1 to 4)
