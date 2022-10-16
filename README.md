# Toss-GPU

This package is used to toss gpu for a local machine when you don't want to check gpu everytime by `nvidia-smi`

### Installation
`pip toss-gpu`

### Usage
Arguments:
* `--min-reqeust`: at least have #gpus (default=1)
* `--max-request`: at most to have #gpus (default=1)
* `--memory-lb`: only consider GPUs that fit memory-lower-bound (unit: MB, default=10,000 MB)

#### As command line
```bash
# Automatically set gpu to environment variables.
read -r N_GPUS, GPU_IDS <<< `toss-gpu --min-request=1`

# Check variables are set
echo ${N_GPUS}, ${GPU_IDS}
```

* Additional scenarios
```bash
# At least give me one GPU, but I'd like to have 4 if it is possible. Each of them should at least have 24000 free memory.
read -r N_GPUS, GPU_IDS <<< `toss-gpu --min-request=1 --max-request=4 --memory-lb 24000`

mpirun -np ${N_GPUS} \
  --allow-run-as-root \
  -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -x CUDA_VISIBLE_DEVICES=${GPU_IDS} \
  --bind-to none \
  python ...
```
**Note** It is not determistic, you can have up to 4 gpus if possible (1 to 4).

#### As a python function
```python
import os
from toss_gpu.utils import get_gpus_on_demand

min_request, max_request = 1, 2
available_gpus = get_gpus_on_demand(
    min_request=min_request,
    max_request=max_request,
    memory_lb=8192,
)

if len(available_gpus) < min_request:
    raise RuntimeError('Cannot request the minumum request GPUs')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_gpus)
```

## TODOs / Roadmaps
* Python API
* More friendly usage for CLI
* Scheduling -- Handing / Kill process
