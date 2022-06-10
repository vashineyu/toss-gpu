"""Automatic detect and assign available gpus to env variables
Usage:
read -r N_GPU GPU_ID <<< `python assign_gpus.py --min_request 2 --max_request 4`
echo $N_GPU  # should get some number
echo $GPU_ID # should get some ids

mpirun -np ${N_GPU} ...

"""
import sys

import click

from .utils import get_gpus_on_demand


@click.command()
@click.option('--min-request', default=1, help='Minimum #gpu to request', type=int)
@click.option('--max-request', default=1, help='Maximum #gpu to request', type=int)
@click.option('--memory-lb', default=10000, help='single GPU should at least have #RAM (mb)', type=int)
def cli(min_request, max_request, memory_lb):
    """
    Toss GPUs by request.
    The numbers of GPU will fit min-request but not always fit max-request.

    Usage:

    # Automatically set gpu to environment variables.

    read -r N_GPUS, GPU_IDS <<< `toss-gpu [OPTIONS]`

    # Check variables are set

    echo ${N_GPUS}, ${GPU_IDS}

    # Run you program

    CUDA_DEVICE_ORDER=PCU_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_IDS} python ...
    """
    gpus = get_gpus_on_demand(
        min_request,
        max_request,
        memory_lb,
    )
    if len(gpus) < min_request:
        raise RuntimeError(
            f'Total available GPU: {len(gpus)}, which is less than min-request: {min_request}',
        )
    gpu_ids = ','.join(str(i) for i in gpus)
    sys.stdout.write(str(len(gpus)))
    sys.stdout.write(' ')
    sys.stdout.write(gpu_ids)


if __name__ == '__main__':
    cli()
