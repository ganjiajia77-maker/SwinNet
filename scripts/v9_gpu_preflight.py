"""Resolve an nvidia-smi GPU index to UUID and inspect free VRAM, without torch."""
import argparse
import csv
import io
import subprocess
import sys


def query_gpu(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = list(csv.reader(io.StringIO(output.strip())))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError('Expected exactly one GPU in nvidia-smi output')
    index, uuid, name, total, used, free = (value.strip() for value in rows[0])
    if not uuid.startswith('GPU-'):
        raise ValueError('nvidia-smi did not provide a GPU UUID')
    return {'index': index, 'uuid': uuid, 'name': name,
            'total_mib': int(total), 'used_mib': int(used), 'free_mib': int(free)}


def query_processes(uuid):
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory,process_name',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(output))
            if len(row) == 4 and row[1].strip() == uuid]


def preflight(gpu, minimum_free_mib, report=sys.stderr):
    info = query_gpu(gpu)
    print(f"[GPU] physical nvidia-smi index={info['index']} uuid={info['uuid']} "
          f"name={info['name']} free={info['free_mib']}MiB used={info['used_mib']}MiB "
          f"total={info['total_mib']}MiB; this device will be logical cuda:0", file=report, flush=True)
    try:
        processes = query_processes(info['uuid'])
    except (OSError, subprocess.SubprocessError):
        processes = None
        print('[GPU] compute-process details unavailable', file=report, flush=True)
    for pid, _, memory, name in processes or []:
        print(f'[GPU PROCESS] pid={pid} used={memory}MiB command={name}', file=report, flush=True)
    if info['free_mib'] < minimum_free_mib:
        raise RuntimeError(f"GPU {info['index']} has only {info['free_mib']}MiB free; "
            f"the preflight reserve threshold is {minimum_free_mib}MiB. "
            'Wait for the occupying task to finish or explicitly select another available GPU. '
            'No process was stopped and batch size was not changed.')
    return info['uuid']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--min_free_mib', type=int, default=6144)
    args = parser.parse_args()
    if args.min_free_mib < 1:
        parser.error('min_free_mib must be positive')
    try:
        uuid = preflight(args.gpu, args.min_free_mib)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        print(f'[GPU PREFLIGHT STOP] {error}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    # stdout is deliberately only the UUID, consumed by the shell launcher.
    print(uuid)


if __name__ == '__main__':
    main()
