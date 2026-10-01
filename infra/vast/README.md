# vast.ai backup launcher

Modal (`infra/modal_app.py`) is the primary GPU path. vast.ai runs the same `uv run vg ...`
commands over SSH on a rented instance; only the launcher differs.

## Instance filters

| Model | GPU | VRAM | Disk | Other |
|---|---|---|---|---|
| `google/gemma-4-12B-it` (24 GB bf16) | A100 40GB, L40S, A6000 | >= 40 GB | >= 200 GB | CUDA >= 12.4 driver, >= 64 GB RAM for keygen |
| `google/gemma-4-31B-it` (62 GB bf16) | A100 80GB, H100 80GB | >= 80 GB | >= 300 GB | >= 128 GB RAM for keygen |

With the vast.ai CLI (`uv tool install vastai`, then `vastai set api-key ...`):

```bash
vastai search offers 'num_gpus=1 gpu_ram>=40 disk_space>=200 cpu_ram>=64 inet_down>=500 reliability>0.98 cuda_vers>=12.4' -o dph
vastai create instance <offer-id> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 200 --ssh
```

## Run

```bash
ssh -p <port> root@<host> 'bash -s' < infra/vast/setup.sh            # after copying the repo, or:
ssh -p <port> root@<host> 'curl -fsSL <raw setup.sh url> | bash -s -- <repo-url>'
ssh -p <port> root@<host> 'cd /workspace/verifiable-gemma && uv run vg demo --model google/gemma-4-12B-it --device cuda --workdir /workspace/state/demo'
```

Key separation on a single rented box is by process and path only: the server subprocess gets
`public.json` and its state directory, never `key.npz`. Destroy the instance (`vastai destroy
instance <id>`) when done; it deletes the key with it.
