#!/bin/bash

srun -Q -J debug \
    -w ailb-login-02 \
    --immediate=20 \
    --partition=all_serial \
    --gres=gpu:1 \
    --time=00:10:00 \
    --account=cvcs2026 \
    --pty /bin/bash