#!/bin/sh
# Copy the bench to the 3090 machine.  rs_direct_socket.py goes along because bench_motor.py
# imports it from the parent folder (~/System_ID/) and the remote does not have it.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=${1:-syaro@10.8.0.14}
ssh "$HOST" 'mkdir -p ~/System_ID/emf_bench'
rsync -a --exclude __pycache__ --exclude ux_shots "$HERE"/ "$HOST":~/System_ID/emf_bench/
rsync -a "$HERE"/../rs_direct_socket.py "$HOST":~/System_ID/
echo "synced to $HOST:~/System_ID/emf_bench/ (+ ~/System_ID/rs_direct_socket.py)"
