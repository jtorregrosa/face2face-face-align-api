#!/bin/sh
. ./venv/bin/activate

BIND_ADDRESS='0.0.0.0:8000'

ARGS="--bind ${BIND_ADDRESS} --workers ${NUM_WORKERS} --worker-class ${WORKER_CLASS} --threads ${NUM_THREADS_PER_WORKER}" make gunicorn