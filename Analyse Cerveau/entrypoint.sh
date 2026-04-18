#!/bin/bash
set -e

SPARK_WORKLOAD=$1

# Signal trapping for graceful shutdown
cleanup() {
    echo "[entrypoint] Caught signal, shutting down..."
    if [ -n "$SPARK_MASTER_PID" ]; then
        kill -TERM "$SPARK_MASTER_PID" 2>/dev/null || true
        wait "$SPARK_MASTER_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "[entrypoint] Workload: $SPARK_WORKLOAD"

case "$SPARK_WORKLOAD" in
    master)
        start-master.sh -p 7077
        SPARK_MASTER_PID=$!
        wait "$SPARK_MASTER_PID"
        ;;
    worker*)
        # Wait for master to be ready
        echo "[entrypoint] Waiting for spark-master..."
        until curl -sf http://spark-master:8080 > /dev/null 2>&1; do
            sleep 2
        done
        echo "[entrypoint] spark-master is ready, starting worker"
        start-worker.sh spark://spark-master:7077
        SPARK_MASTER_PID=$!
        wait "$SPARK_MASTER_PID"
        ;;
    history)
        start-history-server.sh
        SPARK_MASTER_PID=$!
        wait "$SPARK_MASTER_PID"
        ;;
    *)
        echo "[entrypoint] Unknown workload: $SPARK_WORKLOAD"
        echo "Usage: entrypoint.sh [master|worker|history]"
        exit 1
        ;;
esac
