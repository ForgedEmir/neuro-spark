#!/usr/bin/env bash
# Démo EEG BCI tout-en-un (mode INTERACTIF) :
#   1. cleanup résiduels
#   2. reset des répertoires de stream
#   3. pré-amorçage d'un epoch (pour que Spark démarre avec un schéma)
#   4. Spark Structured Streaming EEG en background
#   5. dashboard Dash sur :8052 → toi tu pilotes les epochs via les boutons
#
# Ctrl+C → cleanup automatique de tous les enfants.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="data/stream_eeg"
SPARK_LOG="$LOG_DIR/spark.log"

PIDS=()
cleanup() {
    echo
    echo "==> Cleanup EEG : arrêt des processus enfants..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo "==> Démo EEG terminée. Logs Spark : $SPARK_LOG"
}
trap cleanup EXIT INT TERM

echo "==> [0/3] Kill des processus EEG résiduels"
pkill -f "kedro run --pipeline streaming_eeg" 2>/dev/null && sleep 2 || true
pkill -f "dashboard_eeg.py" 2>/dev/null || true

echo "==> [1/3] Reset des répertoires"
rm -rf data/stream_eeg/input data/stream_eeg/output data/stream_eeg/checkpoint
mkdir -p data/stream_eeg/input

# Pour que Spark déduise les colonnes au démarrage, on dépose un epoch initial
# (Spark sait quels canaux EEG existent grâce au peek dans data/parquet de toute façon,
# mais on évite des warnings)
echo "==> [1bis] Spark va lire le schéma depuis data/parquet/"

echo "==> [2/3] Lancement Spark Streaming EEG (background)"
KEDRO_DISABLE_TELEMETRY=1 \
SPARK_MASTER=local[2] \
SPARK_LOCAL_DIR=spark-tmp \
JAVA_HOME=/usr/lib/jvm/java-17-openjdk \
PYSPARK_PYTHON=.venv/bin/python \
PYSPARK_DRIVER_PYTHON=.venv/bin/python \
    .venv/bin/python -W ignore -m kedro run --pipeline streaming_eeg \
    > "$SPARK_LOG" 2>&1 &
SPARK_PID=$!
PIDS+=("$SPARK_PID")
echo "    Spark PID = $SPARK_PID, logs : $SPARK_LOG"

echo "==> Attente que Spark soit prêt (premier run = entraînement sklearn, ~60 s max)..."
for i in $(seq 1 90); do
    if grep -q "streaming démarré" "$SPARK_LOG" 2>/dev/null; then
        echo "    Spark prêt après ${i}s"
        break
    fi
    if ! kill -0 "$SPARK_PID" 2>/dev/null; then
        echo "!!! Spark s'est arrêté prématurément. Logs :"
        tail -30 "$SPARK_LOG"
        exit 1
    fi
    sleep 1
done

echo
echo "==> [3/3] Dashboard EEG sur http://localhost:8052"
echo "    🎮 Mode INTERACTIF : clique sur les boutons pour envoyer des epochs."
echo "    (Ctrl+C ici arrête TOUT)"
echo
.venv/bin/python dashboard_eeg.py
