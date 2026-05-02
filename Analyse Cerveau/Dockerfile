# ── NeuroSpark Dockerfile (optimisé) ─────────────────────────────────────────
# Multi-stage : sépare spark-base (JVM + OS) de pyspark (Python) de runtime
# Optimisé layer caching : apt d'abord, puis Spark, puis pip, puis jars

# === Stage 1: Base (JDK + OS utilities) ===
FROM python:3.11-bullseye AS spark-base

# Install OS deps — couche la plus stable, reste en cache longtemps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo curl vim unzip rsync \
        openjdk-11-jdk build-essential \
        software-properties-common ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Spark & Hadoop
ENV SPARK_VERSION=3.5.5
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV SPARK_MASTER_PORT=7077
ENV SPARK_MASTER_HOST=spark-master
ENV SPARK_MASTER="spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT"
ENV PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
ENV PYSPARK_PYTHON=python3
ENV IJAVA_CLASSPATH=/opt/spark/jars/*

RUN mkdir -p ${HADOOP_HOME} ${SPARK_HOME}
WORKDIR ${SPARK_HOME}

# Download Spark — archive lourde, mise en cache après apt
RUN curl -fsSL \
    https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    -o spark.tgz && \
    tar xzf spark.tgz --strip-components 1 && \
    rm spark.tgz

RUN chmod u+x /opt/spark/sbin/* /opt/spark/bin/*
ENV PATH="$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin"

# Spark config
RUN echo "spark.master                           spark://spark-master:7077\n\
spark.eventLog.enabled                 true\n\
spark.eventLog.dir                     /opt/spark/spark-events\n\
spark.history.fs.logDirectory          /opt/spark/spark-events" \
    > "$SPARK_HOME/conf/spark-defaults.conf"


# === Stage 2: Python dependencies ===
FROM spark-base AS pyspark

# Install Python deps — change quand requirements.txt change
RUN pip3 install --no-cache-dir \
    pyarrow==16.1.0 \
    pyspark==3.5.1 \
    delta-spark==3.2.0 \
    mne==1.8.0 \
    plotly==5.22.0 \
    scipy==1.14.1 \
    pandas \
    mlflow==2.20.0


# === Stage 3: Runtime (jars + entrypoint) ===
FROM pyspark AS pyspark-runner

# Delta Lake jars
RUN curl -fsSL \
    https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.4.0/delta-core_2.12-2.4.0.jar \
    -o /opt/spark/jars/delta-core_2.12-2.4.0.jar && \
    curl -fsSL \
    https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar \
    -o /opt/spark/jars/delta-spark_2.12-3.2.0.jar && \
    curl -fsSL \
    https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar \
    -o /opt/spark/jars/delta-storage-3.2.0.jar && \
    curl -fsSL \
    https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.4_2.12/1.4.3/iceberg-spark-runtime-3.4_2.12-1.4.3.jar \
    -o /opt/spark/jars/iceberg-spark-runtime-3.4_2.12-1.4.3.jar && \
    curl -fsSL \
    https://repo1.maven.org/maven2/org/apache/hudi/hudi-spark3-bundle_2.12/0.15.0/hudi-spark3-bundle_2.12-0.15.0.jar \
    -o /opt/spark/jars/hudi-spark3-bundle_2.12-0.15.0.jar

COPY entrypoint.sh /opt/spark/entrypoint.sh
RUN sed -i 's/\r$//' /opt/spark/entrypoint.sh && chmod u+x /opt/spark/entrypoint.sh


# === Stage 4: Jupyter (optionnel) ===
FROM pyspark-runner AS pyspark-jupyter

RUN pip3 install --no-cache-d notebook
ENV JUPYTER_PORT=8889
ENV PYSPARK_DRIVER_PYTHON=jupyter
ENV PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --allow-root --ip=0.0.0.0 --port=${JUPYTER_PORT}"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
