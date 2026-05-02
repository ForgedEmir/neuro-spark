"""
Pipeline d'entraînement : RandomForest MLlib avec tracking MLflow.
Extrait du notebook poc_eeg.ipynb — rendu exécutable standalone.
"""
import os
import sys
import json
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# ── Configuration ──
FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
MODELS_DIR = os.environ.get('MODELS_DIR', '/opt/spark/data/models/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')
MLFLOW_TRACKING = os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/spark/mlruns')

# Hyperparamètres
NUM_TREES = int(os.environ.get('RF_NUM_TREES', 100))
MAX_DEPTH = int(os.environ.get('RF_MAX_DEPTH', 10))
SEED = 42
TRAIN_SUBJECTS = [f'S{i:03d}' for i in range(1, 53)]
TEST_SUBJECTS = [f'S{i:03d}' for i in range(53, 67)]


def create_spark_session():
    return SparkSession.builder \
        .appName('EEG-Training') \
        .master(SPARK_MASTER) \
        .config('spark.executor.memory', '4g') \
        .config('spark.driver.memory', '2g') \
        .config('spark.sql.adaptive.enabled', 'true') \
        .getOrCreate()


def main():
    spark = create_spark_session()

    # ── MLflow setup ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment('neuro-spark-eeg-v2')
    mlflow.pyspark.ml.autolog()

    # ── Lecture des features ──
    df = spark.read.parquet(FEATURES_DIR)
    print(f'Features chargées: {df.count():,} epochs')

    # ── Split par sujet (pas de data leakage) ──
    train_df = df.filter(F.col('subject_id').isin(TRAIN_SUBJECTS))
    test_df = df.filter(F.col('subject_id').isin(TEST_SUBJECTS))
    print(f'Train: {train_df.count():,} epochs ({len(TRAIN_SUBJECTS)} sujets)')
    print(f'Test:  {test_df.count():,} epochs ({len(TEST_SUBJECTS)} sujets)')

    # ── Colonnes features ──
    feature_cols = [c for c in df.columns if c not in
                    ['subject_id', 'run_id', 'epoch_id', 'task_label', 'time']]
    print(f'Features: {len(feature_cols)} colonnes')

    # ── Pipeline MLlib ──
    indexer = StringIndexer(inputCol='task_label', outputCol='label')
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    rf = RandomForestClassifier(
        labelCol='label',
        featuresCol='features',
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        seed=SEED,
        weightCol='weight'
    )

    # Ajout d'une colonne de poids pour compenser le déséquilibre
    train_df = train_df.withColumn(
        'weight',
        F.when(F.col('task_label') == 'T0', 0.5).otherwise(1.0)
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    # ── Entraînement ──
    print(f'Entraînement RandomForest: {NUM_TREES} arbres, profondeur max {MAX_DEPTH}')
    model = pipeline.fit(train_df)

    # ── Évaluation ──
    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction', metricName='accuracy'
    )
    accuracy = evaluator.evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction', metricName='f1'
    ).evaluate(predictions)

    # Métriques par classe (calculées manuellement)
    preds_pd = predictions.select('label', 'prediction', 'task_label').toPandas()
    metrics_per_class = {}
    label_names = {0.0: 'T0', 1.0: 'T1', 2.0: 'T2'}
    for label, name in label_names.items():
        total = (preds_pd['label'] == label).sum()
        correct = ((preds_pd['label'] == label) & (preds_pd['prediction'] == label)).sum()
        recall = correct / total if total > 0 else 0
        metrics_per_class[f'recall_{name}'] = recall

    mlflow.log_metrics(metrics_per_class)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1', f1)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1:       {f1:.4f}')
    for name, recall in metrics_per_class.items():
        print(f'  {name}: {recall:.4f}')

    # ── Sauvegarde modèle ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.write().overwrite().save(f'{MODELS_DIR}/rf_model')

    # Sauvegarde des métriques
    with open('/opt/spark/data/metrics.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'f1': f1,
            **metrics_per_class,
            'num_trees': NUM_TREES,
            'max_depth': MAX_DEPTH,
        }, f, indent=2)

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
