import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from kedro.framework.hooks import hook_impl


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        # Load Spark settings from conf/base/spark.yml (overridable per environment)
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        master = os.environ.get("SPARK_MASTER", "local[2]")
        local_dir = os.environ.get("SPARK_LOCAL_DIR", "spark-tmp")

        SparkSession.builder.appName("EEG-Motor-Imagery").master(master).config(
            "spark.local.dir", local_dir
        ).config(conf=spark_conf).getOrCreate()
