from neuro_spark.hooks import SparkHooks

HOOKS = (SparkHooks(),)

# Register spark.yml as a known config pattern so context.config_loader["spark"] works
CONFIG_LOADER_ARGS = {
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    }
}
