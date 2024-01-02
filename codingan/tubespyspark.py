from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, StandardScaler
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import DoubleType
import warnings
warnings.filterwarnings("ignore")



spark = SparkSession.builder.master("local[*]") \
    .appName("Python Spark K-Means") \
    .getOrCreate()

  df = spark.read.format("csv").option("header", "true").load("/home/ubuntu/codingan/movies_dataset.csv")
  df = df.withColumn("Rating", df["Rating"].cast(DoubleType()))

  input_col = ['Rating']

  vec_assembler = VectorAssembler(inputCols = input_col,
                                outputCol = "features")

  final_df = vec_assembler.transform(df)

  model = KMeans(featuresCol = "features", k=5)

  print(model)

  model.transform(final_df).groupBy("prediction").count().show()

  predictions = model.transform(final_df)

  print(predictions.show())