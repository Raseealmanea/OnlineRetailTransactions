import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.clustering.{KMeans, GaussianMixture}
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object MachineLearning {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("CustomerSegmentation")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // =========================
    // Read Dataset
    // =========================
    val inputPath = if (args.nonEmpty) args(0) else "Transaction.csv"

    val ml_df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    // =========================
    // Drop Unnecessary Columns
    // =========================
    val colsToDrop = Seq(
      "CustomerID",
      "LastPurchaseTS",
      "Country",
      "CountryIndex"
    ).filter(ml_df.columns.contains)

    val cleanDF0 = ml_df.drop(colsToDrop: _*)

    // =========================
    // Keep Required Columns
    // =========================
    val requiredCols = Seq(
      "RecencyDays",
      "Frequency",
      "Monetary"
    ).filter(cleanDF0.columns.contains)

    val cleanDF = cleanDF0.na.drop(requiredCols)
      .filter(col("Monetary") < 10000)

    println("=" * 70)
    println("Cleaned Data")
    cleanDF.show(5, false)
    cleanDF.printSchema()

    // =========================
    // Train/Test Split
    // =========================
    val Array(trainDF, testDF) =
      cleanDF.randomSplit(Array(0.7, 0.3), seed = 42)

    println(s"Training Count: ${trainDF.count()}")
    println(s"Testing Count : ${testDF.count()}")

    // =========================
    // Feature Pipeline
    // =========================
    val featureCols = Array(
      "RecencyDays",
      "Frequency",
      "Monetary"
    )

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("assembledFeatures")

    val scaler = new StandardScaler()
      .setInputCol("assembledFeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, scaler))

    val preprocessModel = pipeline.fit(trainDF)

    val trainPrepared = preprocessModel.transform(trainDF)
    val testPrepared = preprocessModel.transform(testDF)

    println("=" * 70)
    println("Prepared Features")
    trainPrepared.select("features").show(5, false)

    // =========================
    // Evaluation Function
    // =========================
    def evaluateClustering(
      predictions: org.apache.spark.sql.DataFrame,
      modelName: String
    ): Double = {

      val evaluator = new ClusteringEvaluator()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMetricName("silhouette")
        .setDistanceMeasure("squaredEuclidean")

      val silhouette = evaluator.evaluate(predictions)

      println("=" * 70)
      println(modelName)
      println("=" * 70)
      println(f"Silhouette Score: $silhouette%.4f")

      println("\nCluster Distribution:")
      predictions.groupBy("prediction")
        .count()
        .orderBy("prediction")
        .show()

      println("\nCluster Profiling:")

      val profile = predictions.groupBy("prediction").agg(
        count("*").alias("customers"),
        avg("RecencyDays").alias("avg_recency"),
        avg("Frequency").alias("avg_frequency"),
        avg("Monetary").alias("avg_monetary")
      ).orderBy("prediction")

      profile.show(false)

      silhouette
    }

    // =========================
    // Find Best K
    // =========================
    var bestK = 2
    var bestKMeansScore = Double.MinValue
    var bestKMeansModel:
      org.apache.spark.ml.clustering.KMeansModel = null

    var bestKMeansPredictions:
      org.apache.spark.sql.DataFrame = null

    for (k <- 2 to 6) {

      val kmeans = new KMeans()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setK(k)
        .setSeed(42)

      val model = kmeans.fit(trainPrepared)

      val predictions = model.transform(testPrepared)

      val evaluator = new ClusteringEvaluator()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMetricName("silhouette")
        .setDistanceMeasure("squaredEuclidean")

      val score = evaluator.evaluate(predictions)

      println(f"K = $k, Silhouette Score = $score%.4f")

      if (score > bestKMeansScore) {
        bestKMeansScore = score
        bestK = k
        bestKMeansModel = model
        bestKMeansPredictions = predictions
      }
    }

    println(s"Best K: $bestK")

    // =========================
    // Evaluate Best K-Means
    // =========================
    val kmeansScore = evaluateClustering(
      bestKMeansPredictions,
      s"K-Means Model with K = $bestK"
    )

    println(f"K-Means Score Saved: $kmeansScore%.4f")

    // =========================
    // Gaussian Mixture Model
    // =========================
    val gmm = new GaussianMixture()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setK(bestK)
      .setSeed(42)

    val gmmModel = gmm.fit(trainPrepared)

    val gmmPredictions = gmmModel.transform(testPrepared)

    val gmmScore = evaluateClustering(
      gmmPredictions,
      s"Gaussian Mixture Model with K = $bestK"
    )

    println("GMM Weights:")
    println(gmmModel.weights.mkString("[", ", ", "]"))

    // =========================
    // Summary Comparison
    // =========================
    println("=" * 70)
    println("SUMMARY COMPARISON")
    println("=" * 70)

    println(f"Best K-Means Silhouette Score: $kmeansScore%.4f")
    println(f"Gaussian Mixture Silhouette Score: $gmmScore%.4f")

    if (kmeansScore > gmmScore) {
      println("Best Model: K-Means")
    }
    else if (gmmScore > kmeansScore) {
      println("Best Model: Gaussian Mixture Model")
    }
    else {
      println("Both models achieved the same score.")
    }

    spark.stop()
  }
}