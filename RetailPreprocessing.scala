// RetailPreprocessing.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, StandardScaler}

object RetailPreprocessing {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Retail Data Preprocessing")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val inputPath = "/Users/raseelalmanea/Downloads/online_retail_II.csv"
    val cleanedOutputPath ="/Users/raseelalmanea/Downloads/cleaned_retail_data"
    val reducedOutputPath ="/Users/raseelalmanea/Downloads/reduced_rfm_customer_level"
    val transformedOutputPath ="/Users/raseelalmanea/Downloads/rfm_transformed"
    // =========================
    // 1. Load Raw Dataset
    // =========================

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    println("Raw dataset loaded")
    println(s"Original rows: ${df.count()}")
    println(s"Original columns: ${df.columns.length}")
    df.printSchema()
    df.show(5, false)

    // =========================
    // 2. Data Cleaning
    // =========================

    val beforeMissing = df.count()

    val missingDescription = df.filter(col("Description").isNull).count()
    val missingCustomerID = df.filter(col("Customer ID").isNull).count()

    println(s"Missing Description rows: $missingDescription")
    println(s"Missing Customer ID rows: $missingCustomerID")

    val dfMissing = df
      .na.fill(Map("Description" -> "Unknown"))
      .na.drop(Seq("Customer ID"))

    val afterMissing = dfMissing.count()

    println(s"Rows after handling missing values: $afterMissing")
    println(s"Rows removed because of missing Customer ID: ${beforeMissing - afterMissing}")

    // Remove invalid Quantity and Price values
    val dfErrors = dfMissing.filter(col("Quantity") > 0 && col("Price") > 0)

    val afterErrors = dfErrors.count()

    println(s"Rows after removing invalid Quantity/Price: $afterErrors")
    println(s"Rows removed due to invalid Quantity/Price: ${afterMissing - afterErrors}")

    // Remove duplicate records
    val duplicateCount = dfErrors.count() - dfErrors.dropDuplicates().count()
    println(s"Duplicate rows detected: $duplicateCount")

    val dfNoDup = dfErrors.dropDuplicates()
    val afterDup = dfNoDup.count()

    println(s"Rows after removing duplicates: $afterDup")
    println(s"Rows removed due to duplicates: ${afterErrors - afterDup}")

    // Detect and remove outliers using IQR on Quantity
    val quantiles = dfNoDup
      .select(col("Quantity").cast(DoubleType))
      .stat
      .approxQuantile("Quantity", Array(0.25, 0.75), 0.01)

    val q1 = quantiles(0)
    val q3 = quantiles(1)
    val iqr = q3 - q1

    val lowerBound = q1 - 1.5 * iqr
    val upperBound = q3 + 1.5 * iqr

    println(s"Q1: $q1")
    println(s"Q3: $q3")
    println(s"IQR: $iqr")
    println(s"Lower bound: $lowerBound")
    println(s"Upper bound: $upperBound")

    val dfClean = dfNoDup.filter(
      col("Quantity").cast(DoubleType) >= lowerBound &&
      col("Quantity").cast(DoubleType) <= upperBound
    )

    val afterOutliers = dfClean.count()

    println(s"Rows after removing Quantity outliers: $afterOutliers")
    println(s"Rows removed as outliers: ${afterDup - afterOutliers}")

    dfClean.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(cleanedOutputPath)

    println(s"Cleaned dataset saved to: $cleanedOutputPath")

    // =========================
    // 3. Data Reduction
    // =========================

    // Feature selection and column renaming
    val slimDF = dfClean
      .drop("_c8")
      .select(
        col("Invoice").alias("InvoiceNo"),
        col("Quantity"),
        col("InvoiceDate"),
        col("Price").alias("UnitPrice"),
        col("Customer ID").alias("CustomerID"),
        col("Country")
      )

    println("Selected relevant columns for RFM analysis")
    println(s"Columns before reduction: ${df.columns.length}")
    println(s"Columns after column reduction: ${slimDF.columns.length}")
    slimDF.show(5, false)

    // Type casting and total amount calculation
    val typedDF = slimDF
      .withColumn("QuantityInt", col("Quantity").cast(IntegerType))
      .withColumn("UnitPriceDbl", col("UnitPrice").cast(DoubleType))
      .withColumn("CustomerIDLong", col("CustomerID").cast(LongType))
      .withColumn("InvoiceTS", to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss"))
      .withColumn("TotalAmount", col("QuantityInt") * col("UnitPriceDbl"))
      .filter(col("QuantityInt").isNotNull)
      .filter(col("UnitPriceDbl").isNotNull)
      .filter(col("CustomerIDLong").isNotNull)
      .filter(col("InvoiceTS").isNotNull)

    typedDF.show(5, false)

    // Reference date for Recency
    val maxTS = typedDF.agg(max(col("InvoiceTS")).alias("MaxTS")).first().getAs[java.sql.Timestamp]("MaxTS")
    println(s"Reference date: $maxTS")

    // Aggregate transaction-level data to customer-level RFM data
    val rfmDF = typedDF
      .groupBy(
        col("CustomerIDLong").alias("CustomerID"),
        col("Country")
      )
      .agg(
        max(col("InvoiceTS")).alias("LastPurchaseTS"),
        countDistinct(col("InvoiceNo")).alias("Frequency"),
        round(sum(col("TotalAmount")), 2).alias("Monetary")
      )
      .withColumn("RecencyDays", datediff(lit(maxTS), col("LastPurchaseTS")))
      .select(
        col("CustomerID"),
        col("Country"),
        col("RecencyDays"),
        col("Frequency"),
        col("Monetary"),
        col("LastPurchaseTS")
      )

    println("Customer-level RFM dataset created")
    println(s"Columns after RFM reduction: ${rfmDF.columns.length}")
    rfmDF.printSchema()
    rfmDF.show(20, false)

    val approxCustomers = typedDF
      .agg(approx_count_distinct("CustomerIDLong").alias("approx_customers"))
      .first()
      .getLong(0)

    println(s"Approximate distinct customers before aggregation: $approxCustomers")
    println(s"Final customer-level rows: ${rfmDF.count()}")

    rfmDF.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(reducedOutputPath)

    println(s"Reduced RFM dataset saved to: $reducedOutputPath")

    // =========================
    // 4. Data Transformation
    // =========================

    val dfParsed = rfmDF.withColumn(
      "LastPurchaseTS_parsed",
      coalesce(
        to_timestamp(col("LastPurchaseTS"), "yyyy-MM-dd HH:mm:ss"),
        to_timestamp(col("LastPurchaseTS"))
      )
    )

    val dfCast = dfParsed
      .drop("LastPurchaseTS")
      .withColumnRenamed("LastPurchaseTS_parsed", "LastPurchaseTS")
      .withColumn("CustomerID", col("CustomerID").cast(IntegerType))
      .withColumn("RecencyDays", col("RecencyDays").cast(IntegerType))
      .withColumn("Frequency", col("Frequency").cast(IntegerType))
      .withColumn("Monetary", col("Monetary").cast(DoubleType))
      .withColumn("LastPurchaseTS", col("LastPurchaseTS").cast(TimestampType))

    dfCast.printSchema()

    // Feature engineering from timestamp
    val dfTime = dfCast
      .withColumn("PurchaseDateOnly", to_date(col("LastPurchaseTS")))
      .withColumn("PurchaseHour", hour(col("LastPurchaseTS")))
      .withColumn("PurchaseMonth", month(col("LastPurchaseTS")))
      .withColumn("DayOfWeek", dayofweek(col("LastPurchaseTS"))) // 1 = Sunday, 7 = Saturday
      .withColumn("is_weekend", when(col("DayOfWeek").isin(6, 7), 1).otherwise(0))

    dfTime.select(
      "CustomerID",
      "LastPurchaseTS",
      "PurchaseHour",
      "PurchaseMonth",
      "DayOfWeek",
      "is_weekend"
    ).show(10, false)

    // Encode Country using StringIndexer
    val countryIndexer = new StringIndexer()
      .setInputCol("Country")
      .setOutputCol("CountryIndex")
      .setHandleInvalid("keep")

    val dfEncoded = countryIndexer.fit(dfTime).transform(dfTime)

    // Assemble features into one vector
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "RecencyDays",
        "Frequency",
        "Monetary",
        "PurchaseHour",
        "PurchaseMonth",
        "is_weekend",
        "CountryIndex"
      ))
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val dfVec = assembler.transform(dfEncoded)

    // Standard scaling
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val scalerModel = scaler.fit(dfVec)
    val dfScaled = scalerModel.transform(dfVec)

    dfScaled.select(
      "CustomerID",
      "Country",
      "RecencyDays",
      "Frequency",
      "Monetary",
      "scaledFeatures"
    ).show(10, false)

    dfScaled
    .withColumn("features", col("features").cast("string"))
    .withColumn("scaledFeatures", col("scaledFeatures").cast("string"))
    .select(
        "CustomerID",
        "Country",
        "CountryIndex",
        "RecencyDays",
        "Frequency",
        "Monetary",
        "LastPurchaseTS",
        "PurchaseHour",
        "PurchaseMonth",
        "DayOfWeek",
        "is_weekend",
        "features",
        "scaledFeatures"
    )
      .coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(transformedOutputPath)

    println(s"Transformed dataset saved to: $transformedOutputPath")

    // =========================
    // 5. Final Summary
    // =========================

    println("========================================")
    println("PREPROCESSING SUMMARY")
    println("Raw Data -> Cleaning -> Reduction -> Transformation -> Final ML Dataset")
    println(s"Original columns: ${df.columns.length}")
    println(s"Reduced columns: ${slimDF.columns.length}")
    println(s"Final transformed columns: ${dfScaled.columns.length}")
    println(s"Rows after cleaning: $afterOutliers")
    println(s"Final customer-level rows: ${rfmDF.count()}")
    println("========================================")

    spark.stop()
  }
}