import org.apache.spark.sql.SparkSession

object SQLOperations {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("SQL_Operations")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // =========================
    // Read Dataset
    // =========================
    val inputPath = if (args.nonEmpty) args(0) else "Transaction.csv"

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)

    df.printSchema()
    df.show(5, false)

    // =========================
    // Create Temp View
    // =========================
    df.createOrReplaceTempView("customers")

    // ==================================================
    // Query 1
    // ==================================================
    val q1 = spark.sql("""
      SELECT
          Country,
          COUNT(DISTINCT CustomerID) AS total_customers,
          ROUND(AVG(RecencyDays), 2) AS avg_recency,
          ROUND(AVG(Frequency), 2) AS avg_frequency,
          ROUND(AVG(Monetary), 2) AS avg_monetary,
          SUM(CASE WHEN Monetary > 500 THEN 1 ELSE 0 END) AS high_value_customers
      FROM customers
      GROUP BY Country
      HAVING COUNT(DISTINCT CustomerID) >= 5
      ORDER BY avg_monetary DESC, avg_frequency DESC
    """)

    println("===== Query 1 =====")
    q1.show(20, false)

    // ==================================================
    // Query 2
    // ==================================================
    val q2 = spark.sql("""
      SELECT
          CustomerID,
          Country,
          RecencyDays,
          Frequency,
          Monetary,
          CASE
              WHEN RecencyDays > 180 AND Frequency < 5 AND Monetary > 500 THEN 'Critical Risk'
              WHEN RecencyDays > 90 AND Frequency < 5 AND Monetary > 500 THEN 'High Risk'
              WHEN RecencyDays > 90 AND Frequency < 8 THEN 'Moderate Risk'
              ELSE 'Lower Risk'
          END AS risk_segment
      FROM customers
      WHERE Monetary > 500
      ORDER BY RecencyDays DESC, Monetary DESC
      LIMIT 20
    """)

    println("===== Query 2 =====")
    q2.show(20, false)

    // ==================================================
    // Query 3
    // ==================================================
    val q3 = spark.sql("""
      SELECT CustomerID, Country, RecencyDays, Frequency, Monetary
      FROM customers
      WHERE RecencyDays > 90 AND Frequency < 5
      ORDER BY RecencyDays DESC
    """)

    println("===== Query 3 =====")
    q3.show(10, false)

    // ==================================================
    // Query 4
    // ==================================================
    val q4 = spark.sql("""
      SELECT
          Country,
          COUNT(*) AS total_customers,
          SUM(CASE WHEN RecencyDays > 90 AND Frequency < 5 THEN 1 ELSE 0 END) AS churn_customers,
          ROUND(
              SUM(CASE WHEN RecencyDays > 90 AND Frequency < 5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
              2
          ) AS churn_percentage
      FROM customers
      GROUP BY Country
      ORDER BY churn_percentage DESC
    """)

    println("===== Query 4 =====")
    q4.show(10, false)

    // ==================================================
    // Query 5
    // ==================================================
    val q5 = spark.sql("""
      WITH risk_segmented AS (
          SELECT
              CustomerID,
              Country,
              RecencyDays,
              Frequency,
              Monetary,
              CASE
                  WHEN RecencyDays > 90 AND Frequency <= 2 AND Monetary > 300 THEN 'Critical Risk'
                  WHEN RecencyDays > 60 AND Frequency <= 4 THEN 'Moderate Risk'
                  ELSE 'Low Risk'
              END AS risk_segment
          FROM customers
      ),
      ranked_customers AS (
          SELECT
              CustomerID,
              Country,
              RecencyDays,
              Frequency,
              Monetary,
              risk_segment,
              DENSE_RANK() OVER(PARTITION BY Country ORDER BY Monetary DESC) AS rank_in_country
          FROM risk_segmented
          WHERE risk_segment IN ('Critical Risk', 'Moderate Risk')
      )
      SELECT *
      FROM ranked_customers
      WHERE rank_in_country <= 3
      ORDER BY Country, rank_in_country
    """)

    println("===== Query 5 =====")
    q5.show(20, false)

    // ==================================================
    // Query 6
    // ==================================================
    val q6 = spark.sql("""
      WITH risk_flags AS (
          SELECT
              Country,
              CustomerID,
              RecencyDays,
              Frequency,
              Monetary,
              CASE
                  WHEN RecencyDays > 90 AND Frequency <= 2 AND Monetary > 300 THEN 1
                  ELSE 0
              END AS high_risk_flag
          FROM customers
      ),
      country_analysis AS (
          SELECT
              Country,
              COUNT(*) AS total_customers,
              SUM(high_risk_flag) AS high_risk_customers,
              ROUND(100.0 * SUM(high_risk_flag) / COUNT(*), 2) AS churn_percentage,
              ROUND(AVG(RecencyDays), 2) AS avg_recency,
              ROUND(AVG(Frequency), 2) AS avg_frequency,
              ROUND(AVG(Monetary), 2) AS avg_monetary,
              ROUND(percentile_approx(Monetary, 0.5), 2) AS median_monetary,
              ROUND(VARIANCE(Monetary), 2) AS monetary_variance
          FROM risk_flags
          GROUP BY Country
          HAVING COUNT(*) >= 10
      )
      SELECT *
      FROM country_analysis
      ORDER BY churn_percentage DESC, avg_monetary DESC
    """)

    println("===== Query 6 =====")
    q6.show(20, false)

    spark.stop()
  }
}