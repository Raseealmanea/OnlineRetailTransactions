import org.apache.spark.sql.SparkSession
import java.io.PrintWriter

case class Customer(
  customerID: Int,
  country: String,
  recencyDays: Int,
  frequency: Int,
  monetary: Double,
  purchaseMonth: Int,
  purchaseHour: Int,
  isWeekend: Int
)

object RDDOperations extends Serializable {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("RDD Operations")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    val inputPath = if (args.nonEmpty) args(0) else "Transaction.csv"

    // Raw RDD
    val rdd = sc.textFile(inputPath)

    // Remove header if present
    val header = rdd.first()
    val dataRDD = rdd.filter(line => line != header)

    // Split rows
    val rows = dataRDD.map(_.split(","))

    // Map (Transformation)
    val customerFeatures = rows.map(row => {
      val customerID = row(0)
      val country = row(1)
      val recency = row(3).toInt
      val frequency = row(4).toInt
      val monetary = row(5).toDouble
      (customerID, country, recency, frequency, monetary)
    })

    // Filter (Transformation)
    val potentialChurnCustomers = customerFeatures.filter {
      case (_, _, recency, frequency, _) => recency > 90 && frequency < 5
    }

    // Customer case class RDD
    val customerRDD = rows.map { row =>
      Customer(
        customerID = row(0).toInt,
        country = row(1),
        recencyDays = row(3).toInt,
        frequency = row(4).toInt,
        monetary = row(5).toDouble,
        purchaseMonth = row(8).toInt,
        purchaseHour = row(7).toInt,
        isWeekend = row(9).toInt
      )
    }

    // 1. flatMap (Transformation)
    val customersFlat = dataRDD.flatMap(line => {
      val cols = line.split(",")
      if (cols.length > 0) List(cols(0)) else List()
    })

    println("=== FlatMap Output (Sample) ===")
    customersFlat.take(5).foreach(println)

    // 2. distinct (Transformation)
    val uniqueCustomers = customersFlat.distinct()

    println("=== Unique Customers (Sample) ===")
    uniqueCustomers.take(5).foreach(println)

    // 3. reduceByKey (Transformation)
    val countryStats = customerRDD
      .map { c =>
        val highRisk = if (c.recencyDays > 90 && c.frequency <= 2 && c.monetary < 300) 1 else 0
        (c.country, (1, highRisk, c.frequency, c.monetary))
      }
      .reduceByKey { (a, b) =>
        (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)
      }

    val countrySummary = countryStats.mapValues {
      case (count, highRisk, freqSum, monetarySum) =>
        (count, highRisk, freqSum.toDouble / count, monetarySum / count, highRisk.toDouble / count)
    }

    println("=== Country Summary (Sample) ===")
    countrySummary.take(5).foreach(println)

    // 4. join (Transformation)
    val customerRiskRDD = customerRDD.map { c =>
      val risk =
        if (c.recencyDays > 90 && c.frequency <= 2 && c.monetary < 300) "High Risk"
        else if (c.recencyDays > 60 && c.frequency <= 4) "Medium Risk"
        else "Low Risk"
      (c.customerID, (c.recencyDays, c.frequency, c.monetary, risk))
    }

    val temporalRDD = customerRDD.map { c =>
      (c.customerID, (c.purchaseMonth, c.purchaseHour, c.isWeekend))
    }

    val enriched = customerRiskRDD.join(temporalRDD)

    println("=== Enriched (Sample) ===")
    enriched.take(5).foreach(println)

    // 5. sortBy (Transformation)
    val mostInactiveChurnCustomers = potentialChurnCustomers.sortBy(
      { case (_, _, recency, _, _) => recency },
      ascending = false
    )

    // 1. count (Action)
    val highValueChurnCustomers = potentialChurnCustomers.filter {
      case (_, _, _, _, monetary) => monetary > 500
    }

    val highValueChurnCount = highValueChurnCustomers.count()
    println("Number of high-value potential churn customers: " + highValueChurnCount)

    // 2. take (Action)
    println("Top 5 most inactive churn-risk customers:")
    mostInactiveChurnCustomers.take(5).foreach(println)

    // 3. first (Action)
    println("=== First Record ===")
    println(rdd.first())

    // 4. collect (Action on small RDD)
    val sample = uniqueCustomers.take(10)
    val sampleRDD = sc.parallelize(sample)

    println("=== Collect Output (Small RDD) ===")
    sampleRDD.collect().foreach(println)

    // 5. reduce (Action)
    val riskScores = customerRiskRDD.map {
      case (_, (r, f, m, l)) =>
        val base = if (l == "High Risk") 3.0 else if (l == "Medium Risk") 2.0 else 1.0
        base + (r / 30.0) + (1.0 / (f + 1)) + (if (m < 100) 1.5 else 1.0)
    }

    val totalRisk = riskScores.reduce(_ + _)
    println("Total Risk Score: " + totalRisk)

    // 6. save using PrintWriter
    val data = enriched.take(100)
    val writer = new PrintWriter("output.txt")

    data.foreach {
      case (id, ((r, f, m, l), (mo, h, w))) =>
        writer.println(s"$id,$r,$f,$m,$l,$mo,$h,$w")
    }

    writer.close()
    println("done")

    spark.stop()
  }
}
