# Project Title: Online Retail Transactions

1.Environmental setup:
- Install Java Development Kit (JDK 17).
- Install SBT (Scala Build Tool) version 1.x.
- Set JAVA_HOME environment variable to the JDK 17 path.

2. HOW TO RUN THE SCRIPTS
-------------------------
Open the terminal in the project root directory and run the following command

- To run Preprocessing:
  sbt "runMain RetailPreprocessing"
  
- To run RDD Analysis:
  sbt "runMain 03_SQLOperations.scala"

- To run SQL Analysis:
  sbt "runMain 02_RDDOperations.scala"

- To run Machine Learning:
  sbt "runMain 04_MachineLearning.scala"

3. SPARK VERSION & DEPENDENCIES
-------------------------------
- Apache Spark Version: 3.5.1
- Scala Version: 2.12.18
- Dependencies (managed via build.sbt):
  * spark-core
  * spark-sql
  * spark-mllib

============================================================
