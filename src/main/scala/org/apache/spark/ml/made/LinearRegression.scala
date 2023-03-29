package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}


trait LinearRegressionParams extends HasInputCol with HasLabelCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  val numIterations = new IntParam(this, "numIterations", "number of iterations")

  def setNumIterations(value: Int): this.type = set(numIterations, value)

  def getNumIterations: Int = $(numIterations)

  setDefault(numIterations -> 1000)

  val learningRate = new DoubleParam(this, "learningRate", "learning rate for gradient descent")

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def getLearningRate: Double = $(learningRate)

  setDefault(learningRate -> 0.1)


  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))


  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    val sizeInput = dataset.select(dataset($(inputCol))).first.getAs[Vector](0).size
    val weights = Vectors.zeros(sizeInput).asBreeze.toDenseVector
    var bias = 1.0

    for (i <- 1 to $(numIterations)) {
      val summary = dataset.select(dataset($(inputCol)), dataset($(labelCol))).rdd.mapPartitions((data: Iterator[Row]) => {
        val result = data.foldLeft(new MultivariateOnlineSummarizer())(
          (summarizer, vector) => summarizer.add(
            {
              val X = vector.getAs[Vector](0).asBreeze.toDenseVector
              val y = vector.getDouble(1)

              val eps = (X.dot(weights)) - y + bias
              mllib.linalg.Vectors.dense((eps * X).toArray :+ eps)
            }
          ))
        Iterator(result)
      }).reduce(_ merge _)

      val gradient = summary.mean.asML.asBreeze.toDenseVector
      weights -= gradient(0 until weights.size) * $(learningRate)
      bias -= gradient(weights.size) * $(learningRate)
    }


    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights), bias)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          (weights.asBreeze dot x.asBreeze) + bias
        })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Double) = weights.asInstanceOf[Vector] -> bias

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encodervector: Encoder[Vector] = ExpressionEncoder()
      implicit val encoderdouble: Encoder[Double] = ExpressionEncoder()

      val (weights, bias) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}