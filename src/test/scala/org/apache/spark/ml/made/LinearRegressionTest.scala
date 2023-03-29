package org.apache.spark.ml.made

import breeze.linalg._
import com.google.common.io.Files
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val inputData = LinearRegressionTest._dataFrame

  private def checkPredict(model: LinearRegressionModel, testCase: Int = 0) = {
    val delta = 0.1
    lazy val realWeights = LinearRegressionTest._realWeights
    lazy val realBias = LinearRegressionTest._realBias

    testCase match {
      case 0 => {
        model.weights(0) should be(realWeights.valueAt(0) +- delta)
        model.weights(1) should be(realWeights.valueAt(1) +- delta)
        model.weights(2) should be(realWeights.valueAt(2) +- delta)

        model.bias should be(realBias +- delta)
      }
      case 1 => {
        model.weights(0) should be(0.0 +- delta)
        model.weights(1) should be(0.0 +- delta)
        model.weights(2) should be(0.0 +- delta)

        model.bias should be(1.0 +- delta)
      }
      case 2 => {
        model.weights(0) should not be (realWeights.valueAt(0) +- delta)
        model.weights(1) should not be (realWeights.valueAt(1) +- delta)
        model.weights(2) should not be (realWeights.valueAt(2) +- delta)
        model.bias should not be (realBias +- delta)
      }
    }


  }

  "Model" should "can predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.dense(0.0, 0.0, 0.0).toDense,
      bias = 2.0
    ).setInputCol("features").setLabelCol("target")

    val vectors = model.transform(inputData)
    vectors.count() should be(100000)
  }

  "Estimator" should "compare weights and bias" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("target")
      .setOutputCol("prediction")

    val model = estimator.fit(inputData)
    checkPredict(model)
  }

  "Estimator" should "read and calculate weights and bias" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setLabelCol("target")
        .setOutputCol("prediction")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val pipelineRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = pipelineRead.fit(inputData).stages(0).asInstanceOf[LinearRegressionModel]
    checkPredict(model)
  }

  "Estimator" should "not learn set num iter = 0" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("target")
      .setOutputCol("prediction")

    estimator.setNumIterations(0)

    val model = estimator.fit(inputData)
    checkPredict(model, testCase = 1)
  }

  "Estimator" should "not learn with 0 learning rate" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("target")
      .setOutputCol("prediction")

    estimator.setLearningRate(0.0)

    val model = estimator.fit(inputData)

    checkPredict(model, testCase = 1)
  }

  "Estimator" should "not learn with negative learning rate" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("target")
      .setOutputCol("prediction")

    estimator.setLearningRate(-1.0)

    val model = estimator.fit(inputData)
    checkPredict(model, testCase = 2)
  }

}

object LinearRegressionTest extends WithSpark {
  lazy val _realWeights = Vectors.dense(1.5, 0.3, -0.7).asBreeze.toDenseVector
  lazy val _realBias: Double = 1.7

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    Seq.fill(100000)(
      Vectors.fromBreeze(DenseVector.rand(3))
    ).map(x => Tuple1(x)).toDF("features")
  }
  lazy val transformUdf =
    udf((x: Vector) => {
      _realWeights.dot(x.asBreeze) + _realBias
    })

  lazy val _dataFrame = _data.withColumn("target", transformUdf(_data("features")))
}