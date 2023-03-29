package org.apache.spark.ml.made

import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._

@Ignore
class StartSparkTest extends AnyFlatSpec with should.Matchers with WithSpark {

  "Spark" should "start context" in {
    val s = spark

    Thread.sleep(60000)
  }

}
