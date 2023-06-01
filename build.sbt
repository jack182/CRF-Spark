name := "crf-spark"

version := "0.1.0"

scalaVersion := "2.12.10"

spName := "hqzizania/crf-spark"

sparkVersion := "3.1.1"

sparkComponents += "mllib"

resolvers += Resolver.sonatypeRepo("public")

/********************
  * Release settings *
  ********************/

spShortDescription := "crf-spark"

spDescription := """A Spark-based implementation of Conditional Random Fields (CRFs) for labeling sequential data""".stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := false

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

