# CRF-Spark-update
A Spark-based implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data.

## Requirements
This documentation is for Spark 1.4+. Other version will probably work yet not tested.

## Features

`CRF-Spark` provides following features:
* Training in parallel based on Spark RDD
* Support a simple format of training and test file. Any other format also can be read by a simple tokenizer.
* A common feature templates design, which is also used in other machine learning tools, such as [CRF++](https://taku910.github.io/crfpp/) and [miralium](https://code.google.com/archive/p/miralium/)
* Fast training based on Limited-memory BFGS optimizaton algorithm (fitting L2-regularized models) or Orthant-wise Limited Memory Quasi-Newton optimizaton algorithm (fitting L1-regularized models)
* Support two verbose levels to provide extra information. VerboseLevel1 shows marginal probabilities for each tag and a conditional probability for the entire output; VerboseLevel2 shows marginal probabilities for all other candidates additionally.
* Support n-best outputs
* Linear-chain (first-order Markov) CRF
* Test can run both in parallel and in serial

## Applications
[A web-based application](https://github.com/gkq/Web-CRF) of this package to Chinese POS Tagging.

## Example

### Scala API

```scala
  val template = Array("U00:%x[-1,0]", "U01:%x[0,0]", "U02:%x[1,0]", "B")

  val train1 = Array("B-NP|--|Friday|-|NNP\tB-NP|--|'s|-|POS", "I-NP|--|Market|-|NNP\tI-NP|--|Activity|-|NN")
  val test1 = Array("null|--|Market|-|NNP\tnull|--|Activity|-|NN")
  val trainRdd1 = sc.parallelize(train).map(Sequence.deSerializer)
  val model1 = CRF.train(template, trainRdd)
  val result1 = model.predict(test.map(Sequence.deSerializer))
  
  val train2 = Array("Friday NNP B-NP\nB-NP 's POS", "Market NNP I-NP\nActivity NN I-NP")
  val test2 = Array("Market NNP\nActivity NN")
  val trainRdd2 = sc.parallelize(train2).map(sentence => {
    val tokens = sentence.split("\n")
    Sequence(tokens.map(token => {
      val tags: Array[String] = token.split(' ')
      Token.put(tags.last, tags.dropRight(1))
    }))
  })
  val model2 = CRF.train(template, trainRdd2)
  val result2 = model2.predict(test2.map(sentence => {
    val tokens = sentence.split("\n")
    Sequence(tokens.map(token => {
      val tags: Array[String] = token.split(' ')
      Token.put(tags)
    }))
  }))
```

### Building From Source

```scala
sbt package
```

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * qian.huang@intel.com
 * jiayin.hu@intel.com
 * rui.sun@intel.com
 * hao.cheng@intel.com
