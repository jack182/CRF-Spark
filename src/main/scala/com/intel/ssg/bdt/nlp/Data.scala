/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.ssg.bdt.nlp

import scala.collection.mutable.ArrayBuffer

/**
  * Class that represents the columns of a token.
  *
  * @param label The last column for this token.
  * @param tags List of tags for this token, expect for the last label.
  */
class Token(
    val label: String,
    val tags: Array[String]) extends Serializable {
  var prob : Array[(String, Double)] = null

  def setProb(probMat: Array[(String, Double)]): Token ={
    this.prob = probMat
    this
  }

  override def toString: String = {
    s"$label|--|${tags.mkString("|-|")}"
  }

  def compare(other: Token): Int = {
    if(this.label == other.label) 1 else 0
  }
}

object Token {
  /**
    * Parses a string resulted from `LabeledToken#toString` into
    * an [[com.intel.ssg.bdt.nlp.Token]].
    *
    */
  def deSerializer(s: String): Token = {
    val parts = s.split("""\|--\|""")
    val label = parts(0)
    val tags = parts(1).split("""\|-\|""")
    Token.put(label, tags)
  }

  def serializer(token: Token): String = {
    token.toString
  }

  def probSerializer(token: Token): String = {
    val strRes = new StringBuffer()
    strRes.append( token.tags.mkString("\t") )
    strRes.append( "\t" + token.label + "\t")
    strRes.append(token.prob.map{
      case (str, p) => str + "/" + p.toString
    }.mkString("\t") )
    strRes.toString
  }

  def put(label: String, tags: Array[String]) = {
    new Token(label, tags)
  }

  def put(tags: Array[String]) = {
    new Token(null, tags)
  }
}

/**
  * Class that represents the tokens of a sentence.
  *
  * @param sequence List of tokens
  */
case class Sequence (sequence: Array[Token]) extends Serializable {
  var seqProb = 0.0

  var candidates: ArrayBuffer[ArrayBuffer[Int]] = null
  var probN: ArrayBuffer[Double] = null
  var labels: ArrayBuffer[String] = null

  def setLabels(labels: ArrayBuffer[String]): Sequence = {
    this.labels = labels
    this
  }

  def setSeqProb(seqProb: Double): Sequence ={
    this.seqProb = seqProb
    this
  }

  def setCandidates(nBest: ArrayBuffer[ArrayBuffer[Int]]): Sequence = {
    this.candidates = nBest
    this
  }

  def setProbN(probN: ArrayBuffer[Double]):Sequence = {
    this.probN = probN
    this
  }

  def showNthBest( k: Int): String = {
    val strRes = new ArrayBuffer[String]()
    strRes.append("#" + k + "\t" +this.probN(k).toString)

    val pairs = this.candidates(k).zip(this.toArray)
    for((t, token) <- pairs) {
      strRes.append(token.tags.mkString("\t") + "\t" + labels(t))
    }
    strRes.mkString("\n")
  }

  def showAll(): String = {
    val idx = candidates.indices
    idx.map(t =>showNthBest(t))
                               .mkString("\n")
  }

  override def toString: String = {
    s"${sequence.mkString("\t")}"
  }

  def toArray: Array[Token] = sequence

  def compare(other: Sequence): Int = {
    this.toArray.zip(other.toArray).map{case(one, two) => one.compare(two)}.sum
  }
}

object Sequence {
  def deSerializer(s: String): Sequence = {
    val tokens = s.split("\t")
    Sequence(tokens.map(Token.deSerializer))
  }
  def serializer(sequence: Sequence): String = {
    sequence.toString
  }
  def probSerializer(sequence: Sequence): String = {
    val strRes = new ArrayBuffer[String]()
    strRes.append("#" + sequence.seqProb.toString)
    strRes ++= sequence.toArray.map( token =>Token.probSerializer(token) )
    strRes.mkString("\n")
  }
}
