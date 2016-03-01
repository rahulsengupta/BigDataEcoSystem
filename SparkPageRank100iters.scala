//This code uses GraphX to calculate PageRank over a Wikipedia WEX Dataset
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.xml.{XML,NodeSeq}

object SimpleApp {
  def main(args: Array[String]) {
  
	val conf = new SparkConf().setAppName("SparkPageRank")
    // sc is the SparkContext.
	val sc = new SparkContext(conf)
	//Don't add the above 2 lines on local, sc is already made
	
	//Start actual code from here
	val inputFile = "s3://rahulassg2/wex.txt"
	val input = sc.textFile(inputFile)
	
	println("Count vertices")
    val numVertices = input.count()
	println(numVertices)
	//Parse the input file to get article links
	println("Making graph..")
    var vertices = input.map(line => {
      val fields = line.split("\t")
      val (title, body) = (fields(1), fields(3).replace("\\n", "\n"))
      val links =
        if (body == "\\N") {
          NodeSeq.Empty
        } else {
          try {
            XML.loadString(body) \\ "link" \ "target"
          } catch {
            case e: org.xml.sax.SAXParseException =>
              System.err.println("Article \"" + title + "\" has malformed XML in body:\n" + body)
            NodeSeq.Empty
          }
        }
      val outEdges = links.map(link => new String(link.text)).toArray
      val id = new String(title)
      (id,(1.0 / numVertices, outEdges))
    })
	println("Parsing done")
	
	val verticesprime = vertices.map( line => (line._1 , (line._2)._2))	
	val links = verticesprime
	var ranks = links.mapValues(v => 1.0)
	
	for (i <- 1 to 100) { //Give number of iterations here
      val contribs = links.join(ranks).values.flatMap{ case (urls, rank) =>
        val size = urls.size
        urls.map(url => (url, rank / size))
      }
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }
	println("Writing overall outputs..")
	val output = ranks.collect()
	val outputrdd = sc.parallelize(output)
	outputrdd.saveAsTextFile("s3://rahulassg2/outputpageranksspark100iters.txt")
    val outputtop100 = outputrdd.top(100)(Ordering.by(_._2))
	val outputtop100rdd = sc.parallelize(outputtop100)
	outputtop100rdd.saveAsTextFile("s3://rahulassg2/outputtop100sparkx100iters.txt")
	
	println("Writing university outputs..")
	
	val univs = output.filter(line => line._1.contains("University"))
	val univsrdd = sc.parallelize(univs)
	univsrdd.saveAsTextFile("s3://rahulassg2/univspageranksspark100iters.txt")
	val univstop100 = univsrdd.top(100)(Ordering.by(_._2))
	val univstop100rdd = sc.parallelize(univstop100)
	univstop100rdd.saveAsTextFile("s3://rahulassg2/univstop100spark100iters.txt")
	
	println("Done writing..")
	
	
  }
}