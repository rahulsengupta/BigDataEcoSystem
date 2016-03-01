//This code uses GraphX to calculate PageRank over a Wikipedia WEX Dataset
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.xml.{XML,NodeSeq}

object SimpleApp {
  def main(args: Array[String]) {
  
	val conf = new SparkConf().setAppName("GraphXPageRank")
    // sc is the SparkContext.
	val sc = new SparkContext(conf)
	//Don't add the above 2 lines on local, sc is already made
	
	//Start actual code from here
	val inputFile = "s3://rahulassg2/wex.txt"
	val input = sc.textFile(inputFile)
	
	println("Count vertices..")
    val numVertices = input.count()
	println(numVertices)
	//Parse the input file to get article links
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
      (id,(outEdges))
    })
	println("Parsing done..")
	
	// Hash function to assign a Hash ID to each article
	def pageHash(title: String): VertexId = {
		title.toLowerCase.replace(" ", "").hashCode.toLong
	}
	
	println("Making graph..")
	//val verticesprime = vertices.map( line => (line._1 , (line._2)._2))	
	val vvertices = vertices.map(line => (pageHash(line._1) , line._1) )
	var linktable = vertices.flatMap(line => {line._2.map {ele => (line._1,ele)}})
	val table = linktable.map(line => ( pageHash(line._1) , pageHash(line._2), 1.0 ) )
	val eedges = table.map(line => Edge(line._1 , line._2 , line._3) )
	val graph = Graph(vvertices, eedges, "").subgraph(vpred = {(v, d) => d.nonEmpty}).cache
	val prGraph = graph.staticPageRank(10).cache //Give number of iterations here
	
	val titleAndPrGraph = graph.outerJoinVertices(prGraph.vertices) {
	(v, title, rank) => (rank.getOrElse(0.0), title)
	}
	
	println("Writing overall outputs..")
	val outputtable = titleAndPrGraph.vertices

	val output = outputtable.map(line => line._2)
	val outputrdd = output
	outputrdd.saveAsTextFile("s3://rahulassg2/outputpageranksgraphx10iters.txt")
    val outputtop100 = outputrdd.top(100)(Ordering.by(_._1))
	val outputtop100rdd = sc.parallelize(outputtop100)
	outputtop100rdd.saveAsTextFile("s3://rahulassg2/outputtop100graphx10iters.txt")
	
	println("Writing university outputs..")
	val univstable = titleAndPrGraph.vertices
	val univstemp = univstable.map(line => line._2)
	val univs = univstemp.filter(line => line._2.contains("University"))
	val univsrdd = univs
	univsrdd.saveAsTextFile("s3://rahulassg2/univspageranksgraphx10iters.txt")
	val univstop100 = univsrdd.top(100)(Ordering.by(_._2))
	val univstop100rdd = sc.parallelize(univstop100)
	univstop100rdd.saveAsTextFile("s3://rahulassg2/univstop100graphx10iters.txt")
	
	println("Done writing..")

	
  }
}