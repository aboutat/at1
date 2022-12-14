package 图书推荐

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS

object Driver {

  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setMaster("local").setAppName("a")
    val sc=new SparkContext(conf)
    val data=sc.textFile("D://test/a/userdata.txt")
    //第一步:数据格式转换
    val ratings=data.map { line =>
      val info=line.split("\t")
      val userId=info(0).toInt
      val bookId=info(1).toInt
      val score=info(2).toInt
      Rating(userId,bookId,score)
    }
    //第二步:建立推荐系统模型
    //隐藏因子k的数量不宜过大,避免产生过大的计算代价
    val model=ALS.train(ratings,30,10,0.01)

    //第三步:为789号用户推荐10部电影
    val u1Result=model.recommendProducts(1,6)
    u1Result.foreach {println}


    //第四步:根据推荐结果+u.item文件(包含了电影信息),获取到推荐的电影名
    //整体思路:
    //①先通过spark读取u.item
    //②把文件数据变为一个Map(电影id,电影名)
    //③从推荐结果中,根据电影id获取电影名
    val bookData=sc.textFile("D://test/a/userbooks.txt")

    //collectAsMap方法:可以将RDD[(key,value)]->Map(key,value)
    val bookMap=bookData.map {line =>
      val info=line.split("\\|")
      val bookId=info(0).toInt
      val bookName=info(1)
      (bookId,bookName)
    }.collectAsMap
    //println(movieMap(345))

    val u1ResultParse=u1Result.map { rat =>
      //获取用户id
      val userId=rat.user
      //获取商品id
      val bookId=rat.product
      val bookName=bookMap(bookId)//？？？？movieid(0)
      //获取评分
      val score=rat.rating

      (userId,bookName,score)
    }

    u1ResultParse.foreach{println}
    //第五步:检验模型推荐的效果,本例中采用直观检验法。
    //处理思路：
    //第1步:先获取789号用户看过的所有电影
    //第2步:再获取789号用户最喜欢的前10部电影
    //第2步:用推荐的10部和他最喜欢的10部电影比对,看是否有相似的电影

    //keyBy的作用:以指定的规则为key来进行查找,下面代码表示以用户id为key进行查找
    //lookup的作用:传入具体查找的key
    //keyBy和lookup一般是成对出现的
    val u1books=ratings.keyBy { x => x.user }.lookup(1)

    //完成第二步,最后返回的数据RDD[(userId,movieName,score)]

    val u1Top6=u1books.sortBy{x=> -x.rating}.take(6)
      .map { x =>(x.user,bookMap(x.product),x.rating) }
    u1Top6.foreach{println}
    //第六步:模型的存储,为了避免每次推荐时都重新训练模型,使用时仅需加载模型使用即可
    /////model.save(sc, "hdfs://hadoop01:9000/rec-result-1909")


  }
}
