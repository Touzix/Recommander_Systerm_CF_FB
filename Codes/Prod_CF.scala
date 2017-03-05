
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

/* Steps:
-importing data 
-construction tuples CF: (UserID, MovieID, Rating)
-construction tuples to identify movies ID: (MovieID, Title)
*/
val small_ratings_raw_data = sc.textFile("ratings.csv")

val small_ratings_data = small_ratings_raw_data.filter(!_.contains("userId")).map(line=>line.split(",")).map(l=>(l(0),l(1),l(2))).map(x=>(x._1.toInt, x._2.toInt,x._3.toDouble)).cache()

small_ratings_data.first

// We do the same for the other file:
val small_movies_raw_data = sc.textFile("movies.csv")

val small_movies_data = small_movies_raw_data.filter(!_.contains("movieId")).map(line=>line.split(",")).map(l=>(l(0),l(1))).map(x=>(x._1.toInt,x._2)).cache()

small_movies_data.count

/*
Collaboratif filtering with MLlib:
numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
rank is the number of latent factors in the model.
iterations is the number of iterations to run.
lambda specifies the regularization parameter in ALS.
implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.
*/

val splits = small_ratings_data.randomSplit(Array(6, 2, 2), seed=0L)//60%train, 20%validation, 20%test
val training_RDD =splits(0) //X,Y train matrix
val validation_RDD=splits(1)//X,Y validation matrix 
val test_RDD=splits(2)//X,Y test matrix

training_RDD.first._1 // pour manipuler les tuples

val validation_for_predict_RDD = validation_RDD.map(x=>(x._1.toInt, x._2.toInt))//contient X validation matrix
val test_for_predict_RDD = test_RDD.map(x=>(x._1, x._2))//contient X test matrix 

//construction du modele:
//import math
import org.apache.spark.mllib.recommendation.ALS
import scala.math

//param de modele:
val seed = 5L
val iterations = 10
val regularization_parameter = 0.1
val ranks = Array(4, 8, 12)

val tolerance = 0.02
val min_error: Double = Double.PositiveInfinity
val best_rank = -1
val best_iteration = -1
val ratings_train = training_RDD.map{case (user, item, rate) =>
    Rating(user, item, rate)
  } //xy train rating matrix
val ratings_valid = validation_RDD.map{case (user, item, rate) =>
    Rating(user, item, rate)
  } //xy validation matrix
val rating_test= test_RDD.map{case (user, item, rate) =>
    Rating(user, item, rate)
  } //xy test matrix 

var errors = Array(0.0, 0.0, 0.0)
var err = 0
for (rank <- ranks){
    val model= ALS.train(ratings_train, rank, iterations, regularization_parameter) //construction du modele 
    val predictions = 
      model.predict(ratings_valid.map{case Rating(u,p,r)=>(u,p)}).map { case Rating(user, product, rate) => 
        ((user, product), rate)
      }//prediction sur xy validation matrix Rating
    val ratesAndPreds = ratings_valid.map { case Rating(user, product, rate) => 
      ((user, product), rate) 
    }.join(predictions) //jointure de xy validation & xy prediction donne un truc de la forme (u1,i2),(4.01,4.2)    
    var error = ratesAndPreds.map { case ((user, product), (r1, r2)) => 
      (r1 - r2)*(r1-r2)
    }.mean()    
    errors(err) = error
    err = err +1
    
}

errors

// a recommander based on all our data set:
val seed = 5L
val iterations = 10
val regularization_parameter = 0.1
val ranks = Array(4, 8, 12,24)

val tolerance = 0.02
val min_error: Double = Double.PositiveInfinity
val best_rank = -1
val best_iteration = -1
val ratings = small_ratings_data.map{case (user, item, rate) =>
    Rating(user, item, rate)
  } //xy train rating matrix
 



var errors = Array(0.0, 0.0, 0.0,0.0)
var err = 0
for (rank <- ranks){
    val model= ALS.train(ratings, rank, iterations, regularization_parameter) //construction du modele 
    val predictions = 
      model.predict(ratings.map{case Rating(u,p,r)=>(u,p)}).map { case Rating(user, product, rate) => 
        ((user, product), rate)
      }//prediction sur xy validation matrix Rating
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) => 
      ((user, product), rate) 
    }.join(predictions) //jointure de xy validation & xy prediction donne un truc de la forme (u1,i2),(4.01,4.2)    
    var error = ratesAndPreds.map { case ((user, product), (r1, r2)) => 
      (r1 - r2)*(r1-r2)
    }.mean()    
    errors(err) = error
    err = err +1
    
}

errors //meilleur err est pour rank==24

val model_finale= ALS.train(ratings, 24, iterations, regularization_parameter)
model_finale

//recomandation for a user:
// we want to recommand movies with minimal number of rates:
def get_counts_and_averages(x:Int,y:Array[Double])={//prend Rating Matrix 
    val nratings = y.size
    var s=0.0
    for (elt<-y) {
    s=s+elt
    }
    s=s/nratings
    (x,(nratings, s))
    
}

val movie_ID_with_ratings_RDD = (ratings.map{case(Rating(u,i,r)) => (i,r)}.groupByKey().map(x=>(x._1,x._2.toArray)))//(it,array_of_rates)
val movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(x=> get_counts_and_averages(x._1,x._2))//(it1,(nbr_rate,mean_of_rate))
val movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(x=> (x._1,x._2._1))//(it,nbr_rate)

//Adding new user rating id==0 is not used in the data base
val new_user_ID=0
val new_user_ratings = Array(
     (0,260,4), // Star Wars (1977)
     (0,1,3), // Toy Story (1995)
     (0,16,3), // Casino (1995)
     (0,25,4), // Leaving Las Vegas (1995)
     (0,32,4), // Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,1), // Flintstones, The (1994)
     (0,379,1), // Timecop (1994)
     (0,296,3), // Pulp Fiction (1994)
     (0,858,5) , // Godfather, The (1972)
     (0,50,4) // Usual Suspects, The (1995)
    )
val new_user_ratings_RDD = sc.parallelize(new_user_ratings)//transform array to RDD


//we add this user to the data base:
val rating_new = ratings.union(new_user_ratings_RDD.map{case (user, item, rate) =>
    Rating(user.toInt, item.toInt, rate.toDouble)})


val new_ratings_model = ALS.train(rating_new, 4, iterations, regularization_parameter) //construction du modele best rank 24 


//getting top recommandations:
//we will start by getting un RDD wich contains all the movies that wasn't watched by the user:
val new_user_ratings_ids = new_user_ratings.map(x=>x._2)//get just ids of movies watched by the yser 
val new_user_unrated_movies_RDD = (small_movies_data.filter(l=>(!new_user_ratings_ids.contains(l._1)))).map(x=>(new_user_ID, x._1))
val new_user_recommendations_RDD = new_ratings_model.predict(new_user_unrated_movies_RDD).map { case Rating(user, product, rate) => 
        ((user, product), rate)}

new_user_recommendations_RDD.take(5)

val new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(x=>(x._1._2, x._2))//(iditem,rate)


new_user_recommendations_rating_RDD.first

val new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(small_movies_data).join(movie_rating_counts_RDD)
//new_user_recommendations_rating_RDD: for the user0: (itid,rate_prédit)
//small_movies_data:(idit,title)
//movie_rating_counts_RDD: (it,nbr_rate_déja_fait)
//(iditem,((rate,title),nbr_rate_deja_fait)))

new_user_recommendations_rating_title_and_count_RDD.take(3)

val new_user_recommendations_rating_title_and_count_RDD_format=new_user_recommendations_rating_title_and_count_RDD.map{ case (i,((r,t),c))=> (t,r,c)}

new_user_recommendations_rating_title_and_count_RDD_format.first

val top_movies = new_user_recommendations_rating_title_and_count_RDD_format.filter(x=>x._3>=25).map(x=>(x._1,x._2)).map(item => item.swap).sortByKey(false, 1).map(item => item.swap).take(5)

top_movies.take(5)

new_user_recommendations_rating_title_and_count_RDD.filter(x=>x._1==37739).map{case (i,((r,t),c)) =>(t,r)}.collect()

//Recommandation for a particular movie:
new_user_recommendations_rating_title_and_count_RDD.filter(x=>x._1==37739).map{case (i,((r,t),c)) =>(t,r)}.collect()


//pour chercher un film: .filter(x=>x._2._1._2=="Greatest Game Ever Played")

new_ratings_model.predict(0,260)//dans la base c'est 4! //c'est la prediction du notre us=0, it=260

new_ratings_model.recommendProducts(0,5)//meilleurs items pour user 0

new_ratings_model.productFeatures.first //variable latente d'item 1  //on peut trouver la similarité entre 2 items avec ça!
 

new_ratings_model.userFeatures.first //variable latente d'utilisateur 0 //on peut trouver la similarité entre 2 utilisateurs avec ça!

