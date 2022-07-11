import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import time
start_time = time.time()

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)


print("Step 1 : Loading Movies names ...")
nameDict = loadMovieNames()
print("\n DONE \n")
print("\n Step 2 : Loading  The ratings Data ... \n")
data = sc.textFile("file:///SparkCourse/ml-100k/u.data")
print("\n DONE \n")

# Map ratings to key / value pairs: user ID => movie ID, rating

# on fait le Mapping de chaque note d'utilisateur à une entité Key , value
# où Key = userId et Value = ( movieId , rating )
print("\n Step 3 : le Mapping de chaque note d'utilisateur à une entité Key , value "
      "\noù Key = userId et Value = ( movieId , rating ) ...\n")

ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))
print("\nDONE\n")
# Emit every movie rated together by the same user.
# Self-join to find every combination.
print("\n Step 4 : Self-join pour obtenir tous les combinaisons possibles ( deux à deux ) de films notés par le même utilisateur\n")
joinedRatings = ratings.join(ratings)
print("\nDONE\n")
# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))
print("\n Step 5 : Filtrer pour se débarasser de duplications  \n")

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
print("\n Done \n")

print("\n Step 6 : Mapping pour rendre le Key = (movie1,movie2)  \n")

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)
print("\n DONE \n")

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
print("\n Step 7 : Appeler groupByKey afin d'avoir des groupements sous forme \n"
      "movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...")
moviePairRatings = moviePairs.groupByKey()
print("\n DONE \n")

print("\n Step 8 : Calculer le score de similarité pour chaque couple de films et on stocke le resulat sur le cache ")

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()
print("\n DONE ")


# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".

scoreThreshold = 0.97
coOccurenceThreshold = 50
print("\n Step 9 : Demander à l'utilisateur d'introduire le Movie ID ")

movieID = int(input("Enter movie ID: "))
print("\n Done ")

# Filter for movies with this sim that are "good" as defined by
# our quality thresholds above
print("\n Step 10 : Filtrer pour obtenir les scores de similarité uniquement pour les couples où MovieId fait partie de (movie1 , movie2)  \n"
      "+ Le respect de contraintes : ScoreThreshold > ", scoreThreshold , " et coOccurenceThreshold > ",coOccurenceThreshold )

filteredResults = moviePairSimilarities.filter(lambda pairSim: \
    (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
    and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)
print("\n DONE \n")

print("\n Step 11 : Ordonner les resultat obtenues ")

# Sort by quality score.
results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)
print("\nDONE \n")

print("\n Step 12 : Display the results")

print("Top 10 similar movies for " + nameDict[movieID])
for result in results:
    (sim, pair) = result
    # Display the similarity result that isn't the movie we're looking at
    similarMovieID = pair[0]
    if (similarMovieID == movieID):
        similarMovieID = pair[1]
    print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
print("--- %s seconds ---" % (time.time() - start_time))
print("\nDONE \n")
