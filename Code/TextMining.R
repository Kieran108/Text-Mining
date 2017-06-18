###############################################
# Text Mining - Kieran Kalair - 29th March 2017
###############################################

####################################
# Helper Functions
####################################
# Helper function to evaluate the accuracy, precision, recall and F_1 score of classifier
evaluateClassifierQuality <- function( predictedLabels, trueLabels ){
  # This function takes in a vector predictedLabels and a vector trueLabels
  # and calculates the accuracy, precision, recall and F_1 score of the corrosponding
  # classifier. Inputs must be the same length. 
  
  # We say 1 is positive class, 0 is negative
  # First, we work out true positives (TP), true negatives (TN), false positives (FP) and false neagtaives (FN)
  TP <- 0 
  TN <- 0
  FP <- 0
  FN <- 0
  # Loop over all predictions we have made
  for( evalCounter in 1:length(trueLabels) ){
    # If it's positive, and we predicted it positive, we update TP
    if( trueLabels[evalCounter] == 1 ){
      if( predictedLabels[evalCounter] == 1 ){
        TP <- TP + 1
        # Otherwise, we must have predicted false when it should be true, so update FN
      }else{
        FN <- FN + 1
      }
      # If we are here, we must have a true label of 0
    }else{
      # If we predict 0, thats a match, so a TN
      if( predictedLabels[evalCounter] == 0 ){
        TN <- TN + 1
        # Otherwise, we must be predicting true, so thats a false pos
      }else{
        FP <- FP + 1
      }
    }
  }
  # We now know all FP, FN, TP, TN
  # Accuracy = (TP+TN)/|D|
  accuracy <- (TP+TN)/length(trueLabels)
  if( is.nan(accuracy) == 1 ){accuracy<-0}
  # Recall = TP/(TP+FN)
  recall <- TP/(TP+FN)
  if( is.nan(recall) == 1 ){recall<-0}
  # Precision = TP/(TP+FP)
  precision <- TP/(TP+FP)
  if( is.nan(precision) == 1 ){precision<-0}
  # F_1 Score = (2*Precision*Recall)/(Precision+Recall)
  F1 <- (2*precision*recall)/(precision+recall)
  if( is.nan(F1) == 1 ){F1<-0}
  
  # Return results in a list
  results <- list( accuracy, precision, recall, F1 )
  # Name the elements to avoid confusion
  names(results) <- c("Accuracy", "Precision", "Recall", "F1Score")
  return(results)
}

# Helper function to calculate the macro accuracy, precision, recall, and F_1 score
calcMacroAverages <- function( listOfListOfVals ){
  # This function takes a list, where each element is an output of the function
  # evaluateClassifierQuality() (so is in it's self, a list) and calcualtes the 
  # macro-average accuracy, precision, recall and F_1 score.
  
  accuracyList <- list()
  precisionList <- list()
  recallList <- list()
  F1List <- list()
  # Loop over the lists in the larger list
  for( macroCounter in 1:length(listOfListOfVals) ){
    currentSubList <- listOfListOfVals[[macroCounter]]
    accuracyList[[macroCounter]] <- currentSubList$Accuracy
    precisionList[[macroCounter]] <- currentSubList$Precision
    recallList[[macroCounter]] <- currentSubList$Recall
    F1List[[macroCounter]] <- currentSubList$F1Score
  }
  
  # Compute averages
  macAveAcc <- Reduce("+",accuracyList)/length(accuracyList)
  macAvePre <- Reduce("+",precisionList)/length(precisionList)
  macAveRec <- Reduce("+",recallList)/length(recallList)
  macAveF1 <- Reduce("+",F1List)/length(F1List)
  
  # Put results into a list
  results <- list( macAveAcc, macAvePre, macAveRec, macAveF1 )  
  # Name to avoid confusion
  names(results) <- c("Accuracy", "Precision", "Recall", "F1Score")
  return(results)
}

##############################################################
####################
# Importing the data
####################
rawData <- read.csv( "reutersCSV.csv" )
# Look at information about the data
#str(rawData)
#colnames(rawData)
# Identify any missing data
rawData <- na.omit(rawData)
# We want the 10 most popular classes - earn,	acquisitions,	money-fx,	grain,	crude,	trade, interest,	ship,	wheat,	corn)
topics <- c( "topic.earn","topic.acq", "topic.money.fx", "topic.grain", "topic.crude", "topic.trade","topic.interest","topic.ship","topic.wheat","topic.corn" )
wantCols <- c( "pid", "fileName", "purpose", topics, "doc.title","doc.text" )
# Get only the bits of data we want - first we get rid of the extra columns
textData <- rawData[,wantCols]
topicsData <- textData[,topics]
# Now we get rid of any rows that aren't about the topics we want
textData <- textData[which(rowSums(topicsData) > 0),]
# Check for missing data - this sum is 0 so there is none
missingCheck <- sum(is.na(textData))
# Check none of the doc.text is blank
textData <- textData[textData$doc.text!="",]
###############################################################
# textData is now a data frame with only the info we want in it
###############################################################
# Uncomment these if you need to install something.
# NOTE: rJava is required for openNLP but there is a known issuie installing this, so if you are on ubuntu, type """ sudo apt-get r-cran-java """ into the terminal to do it manually 
# install.packages('tm')
# install.packages('stringr')
# install.packages('tau')
# install.packages('e1071')
# install.packages('randomForest')
# install.packages('openNLP')
library(tm)                        # For bulk of pre-processing 
library(stringr)                   # For string manipulation 
library(tau)                       # Helps with n-gram extraction
library(e1071)                     # Has lots of classifier functions
library(randomForest)              # Has the algorithm we use for random forest 
library(openNLP)                   # Used with tm

######################
# Pre Processing stage
######################
# Before any pre-processing, we have text like 
#textData$doc.text[1]

# Remove non-graphical characters
textData$doc.text <- str_replace_all(textData$doc.text,"[^[:graph:]]", " ") 

# Stemming of words
for( counter in 1:nrow(textData) ){
  textData$doc.text[counter] <- stemDocument( textData$doc.text[counter], language="english" )
}
textData$doc.text[counter]
# Create a corpus object to make pre-processing easy
docs <- Corpus(VectorSource(textData$doc.tex))   
# Remove punctuation, numbers, capital letters and stop words
docs <- tm_map(docs, removePunctuation)   
docs <- tm_map(docs, removeNumbers)   
docs <- tm_map(docs, tolower)   
docs <- tm_map(docs, removeWords, stopwords("english"))

# We should now have a data frame with no stop words, punctuation, numbers and all lower case 
inspect(docs[1])
# End of pre-processing
###################
# Feature Selection
###################
##########
# Unigrams
##########
# Create document term matrix from our corpus
# We tried both frequency and td-idf but found frequency was as good generally (why line is commented)
#dtm <- DocumentTermMatrix(docs, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)) )
dtm <- DocumentTermMatrix(docs)
# We have too many features, so apply thresholding - remove very sparse terms
dtmRemoved <- removeSparseTerms(dtm, 0.99)

# We now have our set of features, so we find their presence in our documents
# Convert DTM to R type matrix - each row is a document, each col is a term that we have kept as a feature
featureValsAsMat <- as.matrix(dtmRemoved)
# This is how many unigram features we have
#dim(featureValsAsMat)

# Find how many pos and negatives we have for each class (entire data) - This is just for a comment in the report we want to make
# PosNegList <- list()
# for( counter in 1:length(topics) ){
#   PosNegList[[counter]] <- c( length(which(textData[,3+counter]==0)),length(which(textData[,3+counter]==1))  )
#   print(sum(length(which(textData[,3+counter]==0)),length(which(textData[,3+counter]==1))))
# }
# names(PosNegList) <- topics

##########
# Bigrams
##########
# Generate bigrams
bigrams = textcnt(docs, n = 2, method = "string")
# Order them by frequency
bigrams = bigrams[order(bigrams, decreasing = TRUE)]
# Keep the 40 most common
bigrams <- bigrams[1:40]
# Name the entries in bigrams with what the actual phrases
bigramText <- names(bigrams)
# A working matrix to fill in as we go, we will join this to the other features
myMatrix <- matrix(0, nrow=length(docs), ncol=length(bigrams))
colnames(myMatrix) <- bigramText
# Loop over documents, record bigram frequencies
for( counter in 1:length(docs) ){
  currentDoc <- textData$doc.text[counter]
  currentBigramsWithCounts <- textcnt(currentDoc, n = 2, method = "string")
  for( counter2 in 1:length(currentBigramsWithCounts) ){
    if( names(currentBigramsWithCounts)[counter2] %in% colnames(myMatrix) ){
      myMatrix[counter, grep( names(currentBigramsWithCounts)[counter2], colnames(myMatrix))] <- currentBigramsWithCounts[counter2]
    }
  }
}
# Append these new features to the features matrix
featureValsAsMat <- cbind(featureValsAsMat, myMatrix)


########################################################################################################
############################### Feature Selection has now been completed ############################### 
########################################################################################################

###################################
# Split Data - training and testing
###################################
numTrain <-length(which(textData$purpose=="train"))                        # Number of training instances
numTest <- nrow(textData) - length(which(textData$purpose=="train"))       # Number of test instances  
endTrain <- length(which(textData$purpose=="train"))                       # The index of the last training instance
endTest <- nrow(textData)                                                  # The index of the last test instance

# Split into training and testing
# Each row is a document, each column is a feature
trainData <- featureValsAsMat[1:endTrain,]
trainLabels <- textData[1:endTrain,topics]
trainSet <- cbind( trainData, trainLabels )
testData <- featureValsAsMat[-(1:endTrain),]
testLabels <- textData[-(1:endTrain),topics]

# We want to use 10-fold cross validation 
# Randomly shuffle the data - the set so the features AND the labels get done together and match  
trainSet <- trainSet[sample(nrow(trainSet)),]
# Re-assign now it's shuffled
trainData <- trainSet[ , 1:ncol(featureValsAsMat)]
trainLabels <- trainSet[ , -(1:ncol(featureValsAsMat))]
# Create 10 equally size folds
folds <- cut(seq(1,nrow(trainSet)),breaks=10,labels=FALSE)


#######################
# Naive Bayes Base Line
#######################

##############################################################################################################
# Please Note - it takes a long time to loop over all classes and all CV folds, if you test this I reccomend #
# to chane the for loop to only do a few classes (change length(topicsColIndex) to just be 1 or 2 etc...)    #
##############################################################################################################

# To store evaluation metrics in 
NBMetrics <- list()
workingList <- list()
# Input to classifiers
InputToClassifData <- cbind(trainData,trainLabels)
# These are the columns in our featuers matrix that we put the labels in
topicsColIndex <- seq( ncol(featureValsAsMat)+1, ncol(InputToClassifData), by=1 )
# This will create the accuracy table we see in the report
storedFoldsNB <- matrix(, nrow = 10, ncol=10)
storedFoldsNB <- as.data.frame(storedFoldsNB)
storedFoldsNB <- rbind(storedFoldsNB, rep(NA, 10))
rownames(storedFoldsNB) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average")
colnames(storedFoldsNB) <- topics
# Classifier per topic - loop over each topic
for( counter in 1:length(topicsColIndex) ){
  # This gives some visual feedback for how far in we are
  print(counter)
  # And loop over folds for cross-validation
  for( i in 1:10 ){
    # The current columns we want
    currentCols <- c( 1:ncol(featureValsAsMat), topicsColIndex[counter] )
    # Make current data the features + class col we care about
    currentInputData <- InputToClassifData[,currentCols]
    colnames(currentInputData)[ncol(currentInputData)] <- "CurrentLabel"
    # Convert to factor to make comparision easier with output of predict
    currentInputData$CurrentLabel <- as.factor(currentInputData$CurrentLabel)
    # Find out which data points are in what fold
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # Get 9 training folds
    trainFolds <- currentInputData[-testIndexes, ]
    testFolds <- currentInputData[testIndexes, ]
    # Input Data into classifier 
    NBModel <- naiveBayes(CurrentLabel~., data=trainFolds, laplace=TRUE)
    # Make predictions from it  
    modelResults <- predict(NBModel, newdata=testFolds[,-ncol(testFolds)])
    # Evaluate the results, put them in a running list
    workingList[[i]] <- evaluateClassifierQuality( modelResults,testFolds[,ncol(testFolds)] )
    
    # Find out how many are correct
    numRight <- 0
    for( corrCounter in 1:length(modelResults)){
      if( modelResults[corrCounter] == testFolds[corrCounter, ncol(testFolds)] ){
        numRight <- numRight + 1
      }
    }
    percentRight <- 100*numRight/length(modelResults) 
    # This offers some visual feedback as it's running
    print(percentRight)
    # Store value of accuracy in matrix
    storedFoldsNB[ i, counter ] <- percentRight
  }
  # Find averages for this class
  NBMetrics[[counter]] <- calcMacroAverages( workingList )
}

# Fill in accuracy averages
storedFoldsNB[11,] <- colMeans(storedFoldsNB[-11,])
# Work out standard deviations
stdVals <- rep(NA, 10)
for( counter in 1:10 ){
  stdVals[counter] <- sd(storedFoldsNB[1:10,counter])
}
storedFoldsNB <- rbind(storedFoldsNB, stdVals)
rownames(storedFoldsNB) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average", "Standard Deviation")

# Work out macro average of scores - display it to screen so we can write it in report
calcMacroAverages(NBMetrics)

# Running the data on the entire training dataset then seeing how it runs on train
NBOverall <- list()
# Run NB on entire test + train 
trainData <- featureValsAsMat[1:endTrain,]
trainLabels <- textData[1:endTrain,topics]
testData <- featureValsAsMat[-(1:endTrain),]
testLabels <- textData[-(1:endTrain),topics]
# Loop over each topic
for( counter in 1:length(topics) ){
  # Visual update of where we are
  print(counter)
  # Split into two parts, make into data frame
  myTrainData <- cbind(trainData, trainLabels[,counter])
  myTrainData <- as.data.frame(myTrainData)
  myTestData <- cbind(testData, testLabels[,counter])
  myTestData <- as.data.frame(myTestData)
  # Name columns, make sure 1 and 0's are factors
  colnames(myTrainData)[ncol(myTrainData)] <- "CurrentLabel"
  colnames(myTestData)[ncol(myTestData)] <- "CurrentLabel"
  myTrainData$CurrentLabel <- as.factor(myTrainData$CurrentLabel)
  myTestData$CurrentLabel <- as.factor(myTestData$CurrentLabel)
  # Input Data into classifier 
  NBModel <- naiveBayes(CurrentLabel~., data=myTrainData,laplace=TRUE)
  # Make predictions from it  
  modelResults <- predict(NBModel, newdata=myTestData[,-ncol(myTestData)])
  # Evaluate the results
  NBOverall[[counter]] <- evaluateClassifierQuality( modelResults, myTestData[,ncol(myTestData)] )
}
# Print Results
NBOverall
# Average
NBOverallAver <- calcMacroAverages(NBOverall)


################################
# SVM Classifier - Linear kernel
################################

##############################################################################################################
# Please Note - it takes a long time to loop over all classes and all CV folds, if you test this I reccomend #
# to chane the for loop to only do a few classes (change length(topicsColIndex) to just be 1 or 2 etc...)    #
##############################################################################################################

# To store evaluation metrics in 
SVMMetrics <- list()
workingList <- list()

# Input to classifiers
InputToClassifData <- cbind(trainData,trainLabels)
# These are the columns in our featuers matrix that we put the labels in
topicsColIndex <- seq( ncol(featureValsAsMat)+1, ncol(InputToClassifData), by=1 )
storedFoldsSVM <- matrix(, nrow = 10, ncol=10)
storedFoldsSVM <- as.data.frame(storedFoldsSVM)
storedFoldsSVM <- rbind(storedFoldsSVM, rep(NA, 10))
colnames(storedFoldsSVM) <- topics
# Classifier per topic
for( counter in 1:length(topicsColIndex) ){
  for( i in 1:10 ){
    # The current columns we want
    currentCols <- c( 1:ncol(featureValsAsMat), topicsColIndex[counter] )
    # Make current data the features + class col we care about
    currentInputData <- InputToClassifData[,currentCols]
    colnames(currentInputData)[ncol(currentInputData)] <- "CurrentLabel"
    # Convert to factor to make comparision easier with output of predict
    currentInputData$CurrentLabel <- as.factor(currentInputData$CurrentLabel)
    # Find out which data points are in what fold
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # Get 9 training folds
    trainFolds <- currentInputData[-testIndexes, ]
    testFolds <- currentInputData[testIndexes, ]
    # Input Data into classifier - We varied this cost parameter during training and found 1 worked very well
    SVMModel <- svm(CurrentLabel~., data=currentInputData, kernel='linear', cost=1)
    # Make predictions from it  
    modelResults <- predict(SVMModel, newdata=testFolds[,-ncol(testFolds)])
    # Evaluate the results
    workingList[[i]] <- evaluateClassifierQuality( modelResults,testFolds[,ncol(testFolds)] )
    # Find out how many are correct
    numRight <- 0
    for( corrCounter in 1:length(modelResults)){
      if( modelResults[corrCounter] == testFolds[corrCounter, ncol(testFolds)] ){
        numRight <- numRight + 1
      }
    }
    percentRight <- 100*numRight/length(modelResults) 
    print(percentRight)
    # Store value
    storedFoldsSVM[ i, counter ] <- percentRight
  }
  SVMMetrics[[counter]] <- calcMacroAverages( workingList )
}
# Fill in averages
storedFoldsSVM[11,] <- colMeans(storedFoldsSVM[-11,])
# Work out standard deviations
stdVals <- rep(NA, 10)
for( counter in 1:10 ){
  stdVals[counter] <- sd(storedFoldsSVM[1:10,counter])
}
storedFoldsSVM <- rbind(storedFoldsSVM, stdVals)
rownames(storedFoldsSVM) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average", "Standard Deviation")


# Work out macro average of scores
calcMacroAverages(SVMMetrics)

SVMOverall <- list()
# Run SVM on entire test + train 
trainData <- featureValsAsMat[1:endTrain,]
trainLabels <- textData[1:endTrain,topics]
testData <- featureValsAsMat[-(1:endTrain),]
testLabels <- textData[-(1:endTrain),topics]
for( counter in 1:length(topics) ){
  print(counter)
  # Split into two parts, make into data frame
  myTrainData <- cbind(trainData, trainLabels[,counter])
  myTrainData <- as.data.frame(myTrainData)
  myTestData <- cbind(testData, testLabels[,counter])
  myTestData <- as.data.frame(myTestData)
  # Name columns, make sure 1 and 0's are factors
  colnames(myTrainData)[ncol(myTrainData)] <- "CurrentLabel"
  colnames(myTestData)[ncol(myTestData)] <- "CurrentLabel"
  myTrainData$CurrentLabel <- as.factor(myTrainData$CurrentLabel)
  myTestData$CurrentLabel <- as.factor(myTestData$CurrentLabel)
  # Input Data into classifier 
  SVMModel <- svm(CurrentLabel~., data=myTrainData, kernel="linear", cost = 1)
  # Make predictions from it  
  modelResults <- predict(SVMModel, newdata=myTestData[,-ncol(myTestData)])
  # Evaluate the results
  SVMOverall[[counter]] <- evaluateClassifierQuality( modelResults, myTestData[,ncol(myTestData)] )
}

# Print Results
SVMOverall
# Average
SVMOverallAver <- calcMacroAverages(SVMOverall)


#############################
# SVM Classifier - RBF kernel
#############################

##############################################################################################################
# Please Note - it takes a long time to loop over all classes and all CV folds, if you test this I reccomend #
# to chane the for loop to only do a few classes (change length(topicsColIndex) to just be 1 or 2 etc...)    #
##############################################################################################################

# To store evaluation metrics in 
SVMRBFMetrics <- list()
workingList <- list()

# Input to classifiers
InputToClassifData <- cbind(trainData,trainLabels)
# These are the columns in our featuers matrix that we put the labels in
topicsColIndex <- seq( ncol(featureValsAsMat)+1, ncol(InputToClassifData), by=1 )
storedFoldsSVMRBF <- matrix(, nrow = 10, ncol=10)
storedFoldsSVMRBF <- as.data.frame(storedFoldsSVMRBF)
storedFoldsSVMRBF <- rbind(storedFoldsSVMRBF, rep(NA, 10))
colnames(storedFoldsSVMRBF) <- topics

# Classifier per topic
for( counter in 1:length(topicsColIndex) ){
  print(counter)
  for( i in 1:10 ){
    # The current columns we want
    currentCols <- c( 1:ncol(featureValsAsMat), topicsColIndex[counter] )
    # Make current data the features + class col we care about
    currentInputData <- InputToClassifData[,currentCols]
    colnames(currentInputData)[ncol(currentInputData)] <- "CurrentLabel"
    # Convert to factor to make comparision easier with output of predict
    currentInputData$CurrentLabel <- as.factor(currentInputData$CurrentLabel)
    # Find out which data points are in what fold
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # Get 9 training folds
    trainFolds <- currentInputData[-testIndexes, ]
    testFolds <- currentInputData[testIndexes, ]
    
    # Input Data into classifier - by default we have a RBF kernel
    SVMModel <- svm(CurrentLabel~., data=currentInputData)
    # Make predictions from it  
    modelResults <- predict(SVMModel, newdata=testFolds[,-ncol(testFolds)])
    # Evaluate the results
    workingList[[i]] <- evaluateClassifierQuality( modelResults,testFolds[,ncol(testFolds)] )
    
    # Find out how many are correct
    numRight <- 0
    for( corrCounter in 1:length(modelResults)){
      if( modelResults[corrCounter] == testFolds[corrCounter, ncol(testFolds)] ){
        numRight <- numRight + 1
      }
    }
    percentRight <- 100*numRight/length(modelResults) 
    print(percentRight)
    
    # Store value
    storedFoldsSVMRBF[ i, counter ] <- percentRight
  }
  SVMRBFMetrics[[counter]] <- calcMacroAverages( workingList )
}
# Fill in averages
storedFoldsSVMRBF[11,] <- colMeans(storedFoldsSVMRBF[-11,])
# Work out standard deviations
stdVals <- rep(NA, 10)
for( counter in 1:10 ){
  stdVals[counter] <- sd(storedFoldsSVMRBF[1:10,counter])
}
storedFoldsSVMRBF <- rbind(storedFoldsSVMRBF, stdVals)
rownames(storedFoldsSVMRBF) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average", "Standard Deviation")

# Work out macro average of scores
calcMacroAverages(SVMRBFMetrics)

SVMOverallRBF <- list()
# Run SVM on entire test + train 
trainData <- featureValsAsMat[1:endTrain,]
trainLabels <- textData[1:endTrain,topics]
testData <- featureValsAsMat[-(1:endTrain),]
testLabels <- textData[-(1:endTrain),topics]
for( counter in 1:length(topics) ){
  print(counter)
  # Split into two parts, make into data frame
  myTrainData <- cbind(trainData, trainLabels[,counter])
  myTrainData <- as.data.frame(myTrainData)
  myTestData <- cbind(testData, testLabels[,counter])
  myTestData <- as.data.frame(myTestData)
  # Name columns, make sure 1 and 0's are factors
  colnames(myTrainData)[ncol(myTrainData)] <- "CurrentLabel"
  colnames(myTestData)[ncol(myTestData)] <- "CurrentLabel"
  myTrainData$CurrentLabel <- as.factor(myTrainData$CurrentLabel)
  myTestData$CurrentLabel <- as.factor(myTestData$CurrentLabel)
  # Input Data into classifier 
  SVMModel <- svm(CurrentLabel~., data=myTrainData)
  # Make predictions from it  
  modelResults <- predict(SVMModel, newdata=myTestData[,-ncol(myTestData)])
  # Evaluate the results
  SVMOverallRBF[[counter]] <- evaluateClassifierQuality( modelResults, myTestData[,ncol(myTestData)] )
}

# Print Results
SVMOverallRBF
# Average
SVMOverallAver <- calcMacroAverages(SVMOverallRBF)


##############################
# Random Forest Classification
##############################

##############################################################################################################
# Please Note - it takes a long time to loop over all classes and all CV folds, if you test this I reccomend #
# to chane the for loop to only do a few classes (change length(topicsColIndex) to just be 1 or 2 etc...)    #
##############################################################################################################

# To store evaluation metrics in 
RFMetrics <- list()
workingList <- list()

# Input to classifiers
InputToClassifData <- cbind(trainData,trainLabels)
# These are the columns in our featuers matrix that we put the labels in
topicsColIndex <- seq( ncol(featureValsAsMat)+1, ncol(InputToClassifData), by=1 )
storedFoldsRF <- matrix(, nrow = 10, ncol=10)
storedFoldsRF <- as.data.frame(storedFoldsRF)
storedFoldsRF <- rbind(storedFoldsRF, rep(NA, 10))
rownames(storedFoldsRF) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average")
colnames(storedFoldsRF) <- topics
# Classifier per topic
for( counter in 1:length(topicsColIndex) ){
  print(counter)
  for( i in 1:10 ){
    # The current columns we want
    currentCols <- c( 1:ncol(featureValsAsMat), topicsColIndex[counter] )
    # Make current data the features + class col we care about
    currentInputData <- InputToClassifData[,currentCols]
    colnames(currentInputData)[ncol(currentInputData)] <- "CurrentLabel"
    # Convert to factor to make comparision easier with output of predict
    currentInputData$CurrentLabel <- as.factor(currentInputData$CurrentLabel)
    # Find out which data points are in what fold
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # Get 9 training folds
    trainFolds <- currentInputData[-testIndexes, ]
    testFolds <- currentInputData[testIndexes, ]
    # Input Data into classifier 
    RFModel <- randomForest(x=trainFolds[,-ncol(trainFolds)], y=trainFolds[,ncol(trainFolds)], ntree=30)
    # Make predictions from it  
    modelResults <- predict(RFModel, testFolds[,-ncol(testFolds)])
    # Evaluate the results
    workingList[[i]] <- evaluateClassifierQuality( modelResults,testFolds[,ncol(testFolds)] )
    # Find out how many are correct
    numRight <- 0
    for( corrCounter in 1:length(modelResults)){
      if( modelResults[corrCounter] == testFolds[corrCounter, ncol(testFolds)] ){
        numRight <- numRight + 1
      }
    }
    percentRight <- 100*numRight/length(modelResults) 
    print(percentRight)
    # Store value
    storedFoldsRF[ i, counter ] <- percentRight
  }
  RFMetrics[[counter]] <- calcMacroAverages( workingList )
}
# Fill in averages
storedFoldsRF[11,] <- colMeans(storedFoldsRF[-11,])
# Work out standard deviations
stdVals <- rep(NA, 10)
for( counter in 1:10 ){
  stdVals[counter] <- sd(storedFoldsRF[1:10,counter])
}
storedFoldsRF <- rbind(storedFoldsRF, stdVals)
rownames(storedFoldsRF) <- c( "CV Fold 1","CV Fold 2" ,"CV Fold 3" ,"CV Fold 4" ,"CV Fold 5" ,"CV Fold 6" ,"CV Fold 7" ,"CV Fold 8", "CV Fold 9" ,"CV Fold 10", "Average", "Standard Deviation")

# Work out macro average of scores
calcMacroAverages(RFMetrics)

RFOverall <- list()
# Run SVM on entire test + train 
trainData <- featureValsAsMat[1:endTrain,]
trainLabels <- textData[1:endTrain,topics]
testData <- featureValsAsMat[-(1:endTrain),]
testLabels <- textData[-(1:endTrain),topics]
for( counter in 1:length(topics) ){
  print(counter)
  # Split into two parts, make into data frame
  myTrainData <- cbind(trainData, trainLabels[,counter])
  myTrainData <- as.data.frame(myTrainData)
  myTestData <- cbind(testData, testLabels[,counter])
  myTestData <- as.data.frame(myTestData)
  # Name columns, make sure 1 and 0's are factors
  colnames(myTrainData)[ncol(myTrainData)] <- "CurrentLabel"
  colnames(myTestData)[ncol(myTestData)] <- "CurrentLabel"
  myTrainData$CurrentLabel <- as.factor(myTrainData$CurrentLabel)
  myTestData$CurrentLabel <- as.factor(myTestData$CurrentLabel)
  # Input Data into classifier 
  RFModel <- randomForest(x=myTrainData[,-ncol(myTrainData)], y=myTrainData[,ncol(myTrainData)], ntree=30)
  # Make predictions from it  
  modelResults <- predict(RFModel, myTestData[,-ncol(myTestData)])
  # Evaluate the results
  RFOverall[[counter]] <- evaluateClassifierQuality( modelResults, myTestData[,ncol(myTestData)] )
}

# Print Results
RFOverall
# Average
RFOverallAver <- calcMacroAverages(RFOverall)
