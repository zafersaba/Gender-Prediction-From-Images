# read data into memory
images <- read.csv("hw01_images.csv", header=FALSE)
labels <- read.csv("hw01_labels.csv", header=FALSE)

#implement safelog. It is used because normal log() function may have infinity values.
safelog <- function(x) {
  return (log(x + 1e-100))
}

#training data size
K=200

# split into train and test
images_train <- images[1:K,]
images_test <- images[(K+1):400,]
labels_train <- labels[1:K,]
labels_test <- labels[(K+1):400,]

# calculate sample means
means <- sapply(X= 1:2, FUN = function(x) {sapply(X= 1:4096, FUN = function(c) {mean(images_train[labels_train == x,c])})})

# calculate sample deviations
deviations <- sapply(X= 1:2, FUN = function(x) {sapply(X= 1:4096, FUN = function(c) {sqrt(mean((images_train[labels_train == x,c] - means[c,x])^2))})})

# calculate prior probabilities
class_priors <- sapply(X = 1:2, FUN = function(c) {mean(labels_train == c)})

#calculations needed for the score function
cov_diag1 <- diag((deviations[,1])^2)
cov_diag2 <- diag((deviations[,2])^2)             

W1 <- -0.5 * chol2inv(chol(cov_diag1))
w1 <- chol2inv(chol(cov_diag1)) %*% means[,1]
w10 <- -0.5 * t(means[,1]) %*% chol2inv(chol(cov_diag1)) %*% means[,1] -0.5*safelog(det(cov_diag1)) + safelog(class_priors[1])

W2 <- -0.5 * chol2inv(chol(cov_diag2))
w2 <- chol2inv(chol(cov_diag2)) %*% means[,2]
w20 <- -0.5 * t(means[,2]) %*% chol2inv(chol(cov_diag2)) %*% means[,2] -0.5*safelog(det(cov_diag2)) + safelog(class_priors[2])

#calculate the score functions
g1 <- sapply(X=1:200, FUN=function(c) {as.matrix(images_train[c,])%*% W1 %*% t(images_train[c,]) + t(w1) %*% t(images_train[c,]) +w10})   
g2 <- sapply(X=1:200, FUN=function(c) {as.matrix(images_train[c,])%*% W2 %*% t(images_train[c,]) + t(w2) %*% t(images_train[c,]) +w20})

#create the prediction matrix
prediction <- matrix (1:200, nrow=200, ncol=1)

#compare the score functions and modify the prediction 
for(x in 1:200){
  if (g1[x] > g2[x]){
    prediction[x] <- 1
  }
  else {
    prediction[x] <- 2
    }
}

training_confusion_matrix <- table(labels_train,prediction)

#repeat calculations for test data

g1_test <- sapply(X=1:200, FUN=function(c) {as.matrix(images_test[c,])%*% W1 %*% t(images_test[c,]) + t(w1) %*% t(images_test[c,]) +w10})   
g2_test <- sapply(X=1:200, FUN=function(c) {as.matrix(images_test[c,])%*% W2 %*% t(images_test[c,]) + t(w2) %*% t(images_test[c,]) +w20})

prediction_test <- matrix (1:200, nrow=200, ncol=1)

for(x in 1:200){
  if (g1_test[x] > g2_test[x]){
    prediction_test[x] <- 1
  }
  else {
    prediction_test[x] <- 2
  }
}

test_confusion_matrix <- table(labels_test,prediction_test)
