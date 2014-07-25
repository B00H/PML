---
title: 'Human Activity Recognition: Predicting exercise style in humans'
output: html_document
---

## Human Activity Recognition: Predicting exercise style in humans

### Summary
This report details a machine learning algorithm that was applied to a [Weight Lifting Exercises Dataset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). Detailed information on the dataset can be found [here](http://groupware.les.inf.puc-rio.br/har). Briefly, six healthy males aged 20 to 28 performed one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl either correctly (coded as Class A) or incorrectly (i.e., throwing the elbows to the front - coded as Class B, lifting the dumbbell only halfway - coded as Class C, lowering the dumbbell only halfway - coded as Class D), or throwing the hips to the front - coded as Class E), with sensors fitted to the forearm, arm, belt, and dumbbell. 

Computer specifics: R version 3.0.3 (2014-03-06) on darwin10.8.0 

Analysis date: Thu Jul 24 15:55:01 2014

### Data Preprocessing
Load required  packages. 

```r
library(caret); library(randomForest); library(Hmisc); library(knitr); library(markdown)
```


Get the data. 

```r
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./pml-training.csv")) {
        download.file(trainURL, destfile = "./pml_training.csv", method = "curl")
        train_download <- date()
}
if (!file.exists("./pml-testing.csv")) {
        download.file(testURL, destfile = "./pml_testing.csv", method = "curl")
        test_download <- date()
}
train_raw <- read.csv("./pml_training.csv", na.strings = c("", "NA"))
test_raw <- read.csv("./pml_testing.csv")
```

The training dataset was downloaded on Thu Jul 24 15:55:16 2014, and the test dataset was downloaded on Thu Jul 24 15:55:17 2014. 

The dataset `train_raw` will be split into a training (70%) and a cross-validation testing set (30%). 


```r
set.seed(34)
in_train <- createDataPartition(y = train_raw$classe, p = 0.70, list = FALSE)
train <- train_raw[in_train, ]
crossval <- train_raw[-in_train, ]
```

Exploring `train` and `test_raw` using `dim()` and `str()` (output not shown) suggests that not all 160 variables will be needed for the prediction algorithm (e.g., summary statistics, participant identifiers, timestamps) and cross-validation and subsequently removed from both `train` and `test_raw`.


```r
dim(train); str(train)
dim(test_raw); str(test_raw)
remove <- grep("X|user_name|raw_timestamp_part_1|raw_timestamp_part_2|cvtd_timestamp|new_window|num_window|^kurtosis|^skewness|^max|^min|^amplitude|^var|^avg|^stddev", names(train))
train_less <- train[,-remove]
crossval_less <- crossval[, -remove]
test_less <- test_raw[, -remove]
```

Checking correlations between the `train_less` variables (minus the `classe` variable, as this is what the algorithm should predict) suggests highly intercorrelated variables.


```r
corrcheck <- abs(cor(train_less[, -53])); diag(corrcheck) <- 0; which(corrcheck > .80, arr.ind = TRUE)
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_y       9   1
## accel_belt_z      10   1
## accel_belt_x       8   2
## magnet_belt_x     11   2
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_y       9   4
## accel_belt_z      10   4
## pitch_belt         2   8
## magnet_belt_x     11   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_z      10   9
## roll_belt          1  10
## total_accel_belt   4  10
## accel_belt_y       9  10
## pitch_belt         2  11
## accel_belt_x       8  11
## gyros_arm_y       19  18
## gyros_arm_x       18  19
## magnet_arm_x      24  21
## accel_arm_x       21  24
## magnet_arm_z      26  25
## magnet_arm_y      25  26
## accel_dumbbell_x  34  28
## accel_dumbbell_z  36  29
## gyros_dumbbell_z  33  31
## gyros_forearm_z   46  31
## gyros_dumbbell_x  31  33
## gyros_forearm_z   46  33
## pitch_dumbbell    28  34
## yaw_dumbbell      29  36
## gyros_forearm_z   46  45
## gyros_dumbbell_x  31  46
## gyros_dumbbell_z  33  46
## gyros_forearm_y   45  46
```

Plotting distribution of all `train_less` variables (minus the `classe` variable, as this is what the algorithm should predict) suggests non-normal distributions for the majority of predictors (output not shown). 


```r
hist.data.frame(train_less[, -53])
```

Thus, an algorithm that does not assume linearity and independence of predictors is warranted. 

### Model fit
The algorithm used here is random forest, as it is one of the best machine learning algorithms in terms of accuracy. It also provides details regarding importance of variables, with literally no risk of overfitting. In contrast to other classification tree algorithms (e.g.,`rpart`), there is no need for pruning. The model parameters will be as follows: ntrees = 2500 (as a larger number guarantees more stable and robust results), mtry = 8 (square of the number of variables, rounded up - see [here](http://web.stanford.edu/~stephsus/R-randomforest-guide.pdf) for reference). 


```r
set.seed(1234)
model_rf <- randomForest(train_less$classe ~., data=train_less, ntree=2500, mtry=8)
model_rf
```

```
## 
## Call:
##  randomForest(formula = train_less$classe ~ ., data = train_less,      ntree = 2500, mtry = 8) 
##                Type of random forest: classification
##                      Number of trees: 2500
## No. of variables tried at each split: 8
## 
##         OOB estimate of  error rate: 0.49%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    4    0    0    0    0.001024
## B   13 2639    6    0    0    0.007148
## C    0   16 2378    2    0    0.007513
## D    0    0   18 2233    1    0.008437
## E    0    0    2    5 2518    0.002772
```

The out-of-bag error is 0.49%, and the confusion matrix suggests that the model fit the `train_less` data very well. This also means that for cross-validation, the expected out-of-sample error should be very low. 

Calculate variable importance and print in order of importance (highest first). 
 

```r
model_rf_varimp <- varImp(model_rf)
model_rf_varimp$variable <- row.names(model_rf_varimp)
model_rf_varimp[order(model_rf_varimp$Overall, decreasing = TRUE), ]
```

```
##                      Overall             variable
## roll_belt             899.16            roll_belt
## yaw_belt              644.32             yaw_belt
## pitch_forearm         569.84        pitch_forearm
## magnet_dumbbell_z     546.77    magnet_dumbbell_z
## pitch_belt            505.14           pitch_belt
## magnet_dumbbell_y     480.07    magnet_dumbbell_y
## roll_forearm          461.67         roll_forearm
## magnet_dumbbell_x     335.96    magnet_dumbbell_x
## roll_dumbbell         297.10        roll_dumbbell
## accel_dumbbell_y      292.72     accel_dumbbell_y
## magnet_belt_z         276.39        magnet_belt_z
## accel_belt_z          271.96         accel_belt_z
## magnet_belt_y         265.28        magnet_belt_y
## accel_dumbbell_z      229.95     accel_dumbbell_z
## accel_forearm_x       228.07      accel_forearm_x
## gyros_belt_z          208.69         gyros_belt_z
## roll_arm              206.74             roll_arm
## magnet_forearm_z      205.26     magnet_forearm_z
## total_accel_dumbbell  188.94 total_accel_dumbbell
## magnet_arm_x          175.90         magnet_arm_x
## magnet_belt_x         173.76        magnet_belt_x
## yaw_dumbbell          173.74         yaw_dumbbell
## accel_dumbbell_x      171.81     accel_dumbbell_x
## gyros_dumbbell_y      168.13     gyros_dumbbell_y
## accel_forearm_z       167.15      accel_forearm_z
## yaw_arm               160.43              yaw_arm
## magnet_arm_y          159.16         magnet_arm_y
## magnet_forearm_y      157.76     magnet_forearm_y
## accel_arm_x           156.29          accel_arm_x
## magnet_forearm_x      149.82     magnet_forearm_x
## total_accel_belt      140.79     total_accel_belt
## magnet_arm_z          126.14         magnet_arm_z
## yaw_forearm           120.12          yaw_forearm
## pitch_arm             119.85            pitch_arm
## pitch_dumbbell        119.76       pitch_dumbbell
## accel_arm_y           108.33          accel_arm_y
## accel_forearm_y        96.27      accel_forearm_y
## gyros_arm_y            95.94          gyros_arm_y
## gyros_arm_x            92.68          gyros_arm_x
## accel_arm_z            89.76          accel_arm_z
## gyros_dumbbell_x       84.71     gyros_dumbbell_x
## accel_belt_y           83.94         accel_belt_y
## gyros_forearm_y        83.17      gyros_forearm_y
## accel_belt_x           79.60         accel_belt_x
## gyros_belt_y           78.31         gyros_belt_y
## total_accel_forearm    77.23  total_accel_forearm
## total_accel_arm        70.62      total_accel_arm
## gyros_belt_x           64.14         gyros_belt_x
## gyros_forearm_z        55.40      gyros_forearm_z
## gyros_dumbbell_z       54.69     gyros_dumbbell_z
## gyros_forearm_x        51.01      gyros_forearm_x
## gyros_arm_z            39.82          gyros_arm_z
```
### Cross-validation
Cross-validate on `crossval_less` and calculate accuracy and out-of-sample error respectively.


```r
cv <- table(actual = crossval_less$classe, predicted = predict(model_rf, newdata = crossval_less, type = "class"))
acc <- sum(diag(cv))/sum(cv)
oos <- 1-acc
oos
```

```
## [1] 0.006457
```

The prediction accuracy of the model on the cross-validation dataset `crossval_less` is almost 100%, with a very lowout-of-sample error of 0.0065. 

### Conclusion
As expected, the random forest algorithm performed very well, with a cross-validation prediction accuracy of almost 100%. 
