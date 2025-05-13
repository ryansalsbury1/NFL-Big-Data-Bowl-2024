Kaggle - Big Data Bowl 2024

Model

Expected yards after the catch was calculated based on a random forest model using the following variables as inputs at the time of a catch:

Pass Catcher: speed, acceleration, direction, orientation, distance from nearest sideline, distance from opponent endzone, distance from line of scrimmage, distance from first down marker
Four Closest Defenders: y coordinate position, speed, acceleration, direction, orientation, distance from pass catcher

Number of offensive players ahead of the pass catcher (between endzone and pass catcher)

Number of defensive players ahead of the pass catcher (between endzone and pass catcher)

There was a total of 5475 plays in the model, 958 of which were used to validate the accuracy of the model. The r squared when fitting the model on the training set was .49 and when comparing expected yards with actual yards after the catch, the final model produced a correlation of .58 on the test data sample and .75 for the entire data set.
