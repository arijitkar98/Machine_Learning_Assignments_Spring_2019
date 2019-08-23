README for Part 2 of the Assignment:

(1) The code for part 2 of the assignment is present in ass2_2.py and ass2_2s.py
(2) At first, the program make_csv.py needs to run to convert the text files to csv files
(3) This can be done by the command --> python3 make_csv.py
(4) The files containing the training and testing data are now traindata.csv, trainlabel.text, testdata.csv, testlabel.txt and word.txt
(3) The program for My Model can be run using the command --> python3 ass2_2.py
(4) On running the program, the code first loads the training data, training labels, testing data and testing labels. 
(5) It then trains a decision tree (My Model) with Information Gain measure for 25 different values of maximum tree depth ranging from
	1 to 25. 
(6) In each iteration, it prints the depth of tree during training and then prints the final tree. It calculates the accuracy on both 
	training and testing dataset and prints it.
(7) The code finally plots the training and testing accuracy versus the max allowed tree depth.
(8) ass2_2s.py contains the code for the scikit-learn part. It loads the required data and then runs for 40 values of max allowed depth.
(9) It then creates a plot of the accuracies against max depth.
