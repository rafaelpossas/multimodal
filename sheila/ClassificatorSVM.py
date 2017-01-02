import numpy as np
from sklearn import svm

from utils import Utils


class ClassificatorSVM():

    def classify(self, lstFeaturesTraining, lstClassNumbersTraining, lstFeaturesTesting, lstClassNumbersTesting):
        print('========= Training the Classifier SVM ========')
        Utils.printMatrixInformation(lstFeaturesTraining, 'lstItemsTraining')
        Utils.printMatrixInformation(lstClassNumbersTraining, 'lstClassNumbersTraining')

        clf2 = svm.SVC()
        clf2.decision_function_shape = "ovr"
        clf2.fit(lstFeaturesTraining, lstClassNumbersTraining)


        lstResultClassNumbersTraining = clf2.predict(lstFeaturesTraining)
        Utils.printMatrixInformation(lstResultClassNumbersTraining, 'lstResultClassNumbersTraining')

        correct = np.count_nonzero(lstResultClassNumbersTraining == lstClassNumbersTraining)
        print("Quantity of correct classifications {} out of {}".format(correct, lstResultClassNumbersTraining.size))
        print('Accuracy of the system for TRAINING is = {} % \n'.format(
            correct * 100.0 / lstResultClassNumbersTraining.size))

        print('============== Prediction using the Classifier -> Testing Set ==============')
        print('Number of testing Samples=', len(lstFeaturesTesting))
        Utils.printMatrixInformation(lstFeaturesTesting, 'lstItemsTesting')
        Utils.printMatrixInformation(lstClassNumbersTesting, 'lstClassNumbersTesting')

        lstResultClassNumbersTesting = clf2.predict(lstFeaturesTesting)
        Utils.printMatrixInformation(lstResultClassNumbersTesting, 'lstResultClassNumbersTesting')

        correct = np.count_nonzero(lstResultClassNumbersTesting == lstClassNumbersTesting)
        print("Quantity of correct classifications {} out of {}\n".format(correct, lstResultClassNumbersTesting.size))
        print('Accuracy of the system for TESTING is = {} % \n'.format(correct * 100.0 / lstResultClassNumbersTesting.size))
        print('SCORE BY SCKLEARN ON TESTING: ',clf2.score(lstFeaturesTesting, lstClassNumbersTesting))

        print('SCORE BY SCKLEARN ON TRAINING: ', clf2.score(lstFeaturesTraining, lstClassNumbersTraining))