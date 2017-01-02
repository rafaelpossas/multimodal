import itertools

import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import *
from sklearn.model_selection import cross_val_predict
from time import time
from ClassificatorSVM import ClassificatorSVM
from datasetEgoActivities import *
from SensorFeatureExtractor import *
import sklearn.metrics

# TODO ADD NORMALIZATION AND WHITENING

''' Create all possible combinations of the 6 sets of sensors: Combinatorials formed by groups of 1set, 2sets, 3sets.
A set of sensors contains all the axis for an specific sensor (rotation group contains additionally a scalar value).
Basically, a combinatorial in groups of 1 + combinatorials in groups of 2 ... + combinatorials in groups of 6 of the
elements in lstSensors. Each element in the sensor is the set of signals of each kind of sensor.
'''
def createCombinationsOfSensors():
    lstSensors = ([['accx', 'accy', 'accz'],
                   ['grax', 'gray', 'graz'],
                   ['gyrx', 'gyry', 'gyrz'],
                   ['lacx', 'lacy', 'lacz'],
                   ['magx', 'magy', 'magz'],
                   ['rotx', 'roty', 'rotz','rote'] # TODO try a configuration with and without rote
                   ])
    lstSensorsCombinations=[]
    for sizeGroups in range(1, 7):
        lstCombinations = itertools.combinations(lstSensors, sizeGroups)
        counter = 0
        for combination in lstCombinations:
            flattened = list(itertools.chain.from_iterable(combination))
            lstSensorsCombinations.append(flattened)
            counter += 1
        print('the Number of combinations with size of group={} is={} '.format(sizeGroups, counter))
    print('The total number of groups is: ' ,len(lstSensorsCombinations) )
    return lstSensorsCombinations

'''
Function that allow to return a dataset of the features  and their classes
- lstSeq: a list of sequences for each activity, it can be a list of numbers among 1 to 10
- lstActivities: a list of the number of the activities of the dataset. The list can have any set of numbers from 1 to 20
Returns:
- lstItems: matrix where each row contains the concatenated features of the sensors (in lstSensors) for a single sequenceOfActivity.
- lstClassNumbers: ndarray containing the class numbers for the elements in lstItems.
'''
def createDatasetSensors(lstSequencesOfActivities, lstSensors, aSensorFeatureExtractor):
    lstItems = np.zeros((len(lstSequencesOfActivities), len(lstSensors) * aSensorFeatureExtractor.numFeaturesPerSensor))
    lstClassNumbers = np.zeros(len(lstSequencesOfActivities), dtype=int)
    index = 0
    for activity,seq in lstSequencesOfActivities:
        dfSensors = dataset.getDataframeFromFile(activity, seq)
        lstItems[index] = aSensorFeatureExtractor.extractFeatures(dfSensors, lstSensors)
        lstClassNumbers[index] = activity;
        index=index+1
    print('DatasetSensors Created. NumElements={}, NumFeaturesTotalPerElement {}'.format(lstItems.shape[0],lstItems.shape[1]) )
    return lstItems,lstClassNumbers

def classify1SplitSameSequencesOverAllActivities(lstSensors, aSensorFeatureExtractor):
    lstTraining = [(a, s) for a in np.arange(1, 21) for s in np.arange(1, 9)]# all Activities, sequences 1-8
    lstTesting = [(a, s) for a in np.arange(1, 21) for s in np.arange(9, 11)]# all Activities, sequences 9 and 10
    lstFeaturesTraining, lstClassNumbersTraining = createDatasetSensors(lstTraining, lstSensors, aSensorFeatureExtractor)  # 160 for training  40 for testing
    lstFeaturesTesting, lstClassNumbersTesting = createDatasetSensors(lstTesting, lstSensors, aSensorFeatureExtractor)
    ClassificatorSVM().classify(lstFeaturesTraining, lstClassNumbersTraining, lstFeaturesTesting, lstClassNumbersTesting)

''' Performs the combinatorials of all the types of sensor in groups of 1 to 6 (as we have 6 types such as acc, mag, etc).
For each obtained group, perform the classification using leaveoneOut and stores the result in a file.
'''
def classifyCombSensorsStoreInFile(aSensorFeatureExtractor, filename):
    lstSensorsCombinations = createCombinationsOfSensors()
    lstAll = [(a, s) for a in np.arange(1, 21) for s in np.arange(1, 11)]
    #lstAll = [(a, s) for a in [1,2,3,17,18,19,20] for s in np.arange(1, 11)] # Moving activities
    print(lstAll)
    clf = svm.SVC()
    clf.decision_function_shape = "ovr"
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    with open(filename, "a") as myfile:
        myfile.write('Number of Sensors \t Sensors \t AccuracyTest \t AccuracyTrain \n')
    for lstSensors in lstSensorsCombinations:
        lstItems, lstClassNumbers = createDatasetSensors(lstAll, lstSensors, aSensorFeatureExtractor)
        iteratorCrossValidation = LeaveOneOut()
        ''' ***  Choose among computeTrainTestAccManuallySplitting and computeTestAccFromScikit ***
        computeTestAccFromScikit computes accTest and Train, is more complete than computeTestAccFromScikit.
        However, computeTestAccFromScikit generates the confusion matrix'''
        accTest, accTrain = computeTrainTestAccManuallySplitting(lstItems, lstClassNumbers, clf, iteratorCrossValidation)
        # accTest = computeTestAccFromScikit(lstItems, lstClassNumbers, clf, iteratorCrossValidation,lstSensors)
        # accTrain = 0
        '''This 2 lines can be putted instead of the previous one when no training acc is required and quickler computation is needed '''
        with open(filename, "a") as myfile:
            myfile.write('{}\t{}\t{}\t{}\n'.format(len(lstSensors), createFileName(lstSensors), accTest, accTrain))

'''
Classify the items using cross validation.
The indices of the split are received from the cross validation iterator cv.
Then, those indices are used to split the elements in thhe dataset to perform classification using the classifier clf.
The classifier is trained with the training set and the performance is tested over the training and testing set
returning both accuraciess.
'''
def computeTrainTestAccManuallySplitting(lstItems, lstClassNumbers, clf, cv):
    accuraciesTrain = np.zeros(len(lstClassNumbers))
    accuraciesTest = np.zeros(len(lstClassNumbers))
    index=0
    for train, test in cv.split(lstClassNumbers):# split just return the splitted indices
        lstFeaturesTraining = [lstItems[i] for i in train]
        lstClassNumbersTraining = [lstClassNumbers[i] for i in train]
        lstFeaturesTesting = [lstItems[i] for i in test]
        lstClassNumbersTesting = [lstClassNumbers[i] for i in test]
        #ClassificatorSVM().classify(lstFeaturesTraining, lstClassNumbersTraining, lstFeaturesTesting, lstClassNumbersTesting)

        clf.fit(lstFeaturesTraining, lstClassNumbersTraining)
        accuraciesTrain[index]= clf.score(lstFeaturesTraining, lstClassNumbersTraining)
        accuraciesTest[index] = clf.score(lstFeaturesTesting, lstClassNumbersTesting)
        index+=1
    return accuraciesTest.mean(), accuraciesTrain.mean()

'''
Classify the items using cross validation, similar to computeTrainTestAccManuallySplitting, but using the
native methods from Scikit to compute accuracy. These methods return a single accuracy of the testing set,
no training accuracy will be returned.
'''
def computeTestAccFromScikit(lstItems, lstClassNumbers, clf, cv, lstSensors):
    '''Way 1'''
    #scores =  cross_val_score(clf, lstItems, lstClassNumbers, cv=cv)
    #accuracy = scores.mean()

    '''Way 2'''
    predicted = cross_val_predict(clf, lstItems, lstClassNumbers, cv=cv)
    # print(predicted)
    #correct = np.count_nonzero(lstClassNumbers == predicted)
    #print("Quantity of correct classifications countnonzero of scores{} out of {}".format(correct, len(lstItems)))
    accuracy = metrics.accuracy_score(lstClassNumbers, predicted)
    cm = sklearn.metrics.confusion_matrix(lstClassNumbers, predicted)
    plot_confusion_matrix(cm, lstSensors)
    return accuracy






def plot_confusion_matrix(cm, lstSensors, title='Confusion matrix', cmap=plt.cm.Greens):  # Blues
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(dataset.lstActivityNames))
    plt.xticks(tick_marks, dataset.lstActivityNames, rotation=70, fontsize = 9)
    plt.yticks(tick_marks, dataset.lstActivityNames)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    fig.savefig('sensorsProcessing/data/' + createFileName(lstSensors) + '.png')
    fig.savefig('sensorsProcessing/data/' + createFileName(lstSensors) + '.pdf')
    plt.close(fig)

from collections import OrderedDict
def createFileName(lstSensors):
    listNoEnd = [elem[:-1] for elem in lstSensors]
    lstNoRep = list(OrderedDict.fromkeys(listNoEnd))
    return '-'.join(lstNoRep)



# =================== GLOBAL VARIABLES =======================
dataset =  DatasetEgoActivities()

def main():
    aSensorFeatureExtractor = SensorFeatureExtractorStatistical() #Statistical

    lstSensors = ['accx', 'accy', 'accz',
                  #'grax', 'gray', 'graz',
                   'gyrx', 'gyry', 'gyrz',
                  # 'lacx', 'lacy', 'lacz',
                  # 'magx', 'magy', 'magz',
                  #'rotx', 'roty', 'rotz', 'rote'  # another configuration just with and without rote
                  ]
    lstAll = [(a, s) for a in np.arange(1, 21) for s in np.arange(1, 11)]
    lstItems, lstClassNumbers = createDatasetSensors(lstAll, lstSensors, aSensorFeatureExtractor)
    clf = svm.SVC()
    clf.decision_function_shape = "ovr"
    cv = LeaveOneOut()

    # classify1SplitSameSequencesOverAllActivities(lstSensors)
    # print('******Testing a single configuration of Sensors using LeaveOneOut with 2 possible functions and measuring times')
    # t0 = time()
    # print('Accuracy from accelerometer', computeTrainTestAccManuallySplitting (lstItems, lstClassNumbers, clf, cv))
    # t1 = time()
    # print('Accuracy from accelerometer', computeTestAccFromScikit             (lstItems, lstClassNumbers, clf, cv))
    # t2 = time()
    # print(   'function vers1 takes %f' % (t1 - t0) )
    # print(   'function vers2 takes %f' % (t2 - t1) )

    predicted = cross_val_predict(clf, lstItems, lstClassNumbers, cv=cv)
    print(predicted)
    accuracy = metrics.accuracy_score(lstClassNumbers, predicted)
    print('acc', accuracy )
    cm = sklearn.metrics.confusion_matrix(lstClassNumbers, predicted)
    plot_confusion_matrix(cm,lstSensors)



    ''' For quickler computation substitute the indicated line in classifyCombSensorsStoreInFile (no train acc)'''
    print('======= Set of Experiments to Generate the Accuracy Over All Combinatorial of Sensors ========')
    classifyCombSensorsStoreInFile(SensorFeatureExtractorStatistical(), "1ExperimentStats.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatistical(), "8ExperimentStatsNoTrainAccPlotting.csv")
    #classifyCombSensorsStoreInFile(SensorFeatureExtractorRaw(), "2ExperimentRaw.csv")
    ## classifyCombSensorsStoreInFile(SensorFeatureExtractorRawFourier(), "3ExperimentFourier.csv")
    ## classifyCombSensorsStoreInFile(SensorFeatureExtractorStatsFourier(), "4ExperimentPlainAndFourierStats.csv")
    #classifyCombSensorsStoreInFile(SensorFeatureExtractorStatistical(), "5ExperimentStatsNormalized.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorRaw(), "5ExperimentRawNormalized.csv")
    #classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows10.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows25.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows20.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows5.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows3-5sec.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows50.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorStatisticalByWindows(), "6ExperimentStatsByWindows38.csv")
    # classifyCombSensorsStoreInFile(SensorFeatureExtractorSpectrogram(), "7ExperimentsSpectrogram.csv")


if __name__ == '__main__':
    main()
    #print(createCombinationsOfSensors())