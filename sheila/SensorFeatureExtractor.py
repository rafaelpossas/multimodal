from  datasetEgoActivities import *
from scipy.fftpack import fftfreq
from scipy.fftpack import fft, ifft
import numpy as np
# from FeatureExtraction.final import *
from matplotlib import pyplot as plt

'''
Extracts the Features of a set the set of features in lstSensors from a single example of the dataset (1 csv file)
represented by data in apanda dataframe object called dfSensors.
The features can be simply the row features, statistical properties, fourier data, a concatenation of the previous, etc.
'''
class SensorFeatureExtractor():
    numFeaturesPerSensor = 0  # class attribute

    def about(self):
        return ("The num of features per sensor is {}!".format(self.numFeaturesPerSensor))

    @classmethod
    def aboutCls(cls):
        return ("The num of features per sensor is {}!".format(cls.numFeaturesPerSensor))

    @classmethod
    def getNumFeaturesPerSensor(cls):
        return cls.numFeaturesPerSensor

    # Abstract method, defined by convention only
    def extractFeatures(self, dfSensors):
        raise NotImplementedError("Subclass must implement abstract method")

#===============================================================================================
#===============================================================================================
class SensorFeatureExtractorRaw(SensorFeatureExtractor):
    numFeaturesPerSensor = 146
    ''' In theory there are 15 seconds and 10 observations per second (10 Herz) which makes 150 values in time (cells)
     for each sensor (col).
     However, some files have 150 rows, others, 148, 146 (the one with less rows).
     Here, we consider the first 146 so it will work on all files.
    '''
    def extractFeatures(self, dfSensors, lstSensors):
        featureVector = dfSensors[lstSensors].values[0:self.numFeaturesPerSensor,:].ravel()
        return (featureVector) # Better without normalize(featureVector)
        ''' values[0:numFeaturesPerSensor,:] -> It goes until just the row that is present in all files
         Analysis of order of features (Go in a row-wise or col-wise way):
         consider 0accx, 0accy,..,0magx ..., 1accx, .., 2accx ---- OR ---
         1accx, 2accx ..., 149accx, 0accy, 1 accy....
         Tested both on SVM classifier obtaining same results. Then, I am using the row-wise way that is python default.
         To use a col-wise: add order='F' as an argument of ravel function
        '''
#===============================================================================================
class SensorFeatureExtractorStatistical(SensorFeatureExtractor):
    numFeaturesPerSensor = 7

    # def __init__(self, dfSensors, lstSensors):
    #   super().__init__(dfSensors,lstSensors)

    def extractFeatures(self, dfSensors, lstSensors):
        featureVector = (dfSensors.describe()[1:8][lstSensors]).values.ravel(order='F')
        return (featureVector)  # normalize()   better accucracy without normalize

# ===============================================================================================
class SensorFeatureExtractorStatisticalByWindows(SensorFeatureExtractor):
    windowSize = 38 # TODO when windowSize=5 =>ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    numWindows = round(150 / windowSize)
    numFeaturesPerSensor = 7 * numWindows

    # def __init__(self, dfSensors, lstSensors):
    #   super().__init__(dfSensors,lstSensors)

    def extractFeatures(self, dfSensors, lstSensors):
        featureVector = []
        low=0
        top=self.windowSize

        for indexWindow in range(0, int(self.numWindows)):
        #while (top <= len(dfSensors)+windowSize):# +windowSize so the last window has arbitrarySize as long as it is not absurd
            #if low+windowSize>=len(dfSensors): # last window # works when window has sigthly less elements but when having more elements, creates a new window with few elem
            #    top=len(dfSensors) # does not need this. it is an upper bound when i have few elemsbut python takes just the necessary elems correctly.
            if indexWindow==self.numWindows-1: #top+(windowSize/2) >=len(dfSensors): # when having few extra elemes for the next window, put all in current window
                top = len(dfSensors)
            windowDf = dfSensors[low:top].describe()
            print(windowDf.head())
            windowDf = windowDf[1:8][lstSensors]
            featureVector = np.hstack ((featureVector, windowDf.values.ravel(order='F')))
            #print( windowDf)
            #print('low={}, top={}'.format(low,top) )
            low=top
            top += self.windowSize
        #print('FeatureVector==>',featureVector.size)
        return np.nan_to_num(featureVector)  # normalize()   better accucracy without normalize
# 7 feature of stats for window 1 of accx, 7 features of stats for window1 of accy, .... window 1 stats of each sensor
# concatenated with window2 stats of each sensor and then window3
#===============================================================================================
class SensorFeatureExtractorRawFourier(SensorFeatureExtractor):
    numFeaturesPerSensor = 146

    def extractFeatures(self, dfSensors, lstSensors):
        print(lstSensors)
        for col in lstSensors: #dfSensors.columns.values:
            #print('- one col: ',dfSensors[col])
            dfSensors[col] = fft(dfSensors[col]) # ERROR OF NAN CONTINUE WORKING AFTER
            if col == 'accz':
                print('\n*===ACCZ \n', dfSensors[col])
                print('\n*===ACCZ std =>', dfSensors[col].values.std())
                print('\n*===ACCZ mean =>', dfSensors[col].values.mean())
                print('\n*===ACCZ max =>', dfSensors[col].values.max())
                print('\n*===ACCZ min =>', dfSensors[col].values.min())
            # print('- one col Converted: ', dfSensors[col])
            # this changes the original dfSensors object, however it is never used again, so ok
        return dfSensors[lstSensors].values[0:self.numFeaturesPerSensor,:].ravel() # raw of Fourier

#===============================================================================================

class SensorFeatureExtractorStatsFourier(SensorFeatureExtractor):
    numFeaturesPerSensor = 14

    def extractFeatures(self, dfSensors, lstSensors):
        lstFeaturesPlainStats = (dfSensors.describe()[1:8][lstSensors]).values.ravel(order='F')
        for col in lstSensors: #dfSensors.columns.values:
            #print('- one col: ',dfSensors[col])
            dfSensors[col] = fft(dfSensors[col]) # ERROR OF NAN
            # print('- one col Converted: ', dfSensors[col])
            # this changes the original dfSensors object, however it is never used again, so ok
        #print(dfSensors.describe()[1:8][lstSensors])
        # TODO second element of the stats the std sometimes return nan inestead of a real val.
        # Solution: USe dfSensors[col].values.std())  inestead,, for now just put 0
        return np.hstack ((lstFeaturesPlainStats,
         np.nan_to_num( (dfSensors.describe()[1:8][lstSensors]).values.ravel(order='F')  )  )) # Statistics of Fourier GO

#===============================================================================================
#===============================================================================================
import scipy.signal
class SensorFeatureExtractorSpectrogram(SensorFeatureExtractor):
    numFeaturesPerSensor = 74
    numRawFeaturesPerSensor = 146

    def extractFeatures(self, dfSensors, lstSensors):
        featuresConcatenated = np.asarray([])
        for col in lstSensors:  # dfSensors.columns.values:

            #Utils.printMatrixInformation(dfSensors[col].values[0:self.numRawFeaturesPerSensor],'- ONE COL OF DFSENSORS: ')
            f,t, features = scipy.signal.spectrogram(dfSensors[col].values[0:self.numRawFeaturesPerSensor], nperseg=3)  # ERROR OF NAN CONTINUE WORKING AFTER


            print('SIZEEEE of spectogram  = {}  for feature={}'.format(features.shape,col) )
            plt.pcolormesh(t, f, features)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
            features = features.ravel()
            featuresConcatenated = np.hstack((featuresConcatenated,features))
        print('SIZEEEE of featuresConcatenated', featuresConcatenated.shape)
        return featuresConcatenated

#===============================================================================================

def main():
    e = SensorFeatureExtractor()
    r = SensorFeatureExtractorRaw()
    s = SensorFeatureExtractorStatistical()
    fr = SensorFeatureExtractorRawFourier()
    fs = SensorFeatureExtractorStatsFourier()
    sw = SensorFeatureExtractorStatisticalByWindows()

    extractors = [r, s,e]

    for extractor in extractors:
        print('1.',extractor.numFeaturesPerSensor)
        print('2.',extractor.about())
        print('3.', extractor.aboutCls())
        print('4.', SensorFeatureExtractor.numFeaturesPerSensor)
        print('5.', extractor.getNumFeaturesPerSensor() )
    lstSensors = ['accx', 'accy', 'accz',
                  # 'grax', 'gray', 'graz',
                  # 'gyrx', 'gyry', 'gyrz',
                  # 'lacx', 'lacy', 'lacz',
                  # 'magx', 'magy', 'magz',
                  # 'rotx', 'roty', 'rotz', 'rote'
                  ]
    dfSensors = DatasetEgoActivities().getDataframeFromFile(1,1)  #  152 rows (18, 10) --- 148 rows(1,4)
    print('=== BeforeCalling Extract Features ===\n', dfSensors.head())
    print('Features => ', sw.extractFeatures(dfSensors,lstSensors)  )
    #print('=== AfterCalling Extract Features ===\n',dfSensors.head())

if __name__=='__main__':
    main()