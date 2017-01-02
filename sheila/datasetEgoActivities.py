import pandas as pd
'''
    Dataset: http://people.sutd.edu.sg/~1000892/dataset
    * Sensor data:
    Accelerometer(X, Y, Z)      -> 'accx', 'accy', 'accz'
    Gravity(X, Y, Z)            -> 'grax', 'gray', 'graz'
    Gyroscope(X, Y, Z)          -> 'gyrx', 'gyry', 'gyrz'
    Linear Acceleration(X, Y, Z)-> 'lacx', 'lacy', 'lacz'
    Magnetic Field(X, Y, Z)     -> 'magx', 'magy', 'magz' # Not found in the android description in the link below
    Rotation Vector(X, Y, Z, Scalar) >  'rotx', 'roty', 'rotz', 'rote'
    * Total: (19-dim data)
    * Details: https://developer.android.com/guide/topics/sensors/sensors_motion.html
    * Sample rate: 10 Hz
    * Length: 15 s (or 150 samples)

'The rotation vector sensor and the gravity sensor are the most frequently used sensors for motion detection
and monitoring. The rotational vector sensor is particularly versatile and can be used for a wide range of
motion-related tasks, such as detecting gestures, monitoring angular change, and monitoring relative orientation
changes.
For example, the rotational vector sensor is ideal if you are developing a game, an augmented reality application,
a 2-dimensional or 3-dimensional compass, or a camera stabilization app. In most cases, using these sensors is a
better choice than using the accelerometer and geomagnetic field sensor or the orientation sensor.'

'''
class DatasetEgoActivities:
    cols = ['accx', 'accy', 'accz',
            'grax', 'gray', 'graz',
            'gyrx', 'gyry', 'gyrz',
            'lacx', 'lacy', 'lacz',
            'magx', 'magy', 'magz',
            'rotx', 'roty', 'rotz', 'rote'];

    lstActivityNames = ['walking', 'walking upstairs', 'walking downstairs', 'rid.elevator up',
                        'rid.elevator down', 'rid.escalator up', 'rid.escalator down', 'sitting',
                        'eating', 'drinking', 'texting', 'mak.phone calls',
                        'working at PC', 'reading', 'writting sentences', 'organizing files',
                        'running', 'doing push-ups', 'doing sit-ups', 'cycling']


    path = "/home/rafaelpossas/dev/multimodal_dataset/";

    def getFileName(self, activity, sequence, isVideo=False):
        filename = 'act' + '{0:02d}'.format(activity) + 'seq' + '{0:02d}'.format(sequence)

        if isVideo: return self.path+'video/' +filename+'.mp4'
        else:       return self.path+'sensor/'+filename+'.csv'

    '''
    Returns the sensors captures of a specific example of a person developing an activity
    during 15 seconds. Each sensor is in one column of the dataframe  according to the
    variable cols in this class.
    The data in the several rows represent the signal in time obtained from the corresponding sensor
    '''
    def getDataframeFromFile(self, activity, sequence):
        filename = self.getFileName(activity, sequence, isVideo=False)
        return pd.read_csv(filename, sep=',', encoding='latin1', header=None, names=self.cols)

    def printActivities(self):
        for i in range(0, 20):
            print('{}=> {}'.format(i + 1, self.lstActivityNames[i]))
