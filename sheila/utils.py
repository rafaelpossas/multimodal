import numpy as np
import math

class Utils:

    @staticmethod
    def printMatrixInformation(matrix,name='matrix'):
        print('======= ' + name.upper() + ' matrix info =======')
        print(' - matrix =', matrix)
        print(' - type = ', type(matrix))
        print(' - len =  ', len(matrix))
        if type(matrix).__name__=='ndarray':
            print(' - dtype =', (matrix.dtype))
            print(' - shape =', matrix.shape)
            print(' - size  =', (matrix.size))
        print(' - type(matrix[0]) =', type(matrix[0]))
        if  type(matrix[0]).__name__=='ndarray' or  type(matrix[0]).__name__=='list': # len(matrix.shape)>1  or  just for ndarrays not lists
            print(' - len(matrix[0] =', len(matrix[0]))
            print(' - matrix[0][0]  =', matrix[0][0])


    @staticmethod
    # vector1 and vector 2 must be numpay arrays and have same dimension
    def distEuclidean(vector1, vector2):
        diff = vector2 - vector1
        squareDistance = np.dot(diff.T, diff)
        return math.sqrt(squareDistance)  # squareDistance,

    @staticmethod
    def sumElementWise(mat):
        # use np.sum( np.sum(array,axis=1), axis=0 )
        list_sum = np.zeros(len(mat[0, 0]))  # taking the first list
        for j in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                list_sum += mat[j, i]  # summing the 3 dim
        return list_sum

    @staticmethod
    def getRandomInteger(fromm, to):
        return np.random.randint(fromm, to)


if __name__ == '__main__':
    arr = np.array([1, 2, 3])
    matrix = np.array([arr, arr])
    Utils.printMatrixInformation(arr, 'Array')
    Utils.printMatrixInformation(matrix)
    Utils.printMatrixInformation([2,3,4])
    Utils.printMatrixInformation([[2, 3, 4],[5,6,7]])