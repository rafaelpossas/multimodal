import argparse

parser = argparse.ArgumentParser(description="Run commands for egocentric algorithms")

parser.add_argument('--sensor-train-file', type=str, help='The file containing all training sensors inputs', default='data/multimodal_full_train.hdf5')
parser.add_argument('--sensor-test-file', type=str, help='The file containing all test sensors inputs', default='data/multimodal_full_test.hdf5')
parser.add_argument('--sensor-chunk-size', type=int, help='The total number of sensor values per sample', default=50)
parser.add_argument('--sensor-step-size', type=int, help='The moving window step size for sensors', default=1)
parser.add_argument('--lstm-layer-size', type=int, help='The moving window step size for sensors', default=20)
parser.add_argument('--lstm-model-path', type=str, help='The moving window step size for sensors', default='checkpoints/sensor_model.hdf5')
parser.add_argument('--lstm-model-weights', type=str, help='The moving window step size for sensors', default='checkpoints/sensor_model.hdf5')

args = parser.parse_args()