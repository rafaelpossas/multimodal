from SensorAgent import SensorAgent
from SensorDataset_UCI import SensorDatasetUCI
import numpy as np

sensor_agent = SensorAgent('models/acc_xyz.h5', num_classes=5)
dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned", sensors=['acc'])
dt.load_dataset(train_size=0.9, group_size=150, step_size=150,)

print(sensor_agent.predict(dt.x_test[0][np.newaxis, :, :]))
print(np.argmax(dt.y_train[0]))