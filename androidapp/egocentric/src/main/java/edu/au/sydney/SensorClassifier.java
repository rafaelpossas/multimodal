package edu.au.sydney;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class SensorClassifier {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;
    // private static final String MODEL_FILE = "file:///android_asset/frozen_model.pb";
    private static final String MODEL_FILE = "file:///android_asset/sensor_model.pb";
    private static final String INPUT_NODE = "lstm_1_input";
    //private static final String INPUT_NODE = "inputs";
    private static final String[] OUTPUT_NODES = {"output_node0"};
    private static final String OUTPUT_NODE = "output_node0";
    //private static final String[] OUTPUT_NODES = {"y_"};
    //private static final String OUTPUT_NODE = "y_";
    private static final long[] INPUT_SIZE = {1, 30, 3};
    private static final int OUTPUT_SIZE = 20;

    public SensorClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);

        //Downstairs	Jogging	  Sitting	Standing	Upstairs	Walking
        return result;
    }
}
