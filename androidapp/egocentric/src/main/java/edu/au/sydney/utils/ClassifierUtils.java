package edu.au.sydney.utils;

import java.util.List;

/**
 * Created by rafaelpossas on 29/8/17.
 */

public class ClassifierUtils {
    public static float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }
}
