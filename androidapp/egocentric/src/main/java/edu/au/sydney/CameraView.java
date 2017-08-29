package edu.au.sydney;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;


import org.opencv.android.JavaCameraView;

/**
 * Created by rafaelpossas on 26/8/17.
 */

public class CameraView extends JavaCameraView {

    public CameraView(Context context, int cameraId) {
        super(context, cameraId);
    }
    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }
}
