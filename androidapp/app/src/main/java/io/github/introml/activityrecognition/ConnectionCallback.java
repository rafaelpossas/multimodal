package io.github.introml.activityrecognition;

import android.util.Size;

/**
 * Created by rafaelpossas on 16/8/17.
 */

public interface ConnectionCallback {
    void onPreviewSizeChosen(Size size, int cameraRotation);
}
