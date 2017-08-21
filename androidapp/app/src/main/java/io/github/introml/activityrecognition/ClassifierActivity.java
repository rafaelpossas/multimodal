package io.github.introml.activityrecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Environment;
import android.util.Log;
import android.util.Size;
import android.view.Display;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Created by rafaelpossas on 16/8/17.
 */

public class ClassifierActivity extends MainActivity implements ImageReader.OnImageAvailableListener, SensorEventListener {

    private List<Bitmap> bitmaps;

    private Long initiaTime;

    private Size mVideoSize;

    private Bitmap croppedBitmap;

    Matrix cropToFrameTransform;

    private Matrix frameToCropTransform;

    private boolean computing = false;

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final Boolean SAVE_PREVIEW_BITMAP = true;

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    int previewWidth = 0;
    int previewHeight = 0;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    Integer sensorOrientation;

    Classifier image_classifier;

    private static final boolean MAINTAIN_ASPECT = true;

    private long lastProcessingTimeMs;

    private Bitmap cropCopyBitmap;

    private String recording_name;

    private Boolean isRecording = false;

    private File sensorFile = null;



    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }
    @Override
    protected SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }
    private void startRecording(){
        isRecording = true;
        recording_name = "ACT_"+System.currentTimeMillis();
        sensorFile = createSensorFile();
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null)
            return;
        if (computing) {
            image.close();
            return;
        }
        computing = true;

        Bitmap outBitmap = Utils.YUV_420_888_toRGB(image, reader.getWidth(),reader.getHeight(), this);

        if(bitmaps == null) {
            bitmaps = new ArrayList<Bitmap>();
            initiaTime = System.currentTimeMillis();
        }

        if(croppedBitmap!=null) {
            final Canvas canvas = new Canvas(croppedBitmap);
            canvas.drawBitmap(outBitmap, frameToCropTransform, null);
            bitmaps.add(croppedBitmap);
            // For examining the actual TF input.
            if (SAVE_PREVIEW_BITMAP) {
                //ImageUtils.saveBitmap(croppedBitmap, ACTIVITY_ID + File.separator + initiaTime);
            }
        }


        if(bitmaps.size() == 30) {
            bitmaps = null;
            Log.e(CameraFragment.TAG, "Number of Seconds to collect 30 frames: "+ TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis() - initiaTime));
        }

        image.close();
        computing = false;
//        runInBackground(
//                new Runnable() {
//                    @Override
//                    public void run() {
//                        final long startTime = SystemClock.uptimeMillis();
//                        final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
//                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//                        System.out.println(results);
//                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//                        computing = false;
//                    }
//                });
    }

    private File createSensorFile() {
        String directory = recording_name;
        String filename = recording_name +".txt";
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "egocentric" + File.separator + directory;

        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
            Log.e(CameraFragment.TAG, "Directory Already exists");
        }

        final String fname = filename;
        File file = new File(myDir, fname);
        return file;
    }
    @Override
    public void onSensorChanged(SensorEvent event) {
        if(!isRecording)
            startRecording();
        activityPrediction();
        x.add(event.values[0]);
        y.add(event.values[1]);
        z.add(event.values[2]);
        Utils.saveSensor(event.values[0],event.values[1],event.values[2], sensorFile);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {
        if (x.size() == N_SAMPLES && y.size() == N_SAMPLES && z.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(x);
            data.addAll(y);
            data.addAll(z);

            results = classifier.predictProbabilities(toFloatArray(data));
//            downstairsTextView.setText(Float.toString(round(results[0], 2)));
//            joggingTextView.setText(Float.toString(round(results[1], 2)));
//            sittingTextView.setText(Float.toString(round(results[2], 2)));
//            standingTextView.setText(Float.toString(round(results[3], 2)));
//            upstairsTextView.setText(Float.toString(round(results[4], 2)));
//            walkingTextView.setText(Float.toString(round(results[5], 2)));
            float max = -1;
            int idx = -1;
            for (int i = 0; i < results.length; i++) {
                if (results[i] > max) {
                    idx = i;
                    max = results[i];
                }
            }
            sensor_label = labels[idx];
            sensor_prob = Float.toString(max);

            cur_activity_text.setText(sensor_label);
            cur_activity_prob.setText(sensor_prob);

            x.clear();
            y.clear();
            z.clear();
        }
    }
    protected void onPause() {
        getSensorManager().unregisterListener(this);
        results = null;
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
    }
    //@Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        image_classifier =
                ImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();


        sensorOrientation = rotation + screenOrientation;

        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                Utils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        INPUT_SIZE, INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

    }

    //@Override
    protected int getLayoutId() {
        return R.layout.fragment_camera2_video;
    }

    //@Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }
}
