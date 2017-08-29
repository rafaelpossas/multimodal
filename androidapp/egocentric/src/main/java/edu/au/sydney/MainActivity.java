package edu.au.sydney;

import android.app.Activity;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import edu.au.sydney.utils.ClassifierUtils;
import edu.au.sydney.utils.FileUtils;
import edu.au.sydney.utils.MenuUtils;


public class MainActivity extends Activity implements CvCameraViewListener2, SensorEventListener, TextToSpeech.OnInitListener {

    static class SensorXYZ{
        public Float x;
        public Float y;
        public Float z;
    }

    public static final String TAG = "EGOCENTRIC";

    public static final String RECORDING = "RECORDING";
    public static final String PREDICTING = "PREDICTING";

    public static Boolean TEXT_TO_SPEECH = false;


    protected static final int N_SAMPLES = 30;
    protected float[] results;
    protected String sensor_label = "";
    protected String sensor_prob = "";

    protected SensorClassifier classifier;
    protected ImageClassifier image_classifier;

    private static final long TIMEOUT = 1000L;
    public static Integer TOTAL_RECORDING_TIME_SECONDS = 10;
    public static Integer FPS = 15;
    public static Integer HERTZ = 10;

    public static String CUR_STATE = RECORDING;

    private Mat mRgba;
    private Mat mRgbaT;

    private Thread worker;
    private int images_saved = 0;

    private List<SensorXYZ> recording_sensor;
    private List<Float> x = new ArrayList<>();
    private List<Float> y = new ArrayList<>();
    private List<Float> z = new ArrayList<>();

    private BlockingQueue<Mat> frames = new LinkedBlockingQueue<Mat>();

    private CameraBridgeViewBase mOpenCvCameraView;

    private Long snsInitialTime;
    private Long imgInitialTime = 0L;

    private TextView status;
    private TextView status_aux;

    private Boolean isRecordingImg = false;
    private Boolean isRecordingSns = false;

    private String recording_name;

    private File sensorFile = null;

    private Timer timer;

    private TextToSpeech textToSpeech;

    //protected String[] labels = {"Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"};

    private String[] labels = {"walking", "walking upstairs", "walking downstairs", "elevator up", "elevator down", "escalator up",
                          "escalator down", "sitting","eating", "drinking", "texting", "mak.phone calls","working at PC",
                          "reading", "writting sentences", "organizing files","running", "doing push-ups", "doing sit-ups", "cycling"};
    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        status = (TextView) findViewById(R.id.status_text_view);
        status_aux = (TextView) findViewById(R.id.vision_text_view);

        classifier = new SensorClassifier(getApplicationContext());
        image_classifier = (ImageClassifier) ImageClassifier.create(getAssets());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);

        worker = new Thread() {
            @Override
            public void run() {
                while (true) {
                    if(frames.size() > 0){
                        if(isRecordingImg && isRecordingSns)
                            saveImageOnDisk();
                        else if(CUR_STATE.equals(PREDICTING)){
                            Mat cropped = null;

                            try {
                                cropped = frames.poll(TIMEOUT, TimeUnit.MILLISECONDS);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }

                            Bitmap croppedBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(cropped,croppedBitmap);

                            final List<Classifier.Recognition> results = image_classifier.recognizeImage(croppedBitmap);

                            Log.i(TAG, "Results size: "+results.size());
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    status_aux.setText(results.get(0).toString());
                                }
                            });


                        }
                    }else if(frames.size() == 0 && !status.getText().equals("") && (!isRecordingSns
                            && !isRecordingImg) && ! CUR_STATE.equals(PREDICTING)) {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                status.setText("");
                            }
                        });
                    }

                }
            }
        };
        worker.start();

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }
    public void saveImageOnDisk(){
        if(frames.size() > 0){
            Mat inputFrame = null;
            try {
                inputFrame = frames.poll(TIMEOUT, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (inputFrame == null || (inputFrame.cols() == 0 || inputFrame.rows() == 0)) {
                // timeout. Also, with a try {} catch block poll can be interrupted via Thread.interrupt() so not to wait for the timeout.
                return;
            }

            Bitmap outBitmap = Bitmap.createBitmap(inputFrame.cols(), inputFrame.rows(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(inputFrame, outBitmap);
            FileUtils.saveBitmap(outBitmap, recording_name + File.separator + "images");
            if(!isRecordingImg && !isRecordingSns && CUR_STATE.equals(RECORDING)){
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        status.setText("Saving image files on disk ("+frames.size()+" images left)");
                    }
                });

            }
        }
    }

    @Override
    public void onInit(int status) {
        Log.i(TAG,"Text to speech initialized");
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.option_menu, menu); //your file name
        return super.onCreateOptionsMenu(menu);
    }
    @Override
    public boolean onOptionsItemSelected(final MenuItem item) {
        return MenuUtils.onOptionsItemSelected(item, this);
    }
    @Override
    public void onPause() {
        super.onPause();
        getSensorManager().unregisterListener(this);
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgbaT = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mRgbaT.release();
        mRgba.release();
    }

    private void startRecording(){
        isRecordingSns = true;
        isRecordingImg = true;
        images_saved = 0;
        recording_name = "REC_"+System.currentTimeMillis();
        timer = new Timer();
        if(TEXT_TO_SPEECH)
            textToSpeech.speak("Recording started", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        status.setText("Recording ("+images_saved/FPS+")");
                    }
                });
            }
        }, 0, 1000);
        sensorFile = createSensorFile();

        Log.i(TAG, "Recording Started");
    }

    private void stopRecording() {
        if(!isRecordingImg && !isRecordingSns){
            timer.cancel();
            sensorFile = null;
            recording_sensor = null;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    status.setText("Saving image files on disk");
                }
            });
            if(TEXT_TO_SPEECH)
                textToSpeech.speak("Recording stopped", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));

            while(frames.size()!=0){
                saveImageOnDisk();
            }

            Log.i(TAG, "Recording Stopped");
        }
    }

    private File createSensorFile() {
        String directory = recording_name;
        String filename = recording_name +".txt";
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "egocentric" + File.separator + directory;

        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
            Log.e(TAG, "Directory Already exists");
        }

        final String fname = filename;
        File file = new File(myDir, fname);
        return file;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Core.flip(mRgba, mRgba, -1);

        //Log.i(TAG,"Difference: "+diff);

        if(imgInitialTime == 0L || System.currentTimeMillis() - imgInitialTime >= 1000 / FPS){

            imgInitialTime = System.currentTimeMillis();
            if (isRecordingImg) {

                try {
                    frames.put(mRgbaT);
                    images_saved++;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                if (images_saved >= FPS * TOTAL_RECORDING_TIME_SECONDS) {
                    isRecordingImg = false;
                    stopRecording();
                    Log.i(TAG, "Bitmap recording stopped");
                }
            }
            if(CUR_STATE.equals(PREDICTING)) {
                int x = (mRgba.width()/2) - (ImageClassifier.INPUT_SIZE/2);
                int y = (mRgba.height()/2) - (ImageClassifier.INPUT_SIZE/2);

                Rect roi = new Rect(x, y, ImageClassifier.INPUT_SIZE, ImageClassifier.INPUT_SIZE);

                Mat cropped = new Mat(mRgba, roi);

                try {
                    frames.put(cropped);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }



            }
        }

        return mRgba;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if(event.getAction() == MotionEvent.ACTION_DOWN){
            event.setAction(MotionEvent.ACTION_BUTTON_RELEASE);
            return onTrackballEvent(event);
        }
        return super.onTouchEvent(event);
    }

    @Override
    public boolean onTrackballEvent(MotionEvent event) {
        if(event.getAction() == MotionEvent.ACTION_BUTTON_RELEASE && CUR_STATE.equals(RECORDING)){
            Log.i(TAG,"Touch pad button released");
            if(isRecordingImg){
                stopRecording();
            }else{
                final ScheduledExecutorService exec = Executors.newScheduledThreadPool(1);
                if(TEXT_TO_SPEECH)
                    textToSpeech.speak("Recording will start in 5 seconds", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                exec.schedule(new Runnable(){
                    @Override
                    public void run(){
                        startRecording();
                    }
                }, 5, TimeUnit.SECONDS);

            }
        }
        return super.onTrackballEvent(event);
    }

    protected SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        SensorXYZ curSensorValues;

        if(CUR_STATE.equals(PREDICTING))
            activityPrediction();

        if(snsInitialTime == null){
            snsInitialTime = System.currentTimeMillis();
        }

        if(System.currentTimeMillis() - snsInitialTime >= 1000 / HERTZ){

            x.add(event.values[0]);
            y.add(event.values[1]);
            z.add(event.values[2]);

            snsInitialTime = null;


            if(isRecordingSns) {

                if (recording_sensor == null) {
                    recording_sensor = new ArrayList<>();
                }

                curSensorValues = new SensorXYZ();
                curSensorValues.x = event.values[0];
                curSensorValues.y = event.values[1];
                curSensorValues.z = event.values[2];

                if(recording_sensor.size() < HERTZ * TOTAL_RECORDING_TIME_SECONDS)
                    recording_sensor.add(curSensorValues);

                if (recording_sensor.size() == HERTZ * TOTAL_RECORDING_TIME_SECONDS) {
                    isRecordingSns = false;
                    for (int i = 0; i < recording_sensor.size() ; i++) {
                        FileUtils.saveSensor(recording_sensor.get(i).x,recording_sensor.get(i).y,recording_sensor.get(i).z, sensorFile);
                    }
                    stopRecording();

                    Log.i(TAG, "Sensor recording stopped");
                }
            }
        }

    }
    private void activityPrediction() {

        if (x.size() == N_SAMPLES && y.size() == N_SAMPLES
                && z.size() == N_SAMPLES && CUR_STATE.equals(PREDICTING) ) {

            List<Float> data = new ArrayList<>();
            data.addAll(x);
            data.addAll(y);
            data.addAll(z);

            results = classifier.predictProbabilities(ClassifierUtils.toFloatArray(data));

            if(results != null && results.length > 0) {
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

                status.setText(sensor_label);
                //cur_activity_prob.setText(sensor_prob);

            }

            x.clear();
            y.clear();
            z.clear();
        }
    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        Log.i(TAG, "Sensor accuracy changed");
    }
}
