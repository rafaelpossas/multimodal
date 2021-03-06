package edu.au.sydney;

import android.app.Activity;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.TextView;

import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
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
import java.io.IOException;
import java.nio.ByteBuffer;
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
        public Integer cur_activity;
    }

    public static final String TAG = "EGOCENTRIC";

    public static final String RECORDING = "RECORDING";
    public static final String PREDICTING = "PREDICTING";
    public static final String CONFIGURATION = "CONFIGURATION";

    public static Boolean TEXT_TO_SPEECH = true;


    protected static final int N_SAMPLES = 30;
    protected float[] results;
    protected String sensor_label = "";
    protected String sensor_prob = "";

    protected SensorClassifier classifier;
    protected ImageClassifier image_classifier;

    private static final long TIMEOUT = 1000L;
    public static Integer TOTAL_RECORDING_TIME_SECONDS = 300;
    public static Double SECONDS_TO_WARN = 40.0;
    public static Integer FPS = 15;
    public static Integer HERTZ = 15;

    public static String CUR_STATE = RECORDING;

    private Mat mRgba;
    private Mat mRgbaT;

    private int imageWidth;
    private int imageHeight;

    private int images_saved = 0;
    private FFmpegFrameRecorder recorder;
    private Frame yuvImage = null;

    private List<SensorXYZ> recording_acc;
    private List<SensorXYZ> recording_gyr;

    private List<Float> x_acc;
    private List<Float> y_acc;
    private List<Float> z_acc;

    private List<Float> x_gyr;
    private List<Float> y_gyr;
    private List<Float> z_gyr;

    private BlockingQueue<Mat> frames;

    private CameraBridgeViewBase mOpenCvCameraView;

    private Long snsInitialTime;
    private Long imgInitialTime = 0L;

    private TextView status;
    private TextView status_aux;

    private Boolean isRecordingImg = false;
    private Boolean isRecordingSns = false;

    private Boolean isClassifyingSns = false;
    private Boolean isClassifyingImg = false;

    private String recording_name;
    private String current_root_dir;

    private Handler handler;
    private HandlerThread handlerThread;

    private File accFile = null;
    private File gyrFile = null;

    private Timer timer_recording_image;
    private Timer timer_recording_sensor;

    private Timer timer_image_classifier;
    private Timer timer_sensor_classifier;

    private TextToSpeech textToSpeech;


    private String lastVisionLabel = "";

    private Long startTime = null;

    private int recorded_images;

    private int cur_activity;


    private SensorXYZ curAccValues;
    private SensorXYZ curGyroValues;

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
        frames = new LinkedBlockingQueue<Mat>();
        x_acc = new ArrayList<>();
        y_acc = new ArrayList<>();
        z_acc = new ArrayList<>();

        x_gyr = new ArrayList<>();
        y_gyr = new ArrayList<>();
        z_gyr = new ArrayList<>();

    }

    private Thread createWorker(){
        Thread thread = new Thread() {
            @Override
            public void run() {
                if(frames != null && frames.size() > 0){
                    if(CUR_STATE.equals(PREDICTING)){
                        Mat cropped = null;

                        try {
                            cropped = frames.poll(TIMEOUT, TimeUnit.MILLISECONDS);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }

                        Bitmap croppedBitmap = Bitmap.createBitmap(cropped.cols(),
                                cropped.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(cropped,croppedBitmap);

                        final List<Classifier.Recognition> results =
                                image_classifier.recognizeImage(croppedBitmap);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                if(results.size() > 0 &&
                                        !results.get(0).toString().split(" ")[1].equals(lastVisionLabel)
                                        && !textToSpeech.isSpeaking()){

                                    lastVisionLabel = results.get(0).toString().split(" ")[1];
                                    if(TEXT_TO_SPEECH){
                                        textToSpeech.speak(lastVisionLabel.replace("_"," "),
                                                TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                                    }

                                }
                                status_aux.setText(results.get(0).toString());
                            }
                        });


                    }
                }else if(frames != null && frames.size() == 0 && !status.getText().equals("") && (!isRecordingSns
                        && !isRecordingImg) && ! CUR_STATE.equals(PREDICTING)) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            status.setText("");
                        }
                    });
                }


            }
        };
        return thread;
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
        textToSpeech.speak("", TextToSpeech.QUEUE_FLUSH, null, Integer.toString(new Random().nextInt()));
        clearTexts();
        return MenuUtils.onOptionsItemSelected(item, this);
    }

    @Override
    public void onPause() {
        try{

            if(!MainActivity.CUR_STATE.equals(PREDICTING)){
                stopPredicting();
            }
            if(isRecordingImg){
                stopRecordingVideo();
            }
            if(isRecordingSns){
                stopRecordingSensor();
            }

            if(textToSpeech!=null){
                textToSpeech.speak("", TextToSpeech.QUEUE_FLUSH, null, Integer.toString(new Random().nextInt()));
                textToSpeech.shutdown();
            }


            getSensorManager().unregisterListener(this);

            frames = null;

            x_acc.clear();
            y_acc.clear();
            z_acc.clear();

            x_gyr.clear();
            y_gyr.clear();
            z_gyr.clear();

            if (!isFinishing()) {
                Log.i(TAG, "Requesting finish");
                finish();
            }
            handlerThread.quitSafely();

            try {
                handlerThread.join();
                handlerThread = null;
                handler = null;
            } catch (final InterruptedException e) {
                e.printStackTrace();
            }

            super.onPause();

        }catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public void onResume() {
        try{
            super.onResume();
            if(MainActivity.CUR_STATE.equals(PREDICTING)){
                startPredicting();
            }
            getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
            getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST);

            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
            } else {
                Log.d(TAG, "OpenCV library found inside package. Using it!");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }

            handlerThread = new HandlerThread("inference");
            handlerThread.start();
            handler = new Handler(handlerThread.getLooper());


            
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    public void onDestroy() {
        if(mOpenCvCameraView!=null)
            mOpenCvCameraView.disableView();
        super.onDestroy();

    }

    public void onCameraViewStarted(int width, int height) {
        mRgbaT = new Mat();
        mRgba = new Mat();
        imageWidth = width;
        imageHeight = height;
    }

    public void onCameraViewStopped() {
        mRgbaT.release();
        mRgba.release();
    }
    private TimerTask imageClassifyingTask() {
        return new TimerTask() {
            @Override
            public void run(){
                if(CUR_STATE.equals(PREDICTING)) {
                    if(mRgba!= null){
                        int x = (mRgba.width()/2) - (ImageClassifier.INPUT_SIZE/2);
                        int y = (mRgba.height()/2) - (ImageClassifier.INPUT_SIZE/2);

                        Rect roi = new Rect(x, y, ImageClassifier.INPUT_SIZE, ImageClassifier.INPUT_SIZE);

                        Mat cropped = new Mat(mRgba, roi);

                        try {
                            if(frames!=null) {
                                frames.put(cropped);
                                runInBackground(createWorker());
                            }
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                }
            }
        };
    }
    private TimerTask sensorClassifyingTask() {
        return new TimerTask() {
            @Override
            public void run() {
                x_acc.add(curAccValues.x);
                y_acc.add(curAccValues.y);
                z_acc.add(curAccValues.z);

                x_gyr.add(curGyroValues.x);
                y_gyr.add(curGyroValues.y);
                z_gyr.add(curGyroValues.z);

                if(CUR_STATE.equals(PREDICTING)) {
                    //Log.i(TAG,"Predicting activity from sensor");
                    if (x_acc.size() > N_SAMPLES || y_acc.size() > N_SAMPLES || z_acc.size() > N_SAMPLES) {

                        x_acc.clear();
                        y_acc.clear();
                        z_acc.clear();

                        x_gyr.clear();
                        y_gyr.clear();
                        z_gyr.clear();
                    }
                    if (x_acc.size() == N_SAMPLES && y_acc.size() == N_SAMPLES
                            && z_acc.size() == N_SAMPLES && CUR_STATE.equals(PREDICTING)) {

                        List<Float> data = new ArrayList<>();
                        data.addAll(x_acc);
                        data.addAll(y_acc);
                        data.addAll(z_acc);

                        results = classifier.predictProbabilities(ClassifierUtils.toFloatArray(data));

                        if (results != null && results.length > 0) {
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
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Log.i(TAG, "Updating sensor label");
                                    status.setText(sensor_label);
                                }
                            });

                            //cur_activity_prob.setText(sensor_prob);

                        }

                        x_acc.clear();
                        y_acc.clear();
                        z_acc.clear();

                        x_gyr.clear();
                        y_gyr.clear();
                        z_gyr.clear();
                    }
                }

            }
        };
    }
    private TimerTask imageRecordingTask(){
        return new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        status.setText("Recording ("+images_saved / FPS+")");
                        if(TEXT_TO_SPEECH && TOTAL_RECORDING_TIME_SECONDS - (images_saved * 1.0 / FPS) == SECONDS_TO_WARN){
                            textToSpeech.speak(SECONDS_TO_WARN.intValue() + " Seconds Left", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                        }
                    }
                });

                //Log.i(TAG,"Difference: "+diff);
                byte[] byteFrame = new byte[(int) (mRgba.total() * mRgba.channels())];

                mRgba.get(0, 0, byteFrame);

                onFrame(byteFrame);
                images_saved++;

                if(images_saved == FPS * TOTAL_RECORDING_TIME_SECONDS){
                    stopRecordingVideo();
                    if(TEXT_TO_SPEECH)
                        textToSpeech.speak("Recording Stopped", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
                }
            }
        };
    }
    private void stopRecordingSensor() {
        isRecordingSns = false;
        if (recording_acc != null) {
            for (int i = 0; i < recording_acc.size(); i++) {
                FileUtils.saveSensor(recording_acc.get(i).x, recording_acc.get(i).y,
                        recording_acc.get(i).z, recording_acc.get(i).cur_activity,
                        accFile);
            }
        }
        if (recording_gyr != null) {
            for (int i = 0; i < recording_gyr.size(); i++) {
                FileUtils.saveSensor(recording_gyr.get(i).x, recording_gyr.get(i).y,
                        recording_gyr.get(i).z, recording_gyr.get(i).cur_activity,
                        gyrFile);
            }
        }
        accFile = null;
        gyrFile = null;
        recording_acc = new ArrayList<>();
        recording_gyr = new ArrayList<>();
        timer_recording_sensor.cancel();
        Log.i(TAG, "Sensor recording stopped");
    }
    private TimerTask sensorRecordingTask(){
        return new TimerTask() {
            @Override
            public void run() {
                if (recording_acc == null) {
                    recording_acc = new ArrayList<>();
                }
                if (recording_gyr == null) {
                    recording_gyr = new ArrayList<>();
                }
                if (recording_acc.size() < HERTZ * TOTAL_RECORDING_TIME_SECONDS) {
                    if(curAccValues!=null)
                        recording_acc.add(curAccValues);
                }
                if(recording_gyr.size() < HERTZ * TOTAL_RECORDING_TIME_SECONDS) {
                    if(curGyroValues!=null)
                        recording_gyr.add(curGyroValues);
                }

                if (recording_acc.size() == HERTZ * TOTAL_RECORDING_TIME_SECONDS ||
                        recording_gyr.size() == HERTZ * TOTAL_RECORDING_TIME_SECONDS) {
                    stopRecordingSensor();
                }
            }
        };
    }

    private void startPredicting(){
        if(isRecordingImg){
            stopRecordingVideo();
        }
        if(isRecordingSns){
            stopRecordingSensor();
        }
        timer_image_classifier = new Timer();
        timer_image_classifier.scheduleAtFixedRate(imageClassifyingTask(), 1000, 1000/FPS);
        isClassifyingImg = true;

        timer_sensor_classifier = new Timer();
        timer_sensor_classifier.scheduleAtFixedRate(sensorClassifyingTask(), 1000, 1000/HERTZ);
        isClassifyingSns = true;
    }
    private void stopPredicting() {

        if(isClassifyingSns && timer_image_classifier!=null){
            timer_image_classifier.cancel();
            isClassifyingSns = false;
        }

        if(isClassifyingImg && timer_sensor_classifier!=null){
            timer_sensor_classifier.cancel();
            isClassifyingImg = false;
        }

    }

    private void startRecording(){

        recording_name = "REC_"+System.currentTimeMillis();
        accFile = createSensorFile("ACC");
        gyrFile = createSensorFile("GYR");

        cur_activity = 0;

        startRecordingVideo();

        isRecordingSns = true;
        isRecordingImg = true;
        images_saved = 0;

        timer_recording_image = new Timer();
        timer_recording_sensor = new Timer();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if(TEXT_TO_SPEECH)
                    textToSpeech.speak("Recording started", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
            }
        });
        timer_recording_image.scheduleAtFixedRate(imageRecordingTask(), 0, 1000/FPS);

        timer_recording_sensor.scheduleAtFixedRate(sensorRecordingTask(), 0, 1000/HERTZ);

        Log.i(TAG, "Recording Started");
    }


    private File createSensorFile(String sensor_type) {
        String directory = recording_name;
        String filename = sensor_type+"_"+recording_name +".txt";
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "egocentric" + File.separator + directory;

        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
            Log.e(TAG, "Directory Already exists");
        }
        current_root_dir = root;
        final String fname = filename;
        File file = new File(myDir, fname);
        return file;
    }
    private void onFrame(byte[] data){
            // Put the camera preview frame right into the yuvIplimage object
            try {
                // Get the correct time
                if(recorder!=null) {

                    // Record the image into FFmpegFrameRecorder
                    Frame frame = new Frame(imageWidth, imageHeight, Frame.DEPTH_UBYTE, 4);
                    ((ByteBuffer) frame.image[0].position(0)).put(data);
                    if(startTime == null){
                        startTime = System.currentTimeMillis();
                    }
                    //long videoTimestamp = 1000 * (System.currentTimeMillis() - startTime);
                    //recorder.setTimestamp(videoTimestamp);
                    recorder.record(frame);
                    recorded_images++;


                    Log.i(TAG, "Wrote Frame: " + recorded_images);
                }


            }
            catch (Exception e) {
                Log.v(TAG,e.getMessage());
                e.printStackTrace();
            }

    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Core.flip(mRgba, mRgba, -1);

        if(CUR_STATE.equals(RECORDING)){
            if(!status_aux.getText().equals(""))
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        status_aux.setText("");
                    }
                });
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
                cur_activity++;
                textToSpeech.speak("Activity Changed", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
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
    private void clearTexts(){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                status.setText("");
                status_aux.setText("");
            }
        });
    }
    @Override
    public void onSensorChanged(SensorEvent event) {
        if(event.sensor.getType() == Sensor.TYPE_ACCELEROMETER){
            curAccValues = new SensorXYZ();
            curAccValues.x = event.values[0];
            curAccValues.y = event.values[1];
            curAccValues.z = event.values[2];
            curAccValues.cur_activity = new Integer(cur_activity);
        }else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE){
            curGyroValues = new SensorXYZ();
            curGyroValues.x = event.values[0];
            curGyroValues.y = event.values[1];
            curGyroValues.z = event.values[2];
            curGyroValues.cur_activity = new Integer(cur_activity);
        }


    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        Log.i(TAG, "Sensor accuracy changed");
    }

    private void startRecordingVideo() {
        initRecorder();
        recorded_images = 0;

        try {
            if(recorder!=null){
                recorder.start();
                startTime = System.currentTimeMillis();
            }
        } catch(FFmpegFrameRecorder.Exception e) {
            e.printStackTrace();
        }
    }

    private void stopRecordingVideo() {
        Log.i(TAG,TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis()-startTime)+" Seconds");
        isRecordingImg = false;

        if(recorder != null) {

            Log.v(TAG, "Finishing recording, calling stop and release on recorder");
            try {
                recorder.stop();
                recorder.release();

            } catch(FFmpegFrameRecorder.Exception e) {
                e.printStackTrace();
            }
        }
        Log.i(TAG, "Bitmap recording stopped");
        timer_recording_image.cancel();
        clearTexts();
    }


    //---------------------------------------
    // initialize ffmpeg_recorder
    //---------------------------------------
    private void initRecorder() {
        Log.w(TAG,"initRecorder");


        Log.v(TAG, "IplImage.create");
        // }

        File videoFile = new File(current_root_dir + File.separator + recording_name+".mp4");
        boolean mk = videoFile.getParentFile().mkdirs();
        Log.v(TAG, "Mkdir: " + mk);

        boolean del = videoFile.delete();
        Log.v(TAG, "del: " + del);

        try {
            boolean created = videoFile.createNewFile();
            Log.v(TAG, "Created: " + created);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        String ffmpeg_link = videoFile.getAbsolutePath();
        recorder = new FFmpegFrameRecorder(ffmpeg_link, imageWidth, imageHeight, 1);
        Log.v(TAG, "FFmpegFrameRecorder: " + ffmpeg_link + " imageWidth: " + imageWidth + " imageHeight " + imageHeight);

        recorder.setFormat("mp4");
        Log.v(TAG, "recorder.setFormat(\"mp4\")");


        // re-set in the surface changed method as well
        recorder.setFrameRate(FPS);
        Log.v(TAG, "recorder.setFrameRate(frameRate)");
    }

}
