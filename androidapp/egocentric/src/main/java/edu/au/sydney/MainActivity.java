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
    public static Integer TOTAL_RECORDING_TIME_SECONDS = 600;
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

    private List<SensorXYZ> recording_sensor;
    private List<Float> x;
    private List<Float> y;
    private List<Float> z;

    private BlockingQueue<Mat> frames;

    private CameraBridgeViewBase mOpenCvCameraView;

    private Long snsInitialTime;
    private Long imgInitialTime = 0L;

    private TextView status;
    private TextView status_aux;

    private Boolean isRecordingImg = false;
    private Boolean isRecordingSns = false;

    private String recording_name;
    private String current_root_dir;

    private File sensorFile = null;

    private Timer timer;

    private TextToSpeech textToSpeech;

    private Handler handler;
    private HandlerThread handlerThread;

    private String lastVisionLabel = "";

    private Long startTime = null;

    private int recorded_images;

    private int cur_activity;

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
        createWorker().start();
        frames = new LinkedBlockingQueue<Mat>();
        x = new ArrayList<>();
        y = new ArrayList<>();
        z = new ArrayList<>();

    }

    private Thread createWorker(){
        Thread thread = new Thread() {
            @Override
            public void run() {
                while(true){
                    if(frames.size() > 0){
                        if(isRecordingImg) {
                            saveImageOnDisk(false);
                        }
                        else if(CUR_STATE.equals(PREDICTING)){
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
    public void saveImageOnDisk(Boolean singleImages){
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
            if(singleImages){
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
            }else{
                byte[] byteFrame = new byte[(int) (inputFrame.total() * inputFrame.channels())];
                mRgbaT.get(0, 0, byteFrame);
                onFrame(byteFrame);
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
        textToSpeech.speak("", TextToSpeech.QUEUE_FLUSH, null, Integer.toString(new Random().nextInt()));
        clearTexts();
        return MenuUtils.onOptionsItemSelected(item, this);
    }

    @Override
    public void onPause() {
        try{
            if(!MainActivity.CUR_STATE.equals(CONFIGURATION))
                MainActivity.CUR_STATE = RECORDING;

            if(textToSpeech!=null){
                textToSpeech.speak("", TextToSpeech.QUEUE_FLUSH, null, Integer.toString(new Random().nextInt()));
                textToSpeech.shutdown();
            }

            handlerThread.quitSafely();
            try {
                handlerThread.join();
                handlerThread = null;
                handler = null;
            } catch (final InterruptedException e) {
                Log.e(TAG, e.toString());
            }
            getSensorManager().unregisterListener(this);

            frames = null;

            x.clear();
            y.clear();
            z.clear();
            if (!isFinishing()) {
                Log.i(TAG, "Requesting finish");
                finish();
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
            getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);

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

    private void startRecording(){
        recording_name = "REC_"+System.currentTimeMillis();
        sensorFile = createSensorFile();
        cur_activity = 0;
        startRecordingVideo();
        isRecordingSns = true;
        isRecordingImg = true;
        images_saved = 0;
        timer = new Timer();
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if(TEXT_TO_SPEECH)
                    textToSpeech.speak("Recording started", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
            }
        });

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

        Log.i(TAG, "Recording Started");
    }

    private void stopRecording() {

        if(!isRecordingImg && !isRecordingSns){
            while(frames.size()!=0){
                saveImageOnDisk(false);
            }
            stopRecordingVideo();
            timer.cancel();
            sensorFile = null;
            recording_sensor = null;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if(TEXT_TO_SPEECH)
                        textToSpeech.speak("Recording stopped", TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));

                    status.setText("Saving image files on disk");
                }
            });



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
        current_root_dir = root;
        final String fname = filename;
        File file = new File(myDir, fname);
        return file;
    }
    private void onFrame(byte[] data){
            if(startTime == null) {
                startTime = System.currentTimeMillis();
            }
            long videoTimestamp = 1000 * (System.currentTimeMillis() - startTime);

            // Put the camera preview frame right into the yuvIplimage object

            try {

                // Get the correct time
                if(recorder!=null) {
                    //recorder.setTimestamp(videoTimestamp);

                    // Record the image into FFmpegFrameRecorder
                    Frame frame = new Frame(imageWidth, imageHeight, Frame.DEPTH_UBYTE, 4);
                    ((ByteBuffer) frame.image[0].position(0)).put(data);
                    recorder.record(frame);
                    recorded_images++;


                    Log.i(TAG, "Wrote Frame: " + recorded_images);
                }


            }
            catch (FFmpegFrameRecorder.Exception e) {
                Log.v(TAG,e.getMessage());
                e.printStackTrace();
            }

    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Core.flip(mRgba, mRgba, -1);


        //Log.i(TAG,"Difference: "+diff);

        if(imgInitialTime == 0L || System.currentTimeMillis() - imgInitialTime >= 1000 / FPS){

            imgInitialTime = System.currentTimeMillis();
            if(CUR_STATE.equals(RECORDING)){
                if(!status_aux.getText().equals(""))
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            status_aux.setText("");
                        }
                    });
                if (isRecordingImg) {
                    try {
                        if (images_saved < FPS * TOTAL_RECORDING_TIME_SECONDS){
                            mRgbaT = new Mat();
                            mRgba.copyTo(mRgbaT);
                            frames.put(mRgbaT);
                            images_saved++;

                        }
                        Log.i(TAG, "Frames size is: "+frames.size());

                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    if (images_saved >= FPS * TOTAL_RECORDING_TIME_SECONDS && frames.size() == 0) {
                        isRecordingImg = false;
                        stopRecording();
                        Log.i(TAG, "Bitmap recording stopped");
                    }
                }
            }

            if(CUR_STATE.equals(PREDICTING)) {
                int x = (mRgba.width()/2) - (ImageClassifier.INPUT_SIZE/2);
                int y = (mRgba.height()/2) - (ImageClassifier.INPUT_SIZE/2);

                Rect roi = new Rect(x, y, ImageClassifier.INPUT_SIZE, ImageClassifier.INPUT_SIZE);

                Mat cropped = new Mat(mRgba, roi);

                try {
                    if(frames!=null)
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
                cur_activity++;
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
                curSensorValues.cur_activity = new Integer(cur_activity);

                if(recording_sensor.size() < HERTZ * TOTAL_RECORDING_TIME_SECONDS)
                    recording_sensor.add(curSensorValues);

                if (recording_sensor.size() == HERTZ * TOTAL_RECORDING_TIME_SECONDS) {
                    isRecordingSns = false;
                    for (int i = 0; i < recording_sensor.size() ; i++) {
                        FileUtils.saveSensor(recording_sensor.get(i).x,recording_sensor.get(i).y,
                                recording_sensor.get(i).z, recording_sensor.get(i).cur_activity,
                                sensorFile);
                    }
                    stopRecording();

                    Log.i(TAG, "Sensor recording stopped");
                }
            }
        }

    }
    private void activityPrediction() {
        //Log.i(TAG,"Predicting activity from sensor");
        if(x.size() > N_SAMPLES || y.size() > N_SAMPLES || z.size() > N_SAMPLES){
            x.clear();
            y.clear();
            z.clear();
        }
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
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Log.i(TAG, "Updating sensor label");
                        status.setText(sensor_label);
                    }
                });

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
    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    private void startRecordingVideo() {
        initRecorder();
        recorded_images = 0;

        try {
            recorder.start();
        } catch(FFmpegFrameRecorder.Exception e) {
            e.printStackTrace();
        }
    }

    private void stopRecordingVideo() {

        if(recorder != null) {

            Log.v(TAG, "Finishing recording, calling stop and release on recorder");
            try {
                recorder.stop();
                recorder.release();
            } catch(FFmpegFrameRecorder.Exception e) {
                e.printStackTrace();
            }
            recorder = null;
        }
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
