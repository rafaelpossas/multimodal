package io.github.introml.activityrecognition;

import android.Manifest;
import android.app.Fragment;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public abstract class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener, TextToSpeech.OnInitListener {

    static class SensorXYZ{
        public Float x;
        public Float y;
        public Float z;
    }
    protected List<SensorXYZ> sensorValues;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final int PERMISSIONS_REQUEST = 1;

    //private static final int N_SAMPLES = 30;
    protected static final int N_SAMPLES = 200;

    protected TextView cur_activity_text;
    protected TextView cur_activity_prob;
    protected Button record_btn;



    private TextToSpeech textToSpeech;

    protected float[] results;

    protected SensorClassifier classifier;

    private long lastUpdate = 0;
    private long curTime = System.currentTimeMillis();

    protected String sensor_label = "";
    protected String sensor_prob = "";

//    private String[] labels ={"walking", "walking upstairs", "walking downstairs", "elevator up", "elevator down", "escalator up",
//                              "escalator down", "sitting","eating", "drinking", "texting", "mak.phone calls","working at PC",
//                              "reading", "writting sentences", "organizing files","running", "doing push-ups", "doing sit-ups", "cycling"};
    protected String[] labels = {"Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }

        sensorValues = new ArrayList<>();

        cur_activity_prob = (TextView) findViewById(R.id.cur_activity_prob);
        cur_activity_text = (TextView) findViewById(R.id.cur_activity_title);
        record_btn = (Button) findViewById(R.id.record_start_btn);

        record_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                recordBtnListener();
            }
        });

        classifier = new SensorClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);

    }

    protected void setFragment() {
        final Fragment fragment =
                CameraFragment.newInstance(
                        new ConnectionCallback() {
                            @Override
                            public void onPreviewSizeChosen(final Size size, final int rotation) {
                                MainActivity.this.onPreviewSizeChosen(size, rotation);
                            }
                        },
                        this,
                        getLayoutId(),
                        getDesiredPreviewFrameSize());

        getFragmentManager()
                .beginTransaction()
                .add(R.id.container, fragment)
                .commit();
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.length == 0) {
                    return;
                }
                textToSpeech.speak(sensor_label, TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
            }
        }, 2000, 5000);
    }




//    @Override
//    public void onSensorChanged(SensorEvent event) {
//        // In onSensorChanged:
//        if(lastUpdate == 0) {
//            lastUpdate = System.currentTimeMillis();
//        }
//        curTime = System.currentTimeMillis();
//
//        if ((curTime - lastUpdate) >= 100){ // only reads data twice per second
//            lastUpdate = curTime;
//            activityPrediction();
//            x.add(event.values[0]);
//            y.add(event.values[1]);
//            z.add(event.values[2]);
//        }
//
//
//    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                    setFragment();
                } else {
                    requestPermission();
                }
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED && checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) || shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(MainActivity.this, "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    protected abstract SensorManager getSensorManager();
    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);
    protected abstract int getLayoutId();
    protected abstract Size getDesiredPreviewFrameSize();
    protected abstract void recordBtnListener();




}
