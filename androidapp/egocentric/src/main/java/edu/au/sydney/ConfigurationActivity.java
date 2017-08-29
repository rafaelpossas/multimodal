package edu.au.sydney;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.Toast;

import edu.au.sydney.utils.MenuUtils;


/**
 * Created by rafaelpossas on 25/8/17.
 */

public class ConfigurationActivity extends Activity {

    private Spinner spinner_fps;
    private Spinner spinner_hz;
    private Button btnSave;
    private EditText editText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.configuration);

        spinner_fps = (Spinner) findViewById(R.id.fps_dropdown);
        spinner_hz = (Spinner) findViewById(R.id.hz_dropdown);
        btnSave = (Button) findViewById(R.id.btn_save);
        editText = (EditText) findViewById(R.id.edt_seconds);

        ConfigurationActivity.selectSpinnerItemByValue(spinner_fps, MainActivity.FPS.toString());
        ConfigurationActivity.selectSpinnerItemByValue(spinner_hz, MainActivity.HERTZ.toString());
        editText.setText(MainActivity.TOTAL_RECORDING_TIME_SECONDS.toString());

        btnSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                MainActivity.FPS = Integer.parseInt((String) spinner_fps.getSelectedItem());
                MainActivity.HERTZ = Integer.parseInt((String) spinner_hz.getSelectedItem());
                MainActivity.TOTAL_RECORDING_TIME_SECONDS = Integer.parseInt(editText.getText().toString());
                Toast.makeText(getApplicationContext(),"Settings were saved successfully",Toast.LENGTH_SHORT).show();

            }
        });
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
    public static void selectSpinnerItemByValue(Spinner spnr, String value) {
        for (int position = 0; position < spnr.getCount(); position++) {
            if(spnr.getItemAtPosition(position).equals(value)) {
                spnr.setSelection(position);
                return;
            }
        }
    }
}
