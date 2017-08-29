package edu.au.sydney.utils;

import android.content.Context;
import android.content.Intent;
import android.view.MenuItem;

import edu.au.sydney.ConfigurationActivity;
import edu.au.sydney.MainActivity;
import edu.au.sydney.R;

/**
 * Created by rafaelpossas on 25/8/17.
 */

public class MenuUtils {

    public static boolean onOptionsItemSelected(final MenuItem item, Context context) {

        switch (item.getItemId()) {
            case R.id.menu_configuration:
                Intent intent_configuration = new Intent(context, ConfigurationActivity.class);
                context.startActivity(intent_configuration);
                return true;

            case R.id.menu_record:
                Intent intent_record = new Intent(context, MainActivity.class);
                context.startActivity(intent_record);
                MainActivity.CUR_STATE = MainActivity.RECORDING;
                return true;

            case R.id.menu_predict:
                Intent intent_predict = new Intent(context, MainActivity.class);
                context.startActivity(intent_predict);
                MainActivity.CUR_STATE = MainActivity.PREDICTING;
                return true;

        }
        return false;
    }
}
