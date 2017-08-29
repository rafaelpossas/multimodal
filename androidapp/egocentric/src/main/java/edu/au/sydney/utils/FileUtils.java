package edu.au.sydney.utils;

import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import edu.au.sydney.MainActivity;

/**
 * Created by rafaelpossas on 24/8/17.
 */

public class FileUtils {

    public static final String TAG = "FileUtils";

    public static void saveBitmap(final Bitmap bitmap, String dir) {
        saveBitmap(bitmap, System.currentTimeMillis()+".png", dir);
    }
    /**
     * Saves a Bitmap object to disk for analysis.
     *
     * @param bitmap The bitmap to save.
     * @param filename The location to save the bitmap to.
     */
    public static void saveBitmap(final Bitmap bitmap, final String filename, String directory) {
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "egocentric" + File.separator + directory;
        //Log.e(CameraFragment.TAG, "Saving "+bitmap.getWidth()+"x"+bitmap.getHeight()+" to "+root);
        final File myDir = new File(root);
        myDir.mkdirs();

        final String fname = filename;
        final File file = new File(myDir, fname);
        if (file.exists()) {
            file.delete();
        }
        try {
            final FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
            out.flush();
            out.close();
        } catch (final Exception e) {
            Log.e(MainActivity.TAG, e.toString());
        }
    }
    public static void saveSensor(float x,float y, float z, File file) {
        try {
            final FileOutputStream out;
            if(!file.exists()) {
                out = new FileOutputStream(file);
            }else {
                out = new FileOutputStream(file,true);
            }
            OutputStreamWriter myOutWriter = new OutputStreamWriter(out);
            myOutWriter.append(x+"\t"+y+"\t"+z+"\n");
            myOutWriter.close();
            out.close();
        } catch (final Exception e) {
            Log.e(MainActivity.TAG, e.toString());
        }
    }
}
