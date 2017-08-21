package io.github.introml.activityrecognition;

import android.Manifest;
import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Created by rafaelpossas on 16/8/17.
 */

public class CameraFragment extends Fragment {

    private TextureView mTextureView;

    private HandlerThread mBackgroundThread;

    private Handler mBackgroundHandler;

    private CameraDevice mCameraDevice;

    public static final String TAG = "egocentric";

    private CaptureRequest.Builder mPreviewBuilder;

    private static final String[] VIDEO_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
    };


    private ImageReader mImageReader;

    /**
     * The {@link Size} of video recording.
     */
    private Size mVideoSize;

    /**
     * A reference to the current {@link CameraCaptureSession} for
     * preview.
     */
    private CameraCaptureSession mPreviewSession;

    /**
     * A {@link Semaphore} to prevent the app from exiting before closing the camera.
     */
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);

    private static final int REQUEST_VIDEO_PERMISSIONS = 1;

    private static final String FRAGMENT_DIALOG = "dialog";

    /**
     * The {@link Size} of camera preview.
     */
    private Size mPreviewSize;

    /**
     * A {@link ImageReader.OnImageAvailableListener} to receive frames as they are available.
     */
    private ImageReader.OnImageAvailableListener imageListener;

    private ConnectionCallback cameraConnectionCallback;

    /**
     * The layout identifier to inflate for this Fragment.
     */
    private int layout;

    /** The input size in pixels desired by TensorFlow (width and height of a square bitmap). */
    private Size inputSize;

    private Integer sensorOrientation;

    public CameraFragment(){}

    public static CameraFragment newInstance(
            final ConnectionCallback callback,
            final ImageReader.OnImageAvailableListener imageListener,
            final int layout,
            final Size inputSize) {
        return new CameraFragment(callback, imageListener, layout, inputSize);
    }

    private CameraFragment(
            final ConnectionCallback connectionCallback,
            final ImageReader.OnImageAvailableListener imageListener,
            final int layout,
            final Size inputSize) {
        this.cameraConnectionCallback = connectionCallback;
        this.imageListener = imageListener;
        this.layout = layout;
        this.inputSize = inputSize;
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_camera2_video, container, false);
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        mTextureView = (TextureView) view.findViewById(R.id.texture);
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
        if (mTextureView!=null && mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else if (mTextureView!=null) {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            closePreviewSession();
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.");
        } finally {
            mCameraOpenCloseLock.release();
        }
    }
    private boolean hasPermissionsGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(getActivity(), permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Given {@code choices} of {@code Size}s supported by a camera, chooses the smallest one whose
     * width and height are at least as large as the respective requested values, and whose aspect
     * ratio matches with the specified value.
     *
     * @param choices     The list of sizes that the camera supports for the intended output class
     * @param width       The minimum desired width
     * @param height      The minimum desired height
     * @param aspectRatio The aspect ratio
     * @return The optimal {@code Size}, or an arbitrary one if none were big enough
     */
    private static Size chooseOptimalSize(Size[] choices, int width, int height, Size aspectRatio) {
        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getHeight() == option.getWidth() * h / w &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }

        // Pick the smallest of those, assuming we found any
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CameraFragment.CompareSizesByArea());
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }
    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }

    }

    private boolean openCamera(int width, int height) {

        final Activity activity = getActivity();
        if (null == activity || activity.isFinishing()) {
            return false;
        }
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {

            Log.d(TAG, "tryAcquire");
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            String cameraId = manager.getCameraIdList()[0];

            // Choose the sizes for camera preview and video recording
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

            StreamConfigurationMap map = characteristics
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

            if (map == null) {
                Log.e(TAG, "No StreamConfigurationMap available");
                return false;
            }

            final Size largest =
                    Collections.max(
                            Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888)),
                            new CameraFragment.CompareSizesByArea());

            Size[] sizes = map.getOutputSizes(ImageReader.class);
            if (sizes.length == 0)
                return false;

            mVideoSize = chooseOptimalSize(sizes,width, height,largest);
            mImageReader = ImageReader.newInstance(mVideoSize.getWidth(),
                    mVideoSize.getHeight(),
                    ImageFormat.YUV_420_888, 3);
            mImageReader.setOnImageAvailableListener(imageListener,mBackgroundHandler);

            // Get all available size for the textureSurface preview window
            sizes = map.getOutputSizes(SurfaceTexture.class);

            // Get the optimal size for a preview window
            mPreviewSize = chooseOptimalSize(sizes, width, height,largest);
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            cameraConnectionCallback.onPreviewSizeChosen(mPreviewSize, sensorOrientation);
            configureTransform(width, height);
            manager.openCamera(cameraId, mStateCallback, null);

        } catch (CameraAccessException e) {
            Log.e(TAG, "Cannot access the camera.", e);
            return false;
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while trying to lock camera opening.");
            return false;
        } catch (SecurityException e) {
            Log.e(TAG, "No access to camera device", e);
            return false;
        }
        return true;

    }

    private void closePreviewSession() {
        if (mPreviewSession != null) {
            mPreviewSession.close();
            mPreviewSession = null;
        }
    }
    private TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {

        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture,
                                              int width, int height) {
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture,
                                                int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        }

    };

    /**
     * Configures the necessary {@link Matrix} transformation to `mTextureView`.
     * This method should not to be called until the camera preview size is determined in
     * openCamera, or until the size of `mTextureView` is fixed.
     *
     * @param viewWidth  The width of `mTextureView`
     * @param viewHeight The height of `mTextureView`
     */
    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }
    /**
     * {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its status.
     */
    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
            mCameraOpenCloseLock.release();
            if (null != mTextureView) {
                configureTransform(mTextureView.getWidth(), mTextureView.getHeight());

            }
        }
        /**
         * Start the camera preview.
         */
        private void startPreview() {
            if (mCameraDevice == null ||!mTextureView.isAvailable() ||mPreviewSize == null) {
                return;
            }
            try {
                SurfaceTexture texture = mTextureView.getSurfaceTexture();
                texture.setDefaultBufferSize(mPreviewSize.getWidth(),
                        mPreviewSize.getHeight());
                mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                List surfaces = new ArrayList<>();

                Surface previewSurface = new Surface(texture);
                surfaces.add(previewSurface);
                mPreviewBuilder.addTarget(previewSurface);

                Surface readerSurface = mImageReader.getSurface();
                surfaces.add(readerSurface);
                mPreviewBuilder.addTarget(readerSurface);

                mCameraDevice.createCaptureSession(surfaces,
                        new CameraCaptureSession.StateCallback() {

                            @Override
                            public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                                mPreviewSession = cameraCaptureSession;
                                updatePreview();
                            }

                            @Override
                            public void onConfigureFailed(CameraCaptureSession cameraCaptureSession){
                                Log.w(TAG, "Create capture session failed");
                            }
                        }, mBackgroundHandler);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
            Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }
        /**
         * Update the camera preview. {@link #startPreview()} needs to be called in advance.
         */
        private void updatePreview() {
            if (null == mCameraDevice) {
                return;
            }
            try {
                setUpCaptureRequestBuilder(mPreviewBuilder);
                HandlerThread thread = new HandlerThread("CameraPreview");
                thread.start();
                mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }
        private void setUpCaptureRequestBuilder(CaptureRequest.Builder builder) {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        }



    };



}
