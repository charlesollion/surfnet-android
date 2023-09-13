package org.surfrider.surfnet.detection.flow.classes;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;
import android.Manifest;

import org.surfrider.surfnet.detection.flow.classes.velocity_estimator.Basic_fusion;
import org.surfrider.surfnet.detection.flow.classes.velocity_estimator.FraneBack;
import org.surfrider.surfnet.detection.flow.classes.velocity_estimator.IMU_estimator;
import org.surfrider.surfnet.detection.flow.classes.velocity_estimator.KLT;
import org.surfrider.surfnet.detection.flow.classes.velocity_estimator.MotionVectorViz;
import org.surfrider.surfnet.detection.flow.dataTypes.velocity_estimator.OF_output;
import org.surfrider.surfnet.detection.flow.interfaces.velocity_estimator.OpticalFlow;
import org.surfrider.surfnet.detection.flow.interfaces.velocity_estimator.Sensor_fusion;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class MainActivity2 extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private final String TAG = MainActivity2.class.getSimpleName();

    static {
        // load OpenCV
        if (OpenCVLoader.initDebug()){
            Log.d("OpenCV", "success");
        }
        else{
            Log.d("OpenCV", "failed");
        }
    }
    private CameraBridgeViewBase mOpenCvCameraView;
    private ImageView motionVector;
    private Button reset_button, update_features_button;
    private Switch of_type;
    private SeekBar sensitivity_bar;
    private TextView vel_pred_text;
    private Mat curr_frame, mv_mat;
    private OF_output output;
    private OpticalFlow optical_flow;
    private IMU_estimator imu_estimator;
    private Sensor_fusion fusion;
    private float[] fuse_output;
    private MotionVectorViz mv_viewer;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");

                    mOpenCvCameraView.enableView();

                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 100);
        }
        // setContentView(R.layout.main_activity_layout);
        init_ui();
        init_vars();


    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 100) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d("PERMISSION", "camera permission granted");
            } else {
                Log.d("PERMISSION", "no camera permission granted");
            }
        }
    }


    private void init_vars(){
        // first initialize with KLT optical flow
        optical_flow = new KLT(vel_pred_text);
        output = new OF_output();

        // init fusion algorithm
        fusion = new Basic_fusion();
        fuse_output = new float[3];

        // init motion vector viewer
        mv_viewer = new MotionVectorViz(400, 400);
        mv_mat = Mat.zeros(400, 400, CvType.CV_8UC1);
    }

    private void init_ui(){
        // velocity prediction label
        //vel_pred_text = (TextView)findViewById(R.id.vel_pred);
        // reset Button

        // update features Button

        // Image view
        // motionVector = (ImageView) findViewById(R.id.motion_vector);
        // motionVector.setVisibility(View.VISIBLE);
        // Java view
        /*mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        ActivityCompat.shouldShowRequestPermissionRationale(this,
                android.Manifest.permission.CAMERA);    // permission
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);*/

        // init IMU_estimator
        imu_estimator = new IMU_estimator(this.getApplicationContext());
    }

    @Override
    protected void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug())
        {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this,
                            mLoaderCallback);
        }
        else
        {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    @Override
    public void onCameraViewStarted(int width, int height)
    {
        Log.d(TAG, "onCameraViewStarted");
    }

    @Override
    public void onCameraViewStopped()
    {
        Log.d(TAG, "onCameraViewStopped");
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame)
    {
        // get IMU variables
        float[] velocity = imu_estimator.getVelocity();
        float[] imu_position = imu_estimator.getPosition();
        // Convert the velocity to mph
        float xVelocityMph = velocity[0] * 2.23694f;
        float yVelocityMph = velocity[1] * 2.23694f;
        float zVelocityMph = velocity[2] * 2.23694f;

        Log.d("POS", String.valueOf(imu_position[0]) + ", " + String.valueOf(imu_position[1]) + ", " + String.valueOf(imu_position[2]));

        // Get the magnitude of the velocity vector
        float speedMph = (float) Math.sqrt(xVelocityMph * xVelocityMph + yVelocityMph * yVelocityMph + zVelocityMph * zVelocityMph);
        vel_pred_text.setText(String.valueOf(speedMph));

        // get OF output
        curr_frame = inputFrame.rgba();
        output = optical_flow.run(curr_frame);

        if (output.of_frame != null) {

            // fuse the IMU sensor with the Optical Flow
            fuse_output = fusion.getPosition(velocity, imu_position, output.position);

            // get Motion Vector Mat to present
            mv_mat = mv_viewer.getMotionVector(output.position);

            // draw Motion Vector
            Bitmap dst = Bitmap.createBitmap(mv_mat.width(), mv_mat.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mv_mat, dst);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    motionVector.setImageBitmap(dst);
                }
            });

            return output.of_frame;
        }
        return inputFrame.rgba();
    }

}
