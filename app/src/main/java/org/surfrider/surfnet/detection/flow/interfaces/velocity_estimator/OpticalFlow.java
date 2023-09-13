package org.surfrider.surfnet.detection.flow.interfaces.velocity_estimator;

import org.surfrider.surfnet.detection.flow.dataTypes.velocity_estimator.OF_output;

import org.opencv.core.Mat;

public interface OpticalFlow {
    public OF_output run(Mat new_frame);
    public void reset_motion_vector();
    public void UpdateFeatures();

    public void set_sensitivity(int value);

}
