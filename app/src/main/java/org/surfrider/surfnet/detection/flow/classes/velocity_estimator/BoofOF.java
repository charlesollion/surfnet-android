package org.surfrider.surfnet.detection.flow.classes.velocity_estimator;

import org.surfrider.surfnet.detection.flow.dataTypes.velocity_estimator.OF_output;
import org.surfrider.surfnet.detection.flow.interfaces.velocity_estimator.OpticalFlow;

import org.opencv.core.Mat;

public class BoofOF implements OpticalFlow {
    @Override
    public OF_output run(Mat new_frame) {
        return null;
    }

    @Override
    public void reset_motion_vector() {

    }

    @Override
    public void UpdateFeatures() {

    }

    @Override
    public void set_sensitivity(int value) {

    }
}
