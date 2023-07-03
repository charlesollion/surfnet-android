import java.util.LinkedList;
import java.util.Vector;

import android.graphics.Matrix;
import org.apache.commons.math3.filter;



public class Tracker {

    private class Observation {
        public Observation(int frame_nb, Vector<Float> x0, float confidence, int class_id) {
            this.frame_nb = frame_nb;
            this.x0 = x0;
            this.confidence = confidence;
            this.class_id = class_id;
        }
        public int frame_nb;
        public Vector<Float> x0;
        public float confidence;
        public int class_id;
    }


    private final Matrix transition_covariance;
    private final Matrix observation_covariance;
    public Matrix initial_state_mean;
    public Matrix initial_state_covariance;
    public Matrix observation_matrices;
    
    private boolean updated;    
    private boolean enabled;
    private int steps_since_last_observation;
    private LinkedList<Observation> tracklet;
    private final float delta;

    public Tracker(int frame_nb, Vector<Float> x0, float confidence, int class_id, float delta) {
        updated = false;
        enabled = true;
        steps_since_last_observation = 0;
        this.delta = delta;
        Observation o = new Observation(frame_nb, x0, confidence, class_id);
        tracklet = new Tracklet(o);

        /*self.filter = KalmanFilter(
            initial_state_mean=X0,
            initial_state_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance,
            observation_matrices=np.eye(2),
            observation_covariance=self.observation_covariance,
        )*/

    }

    public void AddObservation(Observation o) {
        tracklet.add(o);
        updated = true;
    }

    public void AddObservation(int frame_nb, Vector<Float> x0, float confidence, int class_id) {
        AddObservation(new Observation(frame_nb, x0, confidence, class_id));
    }

    public void UpdateStatus(Matrix flow) {
        if(enabled && !updated) {
            steps_since_last_observation += 1;
            Update(None, None, None, flow);
        }
        else {
            steps_since_last_observation = 0;
        }
        updated = false;
    }

    /*
     *     def build_confidence_function(self, flow: Mat) -> Mat:

        """Build confidence function from flow.

        Args:
            flow (array): the input flow

        Returns:
            The computing confidence distribution based on predictive distribution from flow.
        """

        def confidence_from_multivariate_distribution(
            coord: array, distribution: array
        ) -> Mat:

            """Computes confidence from multivariate distribution.

            Args:
                coord (array): coordinates of the local object
                distribution (array): mathematical input distribution

            Returns:
                The computing confidence distribution.
            """

            delta = self.delta
            x = coord[0]
            y = coord[1]
            right_top = np.array([x + delta, y + delta])
            left_low = np.array([x - delta, y - delta])
            right_low = np.array([x + delta, y - delta])
            left_top = np.array([x - delta, y + delta])

            return (
                distribution.cdf(right_top)
                - distribution.cdf(right_low)
                - distribution.cdf(left_top)
                + distribution.cdf(left_low)
            )

        distribution = self.predictive_distribution(flow)

        return lambda coord: confidence_from_multivariate_distribution(
            coord, distribution
        )
     */
    public float ClassScoreFunction(float conf, int label) {
        // compute a score that takes into account the matching of labels
        // and the confidence
        float class_conf = 0.0f;
        flaot other_conf = 0.0f;
        for (Observation o : linkedList) {
            if(o.class_id == label) {
                class_conf += o.confidence;
            }
            other_conf += o.confidence;
        }

        return (class_conf + conf) / (other_conf + conf);
    }


}
