// src/kalman_filter_2d.cpp
#include "ros2_object_detection/kalman_filter_2d.hpp"

// Standard C++ includes
#include <cmath> // Not strictly needed here, but good practice for math operations

// --- KalmanFilter2D Implementation ---
/**
 * @brief Constructor for the KalmanFilter2D class.
 *
 * Initializes the state transition (A), measurement (H), process noise (Q),
 * and measurement noise (R) covariance matrices for a constant velocity 2D Kalman filter.
 */
KalmanFilter2D::KalmanFilter2D() {
    // Resize matrices and vectors
    A.resize(4, 4);
    H.resize(2, 4);
    Q.resize(4, 4);
    R.resize(2, 2);
    P.resize(4, 4);
    x_hat.resize(4);

    // A: State Transition Matrix (Constant Velocity Model)
    // x_k = x_{k-1} + vx_{k-1}*dt
    // y_k = y_{k-1} + vy_{k-1}*dt
    // vx_k = vx_{k-1}
    // vy_k = vy_{k-1}
    // dt: time step between frames. Assuming 30 FPS, dt = 1/30 seconds.
    // This value might need to be adjusted based on the actual frame rate of your video stream.
    double dt = 1.0 / 30.0; // Default: 1/30 sec (for 30 FPS)

    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;

    // H: Measurement Matrix
    // We only measure position (x, y) directly from the object detector.
    // measured_x = x
    // measured_y = y
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    // Q: Process Noise Covariance Matrix
    // Represents the uncertainty in our state transition model (how much the object's velocity
    // is expected to deviate from a constant model between predictions).
    // Tune these values based on how "smooth" vs. "responsive" you want the filter to be.
    // Higher values make the filter trust its prediction less, leading to faster adaptation but more jitter.
    double process_noise_pos = 1.0;  // Variance in position (pixels^2) per frame for unmodeled acceleration
    double process_noise_vel = 0.5;  // Variance in velocity (pixels/frame)^2 per frame for velocity changes
    Q << process_noise_pos, 0, 0, 0,
         0, process_noise_pos, 0, 0,
         0, 0, process_noise_vel, 0,
         0, 0, 0, process_noise_vel;

    // R: Measurement Noise Covariance Matrix
    // Represents the uncertainty/variance in our measurements (from the object detector).
    // Tune based on the expected accuracy (pixel variance) of your object detector's bounding box centers.
    // Lower values mean the filter trusts new measurements more.
    double measurement_noise_xy = 25.0; // Assuming a variance of 25 pixels^2 (approx. std dev of 5 pixels)
    R << measurement_noise_xy, 0,
         0, measurement_noise_xy;

    // P: Initial Error Covariance Matrix
    // Represents the initial uncertainty in our state estimate.
    // High initial values (e.g., identity matrix scaled by 1000) allow the filter
    // to quickly converge to the first few measurements, effectively trusting them more.
    P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;

    x_hat.setZero(); // Initialize the state estimate to all zeros
}

/**
 * @brief Initializes the Kalman filter state and covariance for a new track.
 * @param x Initial x position (center of bounding box).
 * @param y Initial y position (center of bounding box).
 */
void KalmanFilter2D::init(double x, double y) {
    // Initialize state with detected position and zero initial velocity
    x_hat << x, y, 0, 0;
    // Reset uncertainty to a high value, so the first few updates are heavily weighted by measurements
    P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
}

/**
 * @brief Predicts the next state of the tracked object.
 *
 * Uses the state transition matrix `A` to project the current state estimate
 * forward in time, and updates the error covariance matrix `P`.
 */
void KalmanFilter2D::predict() {
    x_hat = A * x_hat;
    P = A * P * A.transpose() + Q;
}

/**
 * @brief Updates the state of the tracked object using a new measurement.
 * @param measured_x Measured x position (center of bounding box).
 * @param measured_y Measured y position (center of bounding box).
 *
 * This function calculates the Kalman gain and uses it to correct the predicted state
 * based on the new measurement, reducing uncertainty.
 */
void KalmanFilter2D::update(double measured_x, double measured_y) {
    Eigen::VectorXd z(2); // Measurement vector
    z << measured_x, measured_y;

    Eigen::VectorXd y_tilde = z - H * x_hat;           // Innovation (measurement residual)
    Eigen::MatrixXd S = H * P * H.transpose() + R;     // Innovation covariance
    Eigen::MatrixXd K = P * H.transpose() * S.inverse(); // Kalman gain

    x_hat = x_hat + K * y_tilde;
    P = (Eigen::MatrixXd::Identity(4, 4) - K * H) * P;
}