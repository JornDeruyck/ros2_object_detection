// include/ros2_object_detection/kalman_filter_2d.hpp
#ifndef KALMAN_FILTER_2D_HPP
#define KALMAN_FILTER_2D_HPP

#include <Eigen/Dense> // For MatrixXd, VectorXd

/**
 * @brief Simple 2D Kalman Filter for constant velocity model.
 * State: [x, y, vx, vy]
 * Measurement: [x, y]
 */
class KalmanFilter2D {
public:
    Eigen::MatrixXd A; // State transition matrix
    Eigen::MatrixXd H; // Measurement matrix
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd R; // Measurement noise covariance
    Eigen::MatrixXd P; // Error covariance matrix
    Eigen::VectorXd x_hat; // State estimate [x, y, vx, vy]

    KalmanFilter2D(); // Constructor initializes matrices

    /**
     * @brief Initializes the filter state and covariance for a new track.
     * @param x Initial x position.
     * @param y Initial y position.
     */
    void init(double x, double y);

    /**
     * @brief Predicts the next state.
     */
    void predict();

    /**
     * @brief Updates the state with a new measurement.
     * @param measured_x Measured x position.
     * @param measured_y Measured y position.
     */
    void update(double measured_x, double measured_y);

    double getX() const { return x_hat(0); }
    double getY() const { return x_hat(1); }
    double getVx() const { return x_hat(2); }
    double getVy() const { return x_hat(3); }
};

#endif // KALMAN_FILTER_2D_HPP