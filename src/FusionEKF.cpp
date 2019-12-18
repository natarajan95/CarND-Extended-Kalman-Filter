#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  prev_t_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,    0.0,
                 0.0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,    0.0,  0.0,
               0.0, 0.0009,  0.0,
               0.0,    0.0, 0.09;

  // measurement matrix
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;

  // measurement noises
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() = default;

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  if (!is_initialized_) {
    // initialization code


    VectorXd x = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

      // convert polar from polar to cartesian coordinate system
      double meas_rho     = measurement_pack.raw_measurements_[0];
      double meas_phi     = measurement_pack.raw_measurements_[1];
      double meas_px      = meas_rho     * cos(meas_phi);
      double meas_py      = meas_rho     * sin(meas_phi);

      // initial state in case the first measurement comes from radar sensor
      x << meas_px,
           meas_py,
                 0, 
                 0; 
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initial state in case the first measurement comes from lidar sensor
      x << measurement_pack.raw_measurements_[0],
           measurement_pack.raw_measurements_[1],
                                             0.0, 
                                             0.0;
    } else {
      cerr << "unknown sensor type of measurement; skipping initialization" << endl;
      return;
    }

    // state covariance matrix
    MatrixXd P(4, 4);
    P << 1, 0,    0,    0,
         0, 1,    0,    0,
         0, 0, 1000,    0,
         0, 0,    0, 1000;

    // state transition matrix
    MatrixXd F(4, 4);
    F << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    // process covariance matrix
    MatrixXd Q(4, 4);

    ekf_.Init(x, P, F, H_laser_, R_laser_, R_radar_, Q);

    prev_t_ = measurement_pack.timestamp_;

    is_initialized_ = true;
    return;
  }

  double dt = (measurement_pack.timestamp_ - prev_t_) / 1000000.0;
  prev_t_ = measurement_pack.timestamp_;

  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // update the process covariance matrix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt*dt*dt*dt/4*noise_ax_,              0.0,    dt*dt*dt/2*noise_ax_,              0.0,
                           0.0, dt*dt*dt*dt/4*noise_ay_,              0.0,       dt*dt*dt/2*noise_ay_,
              dt*dt*dt/2*noise_ax_,              0.0,    dt*dt*noise_ax_,              0.0,
                           0.0, dt*dt*dt/2*noise_ay_,              0.0,       dt*dt*noise_ay_;

  // prediction step
  ekf_.Predict();

  // update step
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Lidar updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}