#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() = default;

Tools::Tools(const Tools &tools) = default;

Tools::~Tools() = default;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if(estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    return rmse;
  }

  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Fn from assignment
  MatrixXd Hj(3,4);
  //recover state parameters
  const double & px = x_state(0);
  const double & py = x_state(1);
  const double & vx = x_state(2);
  const double & vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  double c1 = px * px + py * py;
  double c2 = sqrt(c1);
  double c3 = (c1 * c2);

  if(fabs(c1) < 0.00001){
    return Hj;
  }

  //Jacobian matrix H
  Hj <<  (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

VectorXd Tools::CarttoPol(const VectorXd &v_cart){
  const double & px = v_cart(0);
  const double & py = v_cart(1);
  const double & vx = v_cart(2);
  const double & vy = v_cart(3);

  // out var to store result
  VectorXd polar_out = VectorXd(3);  
  
  double rho, phi, rho_dot;
  rho = sqrt(px*px + py*py);
  phi = atan2(py, px);

  if (rho < 0.00001) {
    rho = 0.00001;
  }

  rho_dot = (px*vx + py*vy) / rho;

  polar_out << rho, phi, rho_dot;

  return polar_out;

}