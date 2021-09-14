#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    walker_ = world_->addArticulatedSystem(resourceDir_+"/2dwalker.urdf");
    walker_->setName("walker");
    walker_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    world_->setMaterialPairProp("default", "ball", 1.0, 0.8, 0.01);
    soccer_ = world_->addArticulatedSystem(resourceDir_+"/ball2D.urdf");
    ball_gc_init_.setZero(3);
    ball_gv_init_.setZero(3);
    ball_gc_.setZero(3);
    ball_gv_.setZero(3);
    ball_gc_init_[0] = 0.2;
    ball_gc_init_[1] = 0.15;
    ball_reference_.setZero(3);
    ball_reference_vel_.setZero(3);
    soccer_->setState(ball_gc_init_, ball_gv_init_);

    /// get robot data
    gcDim_ = walker_->getGeneralizedCoordinateDim();
    gvDim_ = walker_->getDOF();
    nJoints_ = gvDim_ - 3;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    reference_.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0;
    reference_ << 0, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(5.0);
    walker_->setPdGains(jointPgain, jointDgain);
    walker_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 25;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(1);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    // footIndices_.insert(walker_->getBodyIdx("base_link"));
    // footIndices_.insert(walker_->getBodyIdx("FR_FOOT"));
    // footIndices_.insert(walker_->getBodyIdx("HL_FOOT"));
    // footIndices_.insert(walker_->getBodyIdx("HR_FOOT"));


    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(walker_);
    }
  }

  void init() final { }

  void reset() final {
    flip_obs_ = false;
    speed = 0.0;//(double(rand() % 8) - 2.0) / 10.0;
    phase_ = 0;//rand() % max_phase_;
    sim_step_ = 0;
    total_reward_ = 0;
    setReferenceMotion();
    setBallReference();
    walker_->setState(reference_, gv_init_);
    soccer_->setState(ball_reference_, ball_reference_vel_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    setReferenceMotion();
    setBallReference();
    pTarget12_ = action.cast<double>() * 1;

    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget12_ += reference_.tail(nJoints_);
    pTarget_.tail(nJoints_) = pTarget12_;

    // walker_->setState(reference_, gv_init_);
    // soccer_->setState(ball_reference_, ball_reference_vel_);

    walker_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    phase_ += 1;
    sim_step_ += 1;
    if (phase_ > max_phase_){
      phase_ = 0;
    }

    updateObservation();

    // rewards_.record("torque", walker_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::exp(10.0*-std::pow(bodyLinearVel_[0]-0.5, 2.0)));
    computeReward();
    total_reward_ += rewards_.sum();

    return rewards_.sum();
  }

  void computeReward() {
    float joint_reward = 0, position_reward = 0, orientation_reward = 0;
    for (int j = 0; j < 2; j++) {
      joint_reward += std::pow(gc_[3+j*3]-reference_[3+j*3], 2) + std::pow(gc_[4+j*3]-reference_[4+j*3], 2);
    }
    position_reward += 1.0 * std::pow(gv_[0]-speed, 2) + std::pow(gc_[1]-reference_[1], 2);
    orientation_reward += 2 * (std::pow(gc_[2]-reference_[2], 2));
    // orientation_reward += 5 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2) + std::pow(gv_[5], 2));
    // position_reward = 5 * std::pow(gc_[2]-0.53, 2);
    // orientation_reward = 2 * (std::pow(gc_[4]-0.0, 2) + std::pow(gc_[5]+0.7071068, 2) + std::pow(gc_[6]-0.0, 2)); 

    // float contact_reward = 0.0;
    // raisim::Vec<3> vel1, vel2;
    // walker_->getFrameVelocity(walker_->getFrameIdxByName("HL_ANKLE"), vel1);
    // walker_->getFrameVelocity(walker_->getFrameIdxByName("HR_ANKLE"), vel2);
    // contact_reward += 2 * std::pow(vel1[2], 2); //0.01 for bounding
    // contact_reward += 2 * std::pow(vel2[2], 2);

    float ball_position_reward = 0;
    ball_position_reward += std::pow(ball_reference_[0] - ball_gc_[0], 2) + std::pow(ball_reference_[1] - ball_gc_[1], 2);

    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-orientation_reward));
    rewards_.record("joint", std::exp(-3*joint_reward));
    rewards_.record("ball position", std::exp(-ball_position_reward));
    // rewards_.record("torque", walker_->getGeneralizedForce().squaredNorm());
    // rewards_.record("contact", std::exp(-contact_reward));
  }

  void updateObservation() {
    soccer_->getState(ball_gc_, ball_gv_);
    walker_->getState(gc_, gv_);
    obDouble_ << gc_[1], /// body height
        gc_[2], /// body orientation
        gc_.tail(6), /// joint angles
        gv_[0], gv_[1], gv_[2], /// body linear&angular velocity
        gv_.tail(6), /// joint velocity
        std::cos(phase_ * 3.1415 * 2 / max_phase_), std::sin(phase_ * 3.1415 * 2 / max_phase_), // phase
        speed, //speed
        ball_gc_[0] - gc_[0], ball_gc_[1] - gc_[1], 
        ball_gv_[0], ball_gv_[1], ball_gv_[2];
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_) * 0.0f;

    // std::cout << gc_[2]-reference_[2]<<std::endl;
    if (std::abs(gc_[2]) > 0.5)
      return true;
    if (std::abs(gc_[1]) < 0.6)
      return true;
    if (std::abs(ball_gc_[0] - gc_[0] - ball_reference_[0]) > 0.1)
      return true;

    terminalReward = 0.f;
    return false;
  }

  void setReferenceMotion() {
    reference_ *= 0;
    reference_[0] = speed * 0.02 * sim_step_;
    reference_[1] = 1.3;
    //kicking with one foot
    reference_[3] = -1.57 * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);
    reference_[4] = 1.57 * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);
  }

  void setBallReference() {
    ball_reference_ *= 0;
    ball_reference_[0] = ball_gc_init_[0];
    float init_vel = g_ * max_phase_ * control_dt_ / 2;
    float init_height = 1;
    float max_height = 0.5 * g_ * std::pow(max_phase_ / 2 * control_dt_, 2) + init_height;
    if (phase_ > max_phase_ / 2) {
      float cur_vel = init_vel - g_ * (phase_ - max_phase_ / 2) * control_dt_;
      ball_reference_[1] = (init_vel + cur_vel) / 2.0 * control_dt_ * (phase_ - max_phase_/2) + init_height;
      ball_reference_vel_[1] = cur_vel;
    }
    else {
      ball_reference_[1] = max_height - 0.5 * g_ * std::pow((phase_)*control_dt_, 2);
      ball_reference_vel_[1] = -g_ * (phase_)*control_dt_;
    }
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  bool flip_obs_ = false;
  raisim::ArticulatedSystem* walker_;
  raisim::ArticulatedSystem* soccer_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd ball_gc_init_, ball_gv_init_, ball_gc_, ball_gv_;
  Eigen::VectorXd reference_;
  Eigen::VectorXd ball_reference_, ball_reference_vel_;
  int phase_ = 0;
  int max_phase_ = 60;
  int sim_step_ = 0;
  int max_sim_step_ = 1000;
  double total_reward_ = 0;
  double terminalRewardCoeff_ = 0.;
  double speed = 0.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int mode_ = 0;
  float g_ = 9.81;
};
}