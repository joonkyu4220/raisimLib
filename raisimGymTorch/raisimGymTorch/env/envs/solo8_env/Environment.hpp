#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    solo8_ = world_->addArticulatedSystem(resourceDir_+"/solo8_URDF_v6/solo8.urdf");
    solo8_->setName("solo8");
    solo8_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = solo8_->getGeneralizedCoordinateDim();
    gvDim_ = solo8_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    reference_.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0;
    reference_ << 0, 0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(3.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    solo8_->setPdGains(jointPgain, jointDgain);
    solo8_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 31;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(1);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(solo8_->getBodyIdx("base_link"));
    // footIndices_.insert(solo8_->getBodyIdx("FR_FOOT"));
    // footIndices_.insert(solo8_->getBodyIdx("HL_FOOT"));
    // footIndices_.insert(solo8_->getBodyIdx("HR_FOOT"));


    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(solo8_);
    }
  }

  void init() final { }

  void reset() final {
    speed = 0.0;//(double(rand() % 8) - 2.0) / 10.0;
    mode_ = 0;//rand() % 2;
    if (mode_ == 0) {
      phase_ = 0;//rand() % (max_phase_/4);
      max_phase_ = 60;
      sim_step_ = 0;
      total_reward_ = 0;
      gv_init_[0] = speed;
      gv_init_[4] = -20 * std::sin(3.1415 * phase_ / max_phase_ * 4.0);
      setReferenceMotion();
    }
    else {
      max_phase_ = 30;
      phase_ = rand() % max_phase_;
      sim_step_ = 0;
      total_reward_ = 0;
      gv_init_[0] = speed;
      gv_init_[4] = 0;
      setReferenceMotionBipedalMode();
    }
    solo8_->setState(reference_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    if (mode_ == 0)
      setReferenceMotion();
    else
      setReferenceMotionBipedalMode();
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget12_ += reference_.tail(nJoints_);
    pTarget_.tail(nJoints_) = pTarget12_;

    // solo8_->setState(reference_, gv_init_);

    solo8_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    phase_ += 1;
    sim_step_ += 1;
    if (phase_ > max_phase_ / 4 && mode_ == 0){
      phase_ = 0;
      mode_ = 1;
      max_phase_ = 30;
    }
    if (phase_ > max_phase_) {
      phase_ = 0;
    }

    updateObservation();

    // rewards_.record("torque", solo8_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::exp(10.0*-std::pow(bodyLinearVel_[0]-0.5, 2.0)));
    computeReward();
    total_reward_ += rewards_.sum();

    return rewards_.sum();
  }

  void computeReward() {
    float joint_reward = 0, position_reward = 0, orientation_reward = 0;
    for (int j = 0; j < 4; j++) {
      joint_reward += std::pow(gc_[7+j*2]-reference_[7+j*2], 2) + std::pow(gc_[8+j*2]-reference_[8+j*2], 2);
    }
    position_reward += 1.0 * std::pow(gv_[0]-speed, 2) + std::pow(gc_[1]-reference_[1], 2) + std::pow(gc_[2]-reference_[2], 2);
    orientation_reward += 2 * (std::pow(gc_[4]-reference_[4], 2) + std::pow(gc_[5]-reference_[5], 2) + std::pow(gc_[6]-reference_[6], 2));
    orientation_reward += 5 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2) + std::pow(gv_[5], 2));
    // position_reward = 5 * std::pow(gc_[2]-0.53, 2);
    // orientation_reward = 2 * (std::pow(gc_[4]-0.0, 2) + std::pow(gc_[5]+0.7071068, 2) + std::pow(gc_[6]-0.0, 2)); 
    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-orientation_reward));
    rewards_.record("joint", std::exp(-3*joint_reward));
    rewards_.record("torque", solo8_->getGeneralizedForce().squaredNorm());
  }

  void updateObservation() {
    solo8_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        gc_[3], gc_[4], gc_[5], gc_[6], /// body orientation
        gc_.tail(8), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(8), /// joint velocity
        std::cos(phase_ * 3.1415 * 2 / max_phase_), std::sin(phase_ * 3.1415 * 2 / max_phase_), // phase
        speed, //speed
        mode_;
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

    /// if the contact body is not feet
    // for(auto& contact: solo8_->getContacts())
    //   if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
    //     return true;
    solo8_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    if (std::abs(gc_[4]) > 0.2 || std::abs(gc_[6]) > 0.2)// || std::abs(gc_[5] + 0.7071068) > 0.3)// || std::abs(gc_[4]) > 0.5 || std::abs(gc_[5]) > 0.5 || std::abs(gc_[6]) > 0.5 || rewards_.sum() < 0.5)
      return true;
    if (mode_ == 1 && gc_[2] < 0.3)
      return true;

    terminalReward = 0.f;
    return false;
  }

  void setReferenceMotion() {
    reference_ *= 0;
    reference_[0] = speed * 0.02 * sim_step_;

    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    euler[0] = 0;
    euler[2] = 0;
    euler[1] = std::max(-std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, -3.1415 / 2);
    raisim::eulerVecToQuat(euler, quat);
    
    reference_[3] = quat[0];
    reference_[4] = quat[1];
    reference_[5] = quat[2];
    reference_[6] = quat[3];
    
    reference_[2] = 0.2 + 0.32 * std::sin(2.0*3.1415*phase_/max_phase_);

    reference_[7+4] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
    reference_[7+6] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);

    reference_[7] = 0.65; 
    reference_[8] = -2.0 + 2.0 * std::min(std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, 3.1415 / 2);
    reference_[9] = 0.65;
    reference_[10] = -2.0 + 2.0 * std::min(std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, 3.1415 / 2);

    reference_[12] = -2.0 + std::sin(2.0*3.1415*phase_/max_phase_);
    reference_[14] = -2.0 + std::sin(2.0*3.1415*phase_/max_phase_);
    //quadrupedal_mode
    // for (int i = 0; i < 4; i++) {
    //   if (phase_ <= max_phase_ / 2) {
    //     if (i == 0 || i == 2) {
    //       reference_[7+i*2] = 0.65;
    //       reference_[8+i*2] = -1.0;
    //     }
    //     else {
    //       reference_[7+i*2] = 0.65 - 0.2 * speed * std::sin(2.0*3.1415*phase_/max_phase_);
    //       reference_[8+i*2] = -1.0 - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_);
    //     }
    //   }
    //   else{
    //     if (i == 0 or i == 2) {
    //       reference_[7+i*2] = 0.65 - 0.2 * speed * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
    //       reference_[8+i*2] = -1.0 - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
    //     }
    //     else {
    //       reference_[7+i*2] = 0.65;
    //       reference_[8+i*2] = -1.0;
    //     }
    //   }
    // }
  }

  void setReferenceMotionBipedalMode() {
    reference_[0] = speed * 0.02 * sim_step_;
    reference_[2] = 0.53;
    reference_[3] = 0.7071068;
    reference_[5] = -0.7071068;
    reference_[4] = 0.0;
    reference_[6] = 0.0;
    for (int i = 0; i < 4; i++) {
      if (i == 0 || i == 1) {
        reference_[7+i*2] = 0.0;
        reference_[8+i*2] = 0.0;
      }
      else {
        reference_[7+i*2] = 1.57;
        reference_[8+i*2] = 0.0;
      }
    }
    if (phase_ <= max_phase_ / 2) {
      reference_[7+4] = 1.57 - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[8+4] = 0.7 * std::sin(2.0*3.1415*phase_/max_phase_);
    }
    else {
      reference_[7+6] = 1.57 - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
      reference_[8+6] = 0.7 * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
    }
    reference_[7] = 1.57 + std::sin(2.0*3.1415*phase_/max_phase_);
    reference_[9] = 1.57 + std::sin(2.0*3.1415*phase_/max_phase_ + 3.1415);
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* solo8_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd reference_;
  int phase_ = 0;
  int max_phase_ = 60;
  int sim_step_ = 0;
  int max_sim_step_ = 300;
  double total_reward_ = 0;
  double terminalRewardCoeff_ = 0.;
  double speed = 0.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int mode_ = 0;
};
}