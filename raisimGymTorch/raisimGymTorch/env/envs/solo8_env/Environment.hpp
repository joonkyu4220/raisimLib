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
      phase_ = 0;//(rand() % 2) * (max_phase_/4);
      max_phase_ = 60;
      sim_step_ = 0;
      total_reward_ = 0;
      gv_init_[0] = speed;
      if (phase_ != 0)
        gv_init_[4] = -5;
      setFlipMotion();
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
    if (sim_step_ < 60)
      setFlipMotion();
    else
      setFlipMotion_v2();
    // if (mode_ == 0)
    //   setFlipMotion();
    // else
    //   setReferenceMotionBipedalMode();
    /// action scaling
    pTarget12_ = action.cast<double>() * 1;
    pTarget12_[2] = pTarget12_[0] * 1.0;
    pTarget12_[3] = pTarget12_[1] * 1.0;
    pTarget12_[6] = pTarget12_[4] * 1.0;
    pTarget12_[7] = pTarget12_[5] * 1.0;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget12_ += reference_.tail(nJoints_);
    pTarget_.tail(nJoints_) = pTarget12_;

    solo8_->setState(reference_, gv_init_);

    solo8_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    phase_ += 1;
    sim_step_ += 1;
    if (phase_ > max_phase_ / 2 && mode_ == 0 && !flip_obs_){
      phase_ = max_phase_ / 2;
      mode_ = 0;
      max_phase_ = 60;
    }
    else if (phase_ > max_phase_ / 2) {
      phase_ = max_phase_ / 2;
    }
    if (sim_step_ == 60) {
      flip_obs_ = true;
      phase_ = 0;
      setFlipMotion_v2();
      solo8_->setState(reference_, gv_init_);
    }
    // if (phase_ > max_phase_ / 4 && mode_ == 0){
    //   phase_ = 0;
    //   mode_ = 1;
    //   max_phase_ = 30;
    // }
    // if (phase_ > max_phase_) {
    //   phase_ = 0;
    // }

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
    // orientation_reward += 5 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2) + std::pow(gv_[5], 2));
    // position_reward = 5 * std::pow(gc_[2]-0.53, 2);
    // orientation_reward = 2 * (std::pow(gc_[4]-0.0, 2) + std::pow(gc_[5]+0.7071068, 2) + std::pow(gc_[6]-0.0, 2)); 

    float contact_reward = 0.0;
    raisim::Vec<3> vel1, vel2;
    solo8_->getFrameVelocity(solo8_->getFrameIdxByName("HL_ANKLE"), vel1);
    solo8_->getFrameVelocity(solo8_->getFrameIdxByName("HR_ANKLE"), vel2);
    contact_reward += 2 * std::pow(vel1[2], 2); //0.01 for bounding
    contact_reward += 2 * std::pow(vel2[2], 2);

    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-orientation_reward));
    rewards_.record("joint", std::exp(-3*joint_reward));
    rewards_.record("torque", solo8_->getGeneralizedForce().squaredNorm());
    rewards_.record("contact", std::exp(-contact_reward));
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
    std::cout << flip_obs_ << std::endl;
    if (flip_obs_) {
      double euler[3];
      double quat_2[4];
      quat_2[0] = gc_[3], quat_2[1] = gc_[4], quat_2[2] = gc_[5], quat_2[3] = gc_[6];
      raisim::quatToEulerVec(quat_2, euler);
      euler[1] += 3.1415;
      raisim::Vec<3> euler_2;
      euler_2[0] = euler[0]; euler_2[1] = euler[1]; euler_2[2] = euler[2];
      raisim::eulerVecToQuat(euler_2, quat);
      obDouble_[1] = quat[0], obDouble_[2] = quat[1], obDouble_[3] = quat[2], obDouble_[4] = quat[3];
      obDouble_[5] = 3.1415 - obDouble_[5], obDouble_[7] = 3.1415 - obDouble_[7], obDouble_[9] = 3.1415 - obDouble_[9], obDouble_[11] = 3.1415 - obDouble_[11];
      obDouble_[6] = -obDouble_[6]; obDouble_[8] = -obDouble_[8]; obDouble_[10] = -obDouble_[10]; obDouble_[12] = -obDouble_[12];
    }
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
    // std::cout << gc_[2]-reference_[2]<<std::endl;
    if (std::abs(gc_[4]) > 0.2 || std::abs(gc_[6]) > 0.2)// || std::abs(gc_[5] + 0.7071068) > 0.3)// || std::abs(gc_[4]) > 0.5 || std::abs(gc_[5]) > 0.5 || std::abs(gc_[6]) > 0.5 || rewards_.sum() < 0.5)
      return true;
    if (mode_ == 1 && gc_[2] < 0.3)
      return true;

    // int counter = 0;
    // int contact_id = 0;
    // raisim::Vec<3> contact_vel;
    // for(auto& contact: solo8_->getContacts()) {
    //   solo8_->getContactPointVel(contact_id, contact_vel);
    //   if(solo8_->getBodyIdx("HL_LOWER_LEG") == contact.getlocalBodyIndex() || solo8_->getBodyIdx("HR_LOWER_LEG") == contact.getlocalBodyIndex()) {
    //     counter += 1;
    //     // if (contact_vel.squaredNorm() > 0.1)
    //     //   return true;
    //   }
    //   contact_id++;
    // }
    // // std::cout << counter << std::endl;
    // if (counter < 2)
    //   return true;

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

  void setFlipMotion_v2() {
    reference_ *= 0;
    reference_[0] = speed * 0.02 * sim_step_;

    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    euler[0] = 0;
    euler[2] = 0;
    if (phase_ < max_phase_ / 4) {
      euler[1] = -std::sin(2.0 * 3.1415*phase_/max_phase_) * 1.57;
      reference_[2] = 0.25 + 0.2 * std::sin(2.0*3.1415*phase_/max_phase_);
    }
    else {
      euler[1] = -1.57 - std::sin(2.0 * 3.1415*(phase_-max_phase_/4)/max_phase_) * 1.57;
      reference_[2] = 0.45 - 0.2 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
    }
    euler[1] = -3.14 * phase_ / max_phase_ * 2;
    raisim::eulerVecToQuat(euler, quat);
    
    reference_[3] = quat[0];
    reference_[4] = quat[1];
    reference_[5] = quat[2];
    reference_[6] = quat[3];

    reference_[12] = -1.5;
    reference_[14] = -1.5;

    if (phase_ < max_phase_ / 4){
      reference_[7+4] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[7+6] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      // reference_[12] = -2.0 + 2.0 * std::sin(2.0*3.1415*phase_/max_phase_);
      // reference_[14] = -2.0 + 2.0 * std::sin(2.0*3.1415*phase_/max_phase_);

      reference_[7] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[9] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[8] = -1.5 + 1.5 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[10] = -1.5 + 1.5 * std::sin(2.0*3.1415*phase_/max_phase_);
    }
    else {
      reference_[7+4] = 1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[7+6] = 1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      // reference_[12] = 0.0 + 2.0 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      // reference_[14] = 0.0 + 2.0 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);

      reference_[7] = 1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[9] = 1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[8] = 0.0 + 1.5 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[10] = 0.0 + 1.5 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
    }
    reference_[11] = 0.65 + 3.79*phase_/max_phase_*2;
    reference_[13] = 0.65 + 3.79*phase_/max_phase_*2;

    if (flip_obs_) {
      double temp1 = reference_[7], temp2 = reference_[9];
      reference_[7] = 3.1415 + reference_[11];
      reference_[9] = 3.1415 + reference_[13];
      reference_[11] = 3.1415 + temp1;
      reference_[13] = 3.1415 + temp2;
      temp1 = reference_[8], temp2 = reference_[10];
      reference_[8] = -reference_[12]; reference_[10] = -reference_[14]; reference_[12] = -temp1; reference_[14] = -temp2;
      euler[1] += 3.14;
      raisim::eulerVecToQuat(euler, quat);
      reference_[3] = quat[0];
      reference_[4] = quat[1];
      reference_[5] = quat[2];
      reference_[6] = quat[3];
    }
  }

  void setFlipMotion() {
    reference_ *= 0;
    reference_[0] = speed * 0.02 * sim_step_;

    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    euler[0] = 0;
    euler[2] = 0;
    if (phase_ < max_phase_ / 4) {
      euler[1] = -std::sin(2.0 * 3.1415*phase_/max_phase_) * 1.57;
      reference_[2] = 0.244 + 0.2 * std::sin(2.0*3.1415*phase_/max_phase_);
    }
    else {
      euler[1] = -1.57 - std::sin(2.0 * 3.1415*(phase_-max_phase_/4)/max_phase_) * 1.57;
      reference_[2] = 0.45 - 0.2 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
    }
    euler[1] = -3.14 * phase_ / max_phase_ * 2;
    raisim::eulerVecToQuat(euler, quat);
    
    reference_[3] = quat[0];
    reference_[4] = quat[1];
    reference_[5] = quat[2];
    reference_[6] = quat[3];

    reference_[12] = 1.5;
    reference_[14] = 1.5;

    if (phase_ < max_phase_ / 4){
      reference_[7+4] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[7+6] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
      // reference_[12] = -2.0 + 2.0 * std::sin(2.0*3.1415*phase_/max_phase_);
      // reference_[14] = -2.0 + 2.0 * std::sin(2.0*3.1415*phase_/max_phase_);

      reference_[7] = 0.65 - 2.22 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[9] = 0.65 - 2.22 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[8] = -1.5 + 1.5 * std::sin(2.0*3.1415*phase_/max_phase_);
      reference_[10] = -1.5 + 1.5 * std::sin(2.0*3.1415*phase_/max_phase_);
    }
    else {
      reference_[7+4] = -1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[7+6] = -1.57 + 0.92 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      // reference_[12] = 0.0 + 2.0 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      // reference_[14] = 0.0 + 2.0 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);

      reference_[7] = -1.57 - 2.22 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[9] = -1.57 - 2.22 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[8] = 0.0 + 1.5 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
      reference_[10] = 0.0 + 1.5 * std::sin(2.0*3.1415*(phase_-max_phase_/4)/max_phase_);
    }
    reference_[11] = -0.65 + (3.79-0.65)*phase_/max_phase_*2;
    reference_[13] = -0.65 + (3.79-0.65)*phase_/max_phase_*2;

    if (flip_obs_) {
      reference_[7] = 3.1415 - reference_[7];
      reference_[9] = 3.1415 - reference_[9];
      reference_[11] = 3.1415 - reference_[11];
      reference_[13] = 3.1415 - reference_[13];
      reference_[8] = -reference_[8]; reference_[10] = -reference_[10]; reference_[12] = -reference_[12]; reference_[14] = -reference_[14];
      euler[1] += 3.14;
      raisim::eulerVecToQuat(euler, quat);
      reference_[3] = quat[0];
      reference_[4] = quat[1];
      reference_[5] = quat[2];
      reference_[6] = quat[3];
    }
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  bool flip_obs_ = false;
  raisim::ArticulatedSystem* solo8_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd reference_;
  int phase_ = 0;
  int max_phase_ = 60;
  int sim_step_ = 0;
  int max_sim_step_ = 10000;
  double total_reward_ = 0;
  double terminalRewardCoeff_ = 0.;
  double speed = 0.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int mode_ = 0;
};
}