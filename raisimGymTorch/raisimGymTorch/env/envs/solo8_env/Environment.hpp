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
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(3.0);  //3.0 for biped
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

    // sphere_ = world_->addSphere(0.4, 0.8);
    // sphere_->setAppearance("1,0,0,1");
    // sphere_->setPosition(0.0, 0.0, 0.4);
    // sphere_->setBodyType(raisim::BodyType::STATIC);
    // box_ = world_->addBox(0.5, 0.3, 0.3, 2);
    // box_->setAppearance("1,0,0,1");
    // box_->setPosition(0.7, 0.0, 0.15);
    // box_->setBodyType(raisim::BodyType::STATIC);


    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(solo8_);
    }
  }

  void init() final { }

  void reset() final {
    speed = 0.5;//(double(rand() % 8) - 2.0) / 10.0;
    mode_ = 0;//rand() % 2;
    
    // get up mode
    // if (mode_ == 0) {
    //   phase_ = 0;//rand() % (max_phase_/4);
    //   max_phase_ = 60;
    //   sim_step_ = 0;
    //   total_reward_ = 0;
    //   gv_init_[0] = speed;
    //   gv_init_[4] = -20 * std::sin(3.1415 * phase_ / max_phase_ * 4.0);
    //   setReferenceMotion();
    // }
    // else {
    //   max_phase_ = 30;
    //   phase_ = rand() % max_phase_;
    //   sim_step_ = 0;
    //   total_reward_ = 0;
    //   gv_init_[0] = 0.0;
    //   gv_init_[4] = 0;
    //   setReferenceMotionBipedalMode();
    // }

    //quadrupedal mode
    max_phase_ = 30;
    phase_ = (rand() % 2) * max_phase_ / 2;
    sim_step_ = 0;
    total_reward_ = 0;
    gv_init_[0] = speed;
    gv_init_[2] = 0;//0.75 * std::sin(2.0 * 3.1415 * phase_ / max_phase_);
    setReferenceMotion();
    mode_ = 0;

    solo8_->setState(reference_, gv_init_);
    // sphere_->setPosition(0.0, 0.0, 0.4);
    // sphere_->setVelocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    solo8_->getState(gc_, gv_);
    energy_reward_ = 0;
    // if (mode_ == 0)
    //   setReferenceMotion();
    // else
    //   setReferenceMotionBipedalMode();
    /// action scaling
    pTarget12_ = 2 * action.cast<double>() + gc_.tail(8);
    pTarget12_[2] = pTarget12_[0] * 1.0;
    pTarget12_[3] = pTarget12_[1] * 1.0;
    pTarget12_[6] = pTarget12_[4] * 1.0;
    pTarget12_[7] = pTarget12_[5] * 1.0;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    // pTarget12_ += reference_.tail(nJoints_);
    pTarget_.tail(nJoints_) = pTarget12_;

    // solo8_->setState(reference_, gv_init_);

    solo8_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      energy_reward_ += solo8_->getGeneralizedForce().squaredNorm();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    phase_ += 1;
    sim_step_ += 1;
    // if (phase_ > max_phase_ / 4 && mode_ == 0){
    //   phase_ = 0;
    //   mode_ = 1;
    //   max_phase_ = 30;
    // }
    if (phase_ > max_phase_) {
      phase_ = 0;
    }

    updateObservation();

    // rewards_.record("torque", solo8_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::exp(10.0*-std::pow(bodyLinearVel_[0]-0.5, 2.0)));
    computeReward();
    total_reward_ += rewards_.sum();

    if (speed < 0.5)
      speed += 0.01;

    return rewards_.sum();
  }

  void computeReward() {
    float joint_reward = 0, position_reward = 0, orientation_reward = 0;
    for (int j = 0; j < 4; j++) {
      joint_reward += std::pow(gc_[7+j*2]-reference_[7+j*2], 2) + std::pow(gc_[8+j*2]-reference_[8+j*2], 2);
    }
    // position_reward += 10.0 * std::pow(gv_[0]-speed, 2) + std::pow(gc_[1]-reference_[1], 2) + 100 * std::pow(gc_[2]-reference_[2], 2);
    orientation_reward += 2 * (std::pow(gc_[4]-reference_[4], 2) + std::pow(gc_[5]-reference_[5], 2) + std::pow(gc_[6]-reference_[6], 2));
    orientation_reward += 5 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2) + std::pow(gv_[5], 2));

    //contact base reward
    float contact_reward = 0.0;
    for(auto& contact: solo8_->getContacts()) {
      if (phase_ <= max_phase_ / 8 * 3 && phase_ >= max_phase_ / 8 * 1) {
        if(solo8_->getBodyIdx("FL_LOWER_LEG") == contact.getlocalBodyIndex() || solo8_->getBodyIdx("FR_LOWER_LEG") == contact.getlocalBodyIndex()) {
          contact_reward += 0.1 * std::pow(contact.getImpulse()[2], 2);
          raisim::Vec<3> vel1, vel2;
          solo8_->getFrameVelocity(solo8_->getFrameIdxByName("HL_ANKLE"), vel1);
          solo8_->getFrameVelocity(solo8_->getFrameIdxByName("HR_ANKLE"), vel2);
          contact_reward += 0.01 * std::pow(vel1[0], 2);
          contact_reward += 0.01 * std::pow(vel2[0], 2);
        }
      }
      else if (phase_ <= max_phase_ / 8 * 7 && phase_ >= max_phase_ / 8 * 5) {
        if(solo8_->getBodyIdx("HL_LOWER_LEG") == contact.getlocalBodyIndex() || solo8_->getBodyIdx("HR_LOWER_LEG") == contact.getlocalBodyIndex()) {
          contact_reward += 0.1 * std::pow(contact.getImpulse()[2], 2);
          raisim::Vec<3> vel1, vel2;
          solo8_->getFrameVelocity(solo8_->getFrameIdxByName("FL_ANKLE"), vel1);
          solo8_->getFrameVelocity(solo8_->getFrameIdxByName("FR_ANKLE"), vel2);
          contact_reward += 0.01 * std::pow(vel1[0], 2);
          contact_reward += 0.01 * std::pow(vel2[0], 2);
        }
      }
    }

    position_reward = 5.0 * std::pow(gv_[0]-speed, 2) + std::pow(gc_[1]-reference_[1], 2) + 0.0 * std::pow(gv_[2]-0.0, 2);
    // std::cout << gv_[2]-0.0 << std::endl;
    // orientation_reward = 2 * (std::pow(gv_[3]-0.0, 2) + std::pow(gv_[4]-0.0, 2) + std::pow(gv_[5]-0.0, 2)); 
    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-orientation_reward));
    rewards_.record("joint", std::exp(-3*joint_reward));
    rewards_.record("torque", energy_reward_);
    rewards_.record("contact", contact_reward);
  }

  void updateObservation() {
    solo8_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    raisim::Vec<3> ball_vel1, ball_vel2;
    // sphere_->getVelocity(ball_vel1, ball_vel2);

    obDouble_ << gc_[2], /// body height
        gc_[3], gc_[4], gc_[5], gc_[6], /// body orientation
        gc_.tail(8), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(8), /// joint velocity
        std::cos(phase_ * 3.1415 * 2 / max_phase_), std::sin(phase_ * 3.1415 * 2 / max_phase_), // phase
        speed, //speed
        mode_;
        // ball_vel2[0], ball_vel2[1], ball_vel2[2];
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
    if (std::abs(gc_[4]) > 0.2 || std::abs(gc_[6]) > 0.2 || std::abs(gc_[5]) > 0.2 || gc_[2] < 0.2)// || std::abs(gc_[5] + 0.7071068) > 0.3)// || std::abs(gc_[4]) > 0.5 || std::abs(gc_[5]) > 0.5 || std::abs(gc_[6]) > 0.5 || rewards_.sum() < 0.5)
      return true;
    if (mode_ == 1 && gc_[2] < 0.3)
      return true;

    terminalReward = 0.f;
    return false;
  }

  void setReferenceMotion() {
    // get up mode
    // reference_ *= 0;
    // reference_[0] = speed * 0.02 * sim_step_;

    // raisim::Vec<4> quat;
    // raisim::Vec<3> euler;
    // euler[0] = 0;
    // euler[2] = 0;
    // euler[1] = std::max(-std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, -3.1415 / 2);
    // raisim::eulerVecToQuat(euler, quat);
    
    // reference_[3] = quat[0];
    // reference_[4] = quat[1];
    // reference_[5] = quat[2];
    // reference_[6] = quat[3];
    
    // reference_[2] = 0.2 + 0.32 * std::sin(2.0*3.1415*phase_/max_phase_) + 0.3;
    // reference_[0] = 0.7;

    // reference_[7+4] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);
    // reference_[7+6] = 0.65 + 0.92 * std::sin(2.0*3.1415*phase_/max_phase_);

    // reference_[7] = 0.65; 
    // reference_[8] = -2.0 + 2.0 * std::min(std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, 3.1415 / 2);
    // reference_[9] = 0.65;
    // reference_[10] = -2.0 + 2.0 * std::min(std::sin(2.0*3.1415*std::min(phase_*1.0, 2.0 * max_phase_/4.0)/max_phase_/2.0) * 3.1415, 3.1415 / 2);

    // reference_[12] = -2.0 + std::sin(2.0*3.1415*phase_/max_phase_);
    // reference_[14] = -2.0 + std::sin(2.0*3.1415*phase_/max_phase_);
    
    //quadrupedal_mode
    reference_ *= 0;
    reference_[0] = speed * 0.02 * sim_step_;
    reference_[2] = 0.3; //+ 0.3 * std::sin(3.1415 * phase_ / max_phase_); for ponking
    reference_[3] = 1.0;
    reference_[4] = 0.0;
    reference_[5] = 0.0;
    reference_[6] = 0.0;
    for (int i = 0; i < 4; i++) {
      if (phase_ <= max_phase_) {
        if (i > 4) {
          reference_[7+i*2] = 0.65;
          reference_[8+i*2] = -1.0;
        }
        else {
          reference_[7+i*2] = 0.65;// - 0.2 * speed * std::sin(3.1415*phase_/max_phase_);
          reference_[8+i*2] = -1.0;// - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);//-2.0 + 1.5 * std::sin(3.1415*phase_/max_phase_);
        }
      }
      else{
        if (i == 0 || i == 1 || i == 2 || i == 3) {
          reference_[7+i*2] = 0.65 - 0.2 * speed * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
          reference_[8+i*2] = -1.0 - 0.7 * std::sin(2.0*3.1415*phase_/max_phase_ - 3.1415);
        }
        else {
          reference_[7+i*2] = 0.65;
          reference_[8+i*2] = -1.0;
        }
      }
    }
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
  // raisim::Sphere* sphere_;
  raisim::Box* box_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd reference_;
  int phase_ = 0;
  int max_phase_ = 80;
  int sim_step_ = 0;
  int max_sim_step_ = 300;
  double total_reward_ = 0;
  double terminalRewardCoeff_ = 0.;
  double speed = 0.0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int mode_ = 0;
  float energy_reward_ = 0;
};
}