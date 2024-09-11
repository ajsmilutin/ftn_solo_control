#pragma once
#include <pinocchio/multibody/model.hpp>
// FTN solo includes
#include <ftn_solo_control/motions/motion.h>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/friction_cone.h>

namespace ftn_solo_control {

bool GetEndOfMotion(const pinocchio::Model &model, pinocchio::Data &data,
                    const FrictionConeMap &friction_cones,
                    const std::vector<boost::shared_ptr<Motion>> &motions,
                    RefVectorXd q);

}