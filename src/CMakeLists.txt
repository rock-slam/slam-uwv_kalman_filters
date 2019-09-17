rock_library(uwv_kalman_filters
    SOURCES VelocityUKF.cpp
            PoseUKF.cpp
            BottomUKF.cpp
    HEADERS VelocityUKF.hpp
            PoseUKF.hpp
            PoseState.hpp
            PoseUKFConfig.hpp
            BottomUKF.hpp
    DEPS_PKGCONFIG pose_estimation
                   uwv_dynamic_model
                   eigen3
                   base-types
                   base-logging
                   dynamic_model_svr
    DEPS_CMAKE LAPACK)
