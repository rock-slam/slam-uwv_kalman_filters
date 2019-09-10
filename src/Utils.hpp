#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>


int count_rows(std::string file);

int count_columns(std::string file);

Eigen::MatrixXd readCSV(std::string file);


class Scaler
{
    public:
        Scaler();

        struct scale_params
        {
            Eigen::Array<double, 1, Eigen::Dynamic> means;
            Eigen::Array<double, 1, Eigen::Dynamic> std_devs;
        };

        /** */
        void fit(const Eigen::ArrayXXd &data);

        /** */
        Eigen::ArrayXXd transform(const Eigen::ArrayXXd &data);

        /** */
        Eigen::ArrayXXd inverse_transform(const Eigen::ArrayXXd &data);
    private:
        scale_params scalerParams;
};


Eigen::MatrixXd mahalanobis_kernel(const Eigen::MatrixXd &X, const Eigen::MatrixXd &s);
Eigen::MatrixXd mahalanobis_kernel(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &s);

double cov(const Eigen::ArrayXd &X, const Eigen::ArrayXd &Y);

Eigen::MatrixXd cov(const Eigen::MatrixXd &X);


Eigen::MatrixXd Pinv(const Eigen::MatrixXd &matrix);
#endif
