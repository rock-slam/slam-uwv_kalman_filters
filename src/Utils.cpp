#include "Utils.hpp"


int count_rows(std::string file)
{
    std::ifstream in;
    in.open(file.c_str(),std::ios::in);
    std::string line;
    int rows = 0;

    if (in.is_open()) {
        while (std::getline(in, line))
        {
            rows++;
        }
        in.close();
    }
    else
    {
        std::cerr << "Failed to open file: " << file << std::endl;
        std::cerr << "Only comma-separated csv files are supported" << std::endl;
    }

    if (rows <= 0)
    {
        std::cerr << "File is empty!" << std::endl;
    }
    return rows;
}


int count_columns(std::string file)
{
    std::ifstream in;
    in.open(file.c_str(),std::ios::in);
    std::string line;
    int cols = 1;
    if (in.is_open()) {
        std::getline(in, line);
        char *ptr = (char *)line.c_str();
        int len = line.length();
        for (int i = 0; i < len; i++)
        {
            if (ptr[i] == ',')
            {
                cols++;
            }
        }
        in.close();
    }
    else
    {
        std::cerr << "Failed to open file: " << file << std::endl;
        std::cerr << "Only comma-separated csv files are supported" << std::endl;
    }

    if (cols <= 0)
    {
        std::cerr <<"The file contains no columns" << std::endl;
    }
    return cols;
}


Eigen::MatrixXd readCSV(std::string file) {
    int rows = count_rows(file);
    int cols = count_columns(file);

    std::ifstream in(file);

    std::string line;

    int row = 0;
    int col = 0;

    Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

    if (in.is_open()) {

    while (std::getline(in, line)) {
      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }
    in.close();
    }
    return res;
}



Scaler::Scaler()
{}


void Scaler::fit(const Eigen::ArrayXXd &data)
{
    scalerParams.means = data.colwise().mean(); //columnwise mean
    scalerParams.std_devs = ((data.rowwise() - data.colwise().mean()).square().colwise().sum()/data.rows()).sqrt(); //columnwise std_dev
}


Eigen::ArrayXXd Scaler::transform(const Eigen::ArrayXXd &data)
{
    Eigen::ArrayXXd scaledData(data.rows(), data.cols());

    scaledData = (data.rowwise() - scalerParams.means).rowwise()/scalerParams.std_devs;
    return scaledData;
}


Eigen::ArrayXXd Scaler::inverse_transform(const Eigen::ArrayXXd &data)
{
    Eigen::ArrayXXd invData(data.rows(), data.cols());

    invData = (data.rowwise() * scalerParams.std_devs).rowwise() + scalerParams.means;
    return invData;
}


Eigen::MatrixXd mahalanobis_kernel(const Eigen::MatrixXd &X, const Eigen::MatrixXd &s)
{
    int n = X.rows();
    Eigen::ArrayXXd K(n, n);
    for(int i=0;i<n;++i)
    {
        for(int j=i;j<n;++j)
        {
            K(i,j) = K(j,i) = (X.row(i)-X.row(j))*s*(X.row(i)-X.row(j)).transpose();
        }
    }

    K *= -1;
    K=K.exp(); // exponentiate K in-place
    return K;
}



Eigen::MatrixXd mahalanobis_kernel(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &s)
{
    int n = X.rows(), m = Y.rows();
    Eigen::ArrayXXd K(n, m);
    for(int i=0;i<n;++i)
    {
        for(int j=0;j<m;++j)
        {
            K(i,j) = (X.row(i)-Y.row(j))*s*(X.row(i)-Y.row(j)).transpose();
        }
    }

    K *= -1;
    K=K.exp(); // exponentiate K in-place
    return K;
}



double cov(const Eigen::ArrayXd &X, const Eigen::ArrayXd &Y)
{
    return ((X - X.mean())*(Y - Y.mean())).sum()/(X.size()-1);
}


Eigen::MatrixXd cov(const Eigen::MatrixXd &X)
{

    Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = centered.adjoint() * centered/(X.rows()-1); //

    return cov;
}


Eigen::MatrixXd Pinv(const Eigen::MatrixXd &matrix)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix*matrix.transpose(),
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd auxPseudoInverse = svd.solve(
                       Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows()));
    Eigen::MatrixXd pseudoInverse;
    pseudoInverse = matrix.transpose()*auxPseudoInverse;
    return pseudoInverse;
}
/*
    */
