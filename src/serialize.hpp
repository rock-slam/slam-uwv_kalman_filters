#ifndef _SERIALIZE_HPP_
#define _SERIALIZE_HPP_

#include <iostream>
#include <dynamic_model_svr/SVR.hpp>
#include "Utils.hpp"
#include <time.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, dynamic_model_svr::SVR::SVRParams & svrparams, const unsigned int version){
    ar & svrparams.svm_type;
    ar & svrparams.kernel;
    ar & svrparams.degree;
    ar & svrparams.gamma;
    ar & svrparams.coef0;
    ar & svrparams.tol;
    ar & svrparams.C;
    ar & svrparams.nu;
    ar & svrparams.epsilon;
    ar & svrparams.sample_weight;
    ar & svrparams.shrinking;
    ar & svrparams.probability;
    ar & svrparams.cache_size;
    ar & svrparams.max_iter;
    ar & svrparams.random_seed;
}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t,
                      const unsigned int file_version){
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    if( rows * cols != t.size() )
    t.resize( rows, cols );

    for(size_t i=0; i<t.size(); i++)
    ar & t.data()[i];
}


template <class Archive>
void serialize (Archive & ar, dynamic_model_svr::SVR::SVRFitOutput & svrfitoutput, const unsigned int file_version){

  ar & svrfitoutput.support;
  ar & svrfitoutput.support_vectors;
  ar & svrfitoutput.n_class_SV;
  ar & svrfitoutput.sv_coef;
  ar & svrfitoutput.intercept;
  ar & svrfitoutput.probA;
  ar & svrfitoutput.probB;
}
} // namespace serialization
} // namespace boost


//  save the serialized matrix X into a file 
void save (Eigen::MatrixXd & X, std::string file_name){

    std::ofstream ofsM(file_name, std::ofstream::out);
if (ofsM.good()==true){
    boost::archive::text_oarchive oaM(ofsM);
    // write class instance to archive
    oaM << X;
    ofsM.close();
}else {
    std::cerr<<"couldn't save into file"<<std::endl;
}
}

// load the serialized matrix
void load(Eigen::MatrixXd & M_read, std::string   file_name){

   
    std::ifstream ifsM(file_name, std::ifstream::in);
if (ifsM.good() == true) {
    boost::archive::text_iarchive iaM(ifsM);
    // read class state from archive
    iaM >> M_read;
    ifsM.close();
}else {
std::cerr<< "couldn't open the file\n";
}
}
// overloading operator << 


// save the serialized matrix
void save (dynamic_model_svr::SVR::SVRParams &params, std::string file_name){

    std::ofstream ofsParam(file_name, std::ofstream::out);
if (ofsParam.good()==true){
    boost::archive::text_oarchive oaParam(ofsParam);
    // write class instance to archive
    oaParam << params;
    ofsParam.close();
}else {
std::cerr<< "Eror saving the the serialized parameters"<<std::endl;
}
}

// load the serialized parameters from the file
void load (dynamic_model_svr::SVR::SVRParams & params_read, std::string  file_name){
    std::ifstream  ifsParam ( file_name, std::ifstream::in);
if ( ifsParam.good() == true) {
    boost::archive::text_iarchive iaParam(ifsParam);
    // read class state from archive
    iaParam >> params_read;
    ifsParam.close();
}else {
   std::cerr<< "Error loading parameters from the file" << std::endl;
}
}
// saving the serialized fitoutput into a file

void save (dynamic_model_svr::SVR::SVRFitOutput &out, std::string  file_name){

std::ofstream ofsfitop(file_name, std::ofstream::out);
if (ofsfitop.good() == true) {
    boost::archive::text_oarchive oafitoutput(ofsfitop);
    // write class instance to archive
    oafitoutput << out;
    ofsfitop.close();
}else {
std:: cerr<< "Error saving the fitoutput"<<std::endl;
}
}


// loading the fitoutput from a file
void load (dynamic_model_svr::SVR::SVRFitOutput & FOutput_read,std::string  file_name  ){
    std::ifstream  ifsfitop ( file_name, std::ifstream::in);
if (ifsfitop.good() == true ) { 
   boost::archive::text_iarchive iafitoutput(ifsfitop);
    // read class state from archive
    iafitoutput >> FOutput_read;
    ifsfitop.close();
}else {

std::cerr <<"Error loading fitoutput"<<std::endl;
}
}
#endif
