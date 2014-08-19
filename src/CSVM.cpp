/*
 * CSVM.cpp
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */

#include "CSVM.h"
#include "libSVMWrapper.h"
#include <iostream>
#include <cmath>

#define INF HUGE_VAL
#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif

extern Settings InputSettings;

/// <summary>
/// scale features and compute distances for kernel matrix
/// </summary>
/// <param name="Data">unscaled features</param>
/// <param name="Label">labels</pa
CSVM::CSVM(Eigen::MatrixXd Data, Eigen::VectorXd Label, double Regularisation = 0.1)
{

	X = Data;
	Y = Label;
	C = Regularisation;
	NumberOfData = Y.rows();
	Kernel = Eigen::MatrixXd::Zero(NumberOfData,NumberOfData);

	// scale Data
	ScaleMin = X.minCoeff();
	X = X - ScaleMin * Eigen::MatrixXd::Ones(X.rows(), X.cols());
	ScaleMax = X.maxCoeff();
	X = X / ScaleMax;



	ComputePairwiseDistances();
}

CSVM::~CSVM()
{
	// TODO Auto-generated destructor stub
}
/// <summary>
/// compute distances for kernel matrix
/// </summary>
void CSVM::ComputePairwiseDistances()
{
	Eigen::VectorXd nsq = X.array().square().rowwise().sum();

	// parallel
	// TODO: feature matrix should be row major
	A = Eigen::MatrixXd::Zero(NumberOfData,NumberOfData);

	#pragma omp parallel for
	for(int i=0;i<NumberOfData;i++){
		Eigen::VectorXd X2 = X.row(i).array().square().rowwise().sum();
		A.col(i) = nsq + X2.replicate(NumberOfData,1) - 2*X*X.row(i).transpose();
	}

	/* alternative version (not parallel):
	 * A = ((nsq.replicate(1, X.rows()) - (2 * X) * X.transpose()).transpose()) + nsq.replicate(1, X.rows());
	 */
}
/// <summary>
/// update kernel matrix
/// </summary>
void CSVM::SetGamma(double g)
{
	// since eigen is colummajor
	#pragma omp parallel for
	for(int i=0; i<NumberOfData; ++i){
		Kernel.col(i) = (-1.0 * g * A.col(i)).array().exp();
	}

	/* alternative version (not parallel):
	 * Kernel = (-1.0 * g * A);
	 * Kernel = Kernel.array().exp();
	 */
}
/// <summary>
/// get dual objective value
/// </summary>
double CSVM::DualValue(const Eigen::MatrixXd &Kernel, const Eigen::VectorXd Alpha){
	return  Alpha.sum()    - 0.5* (double)(Alpha.cwiseProduct(Y).transpose() * (Kernel*Alpha.cwiseProduct(Y)));
}
/// <summary>
/// get primal objective value
/// </summary>
double CSVM::PrimalValue(const Eigen::MatrixXd &Kernel, const Eigen::VectorXd Beta,const Eigen::VectorXd Xi,double Bias){
	return C*Xi.sum() + 0.5*(double)(Beta.transpose() * (Kernel*Beta));
}
/// <summary>
/// get corresponding primal from dual
/// </summary>
PrimalSolution CSVM::ObtainPrimalFromDual(const Eigen::MatrixXd &Kernel, const DualSolution DS)
{
	// see appendix of my thesis for estimation of the bias value
	PrimalSolution PS;
	PS.Beta = DS.Alpha.cwiseProduct(Y);

	if(InputSettings.FixedBias == 0){
		// adapt bias (dynamic bias)
		int NumberOfFreeSV = 0;
		double ub = INF, lb=-INF, SumValueOfFreeSV=0;

		Eigen::VectorXd tmp = Y - Kernel*(Y.cwiseProduct(DS.Alpha));

		for(int i=0;i<NumberOfData;i++){

			if(DS.Alpha(i) == C){
				if(Y(i)==+1)
					ub = min(ub,tmp(i));
				else
					lb = max(lb,tmp(i));
			}else if(DS.Alpha(i) == 0){
				if(Y(i)==-1)
					ub = min(ub,tmp(i));
				else
					lb = max(lb,tmp(i));
			}else{
				++NumberOfFreeSV;
				SumValueOfFreeSV += tmp(i);
			}
		}

		PS.Bias = (NumberOfFreeSV>0) ?   SumValueOfFreeSV/NumberOfFreeSV :  (ub+lb)/2;
	}else{
		PS.Bias = DS.Bias;
	}

	// get slack values
	PS.Xi = Eigen::MatrixXd::Ones(NumberOfData,1)-Y.cwiseProduct(Kernel*PS.Beta +PS.Bias*Eigen::MatrixXd::Ones(NumberOfData,1));

    // recompute slack variables = max (0, ... )
	#pragma omp parallel for
	for(int i=0;i<NumberOfData;i++){
			if(PS.Xi(i)<0)
				PS.Xi(i)=0;
		}
    
	// get primal objective value
	PS.Value = C*PS.Xi.sum() + 0.5*(double)(PS.Beta.transpose() * (Kernel*PS.Beta));

	return PS;
}
