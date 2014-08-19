/*
 * CSVM.h
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */

#include <Eigen/Dense>
#include "libSVMWrapper.h"

#ifndef CSVM_H_
#define CSVM_H_

class CSVM
{
protected:
	double C, ScaleMin, ScaleMax, Gamma;
	Eigen::MatrixXd X;
	Eigen::VectorXd Y;
	int NumberOfData;



public:
	void ComputePairwiseDistances();
	Eigen::MatrixXd A;
	Eigen::MatrixXd Kernel;

	CSVM(Eigen::MatrixXd Data, Eigen::VectorXd Label, double Regularisation);
	virtual ~CSVM();

	double DualValue(const Eigen::MatrixXd &Kernel, const Eigen::VectorXd Alpha);
	double PrimalValue(const Eigen::MatrixXd &Kernel, const Eigen::VectorXd Beta,const Eigen::VectorXd Xi,double Bias);

	void SetGamma(double g);
	PrimalSolution ObtainPrimalFromDual(const Eigen::MatrixXd &Kernel, const DualSolution DS);
};

#endif /* CSVM_H_ */
