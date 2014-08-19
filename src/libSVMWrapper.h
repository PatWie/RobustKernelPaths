/*
 * libSVMWrapper.h
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */

#ifndef libSVMWrapper_H_
#define libSVMWrapper_H_

#include <Eigen/Dense>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "svm.h"
#include "utils.h"

class libSVMWrapper
{
private:
	Eigen::VectorXd Y;

	int NumberOfData;

	struct svm_parameter libSVM_Parameter;                	// LibSVM Parameter
	struct svm_problem libSVM_Problem; 		               	// problem description
	struct svm_model *libSVM_Model = NULL;							// result of libSVM
	struct svm_node *libSVM_x_space;
public:
	libSVMWrapper(Eigen::VectorXd Labels, double C);
	virtual ~libSVMWrapper();
	DualSolution Solve(const Eigen::MatrixXd &Kernel);



};

#endif /* libSVMWrapper_H_ */
