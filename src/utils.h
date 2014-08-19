/*
 * utils.h
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */

#ifndef CSVMUtils_H_
#define CSVMUtils_H_

#include <Eigen/Dense>

struct DualSolution
{
	double Bias;
	double Value;
	Eigen::VectorXd Alpha;
	double Gamma=0;
};

struct PrimalSolution
{
	double Value;
	double Bias;
	Eigen::VectorXd Beta;
	Eigen::VectorXd Xi;
	double Gamma=0;
};

// parse console options
struct Settings{
	int Algorithm;
	double Epsilon;
	double Costs;
	double Left;
	double Right;
	char input_file_name[1024];
	int FixedBias; // 0=dynamic, 1 = fixed
};



#endif /* CSVMUtils_H_ */
