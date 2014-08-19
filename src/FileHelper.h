/*
 * FileHelper.h
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 */



#ifndef FILEHELPER_H_
#define FILEHELPER_H_

#include <Eigen/Dense>

class FileHelper
{

public:

	static char* readline(FILE *input);
	static void read_problem(const char *filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y);
};

#endif /* FILEHELPER_H_ */
