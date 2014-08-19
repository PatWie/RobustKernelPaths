/*
 * libSVMWrapper.cpp
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */

#include "libSVMWrapper.h"
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// make libSVM quiet!
void print_null(const char *s)
{
}
/// <summary>
/// use a kernel matrix and solve the svm problem
/// </summary>
/// <param name="Labels">vector of all Labels</param>
/// <param name="C">regularisation parameter</param>
libSVMWrapper::libSVMWrapper(Eigen::VectorXd Labels, double C)
{
	svm_set_print_string_function(print_null);
	Y = Labels;

	// we need these parameters
	libSVM_Parameter.svm_type = C_SVC;
	libSVM_Parameter.kernel_type = PRECOMPUTED;
	libSVM_Parameter.C = C;

	// dummy values
	libSVM_Parameter.degree = 3;
	libSVM_Parameter.gamma = 0;
	libSVM_Parameter.coef0 = 0;
	libSVM_Parameter.nu = 0.5;
	libSVM_Parameter.cache_size = 100;
	libSVM_Parameter.eps = 1e-3;
	libSVM_Parameter.p = 0.1;
	libSVM_Parameter.shrinking = 1;
	libSVM_Parameter.probability = 0;
	libSVM_Parameter.nr_weight = 0;
	libSVM_Parameter.weight_label = NULL;
	libSVM_Parameter.weight = NULL;

	// the labels do not change
	NumberOfData = Labels.rows();
	int sc = Labels.rows() + 1;
	libSVM_Problem.y = Malloc(double,NumberOfData);
	libSVM_Problem.l = NumberOfData;

	for (int i = 0; i < NumberOfData; i++)
		libSVM_Problem.y[i] = Labels[i];

	// get space for kernel matrix (fixed size) only once
	libSVM_Problem.x = Malloc(struct svm_node *,NumberOfData);
	libSVM_x_space = Malloc(struct svm_node, NumberOfData* (sc+1));

	// get space only once
	for (int i = 0; i < libSVM_Problem.l; i++)
	{
		libSVM_Problem.x[i] = &libSVM_x_space[i * (sc + 1)];
		libSVM_Problem.y[i] = Labels[i];
	}

}

libSVMWrapper::~libSVMWrapper()
{
	// destroy everything
	svm_free_and_destroy_model(&libSVM_Model);
	svm_destroy_param(&libSVM_Parameter);
	free(libSVM_Problem.y);
	free(libSVM_Problem.x);
	free(libSVM_x_space);

}

/// <summary>
/// use a kernel matrix and solve the svm problem
/// </summary>
/// <param name="Kernel">precomputed kernel</param>
DualSolution libSVMWrapper::Solve(const Eigen::MatrixXd &Kernel)
{
	// unfortunately libsvm needs "svm_nodes", so we have to copy everything
	// workaround pointer to matrix entries (eigen3 does not allow it?)
	// copy entries (code from libSVM)
	int j = 0, sc = NumberOfData + 1;

	// TODO
	// libSVM_x_space[j+k].value = *(Kernel.data() + (k-1)*NumberOfData + i);  // by reference or pointer
	#pragma omp parallel for
	for (int i = 0; i < libSVM_Problem.l; i++)
	{
		j = (sc+1)*i ;
		for (int k = 0; k < sc; k++)
		{
			libSVM_x_space[j].index = k + 1;
			if (k == 0)
			{
				libSVM_x_space[j].value = i + 1;
			}
			else
			{
				libSVM_x_space[j+k].value = Kernel(i, k - 1);
			}

		}
		j = ((sc+1)*i+sc) ;
		libSVM_x_space[j+1].index = -1;
	}


	#ifdef DEBUG
	for (int i = 0; i < libSVM_Problem.l; i++)
	{
		if ((int) libSVM_Problem.x[i][0].value <= 0 || (int) libSVM_Problem.x[i][0].value > sc)
		{
			printf("Wrong input format: sample_serial_number out of range\n");
			exit(0);
		}
	}


	const char *error_msg;
	error_msg = svm_check_parameter(&libSVM_Problem, &libSVM_Parameter);
	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	#endif

	// train the model
	if (libSVM_Model != NULL)
	{
		svm_free_model_content(libSVM_Model);
		libSVM_Model = NULL;
	}
	libSVM_Model = svm_train(&libSVM_Problem, &libSVM_Parameter);
	// extract results

	// bias is easy
	double Bias = -1 * libSVM_Model->rho[0];

	// alpha should be dense now
	Eigen::VectorXd Alpha = Eigen::MatrixXd::Zero(NumberOfData, 1);
	for (int i = 0; i < libSVM_Model->l; i++)
	{
		Alpha(libSVM_Model->sv_indices[i] - 1) = (libSVM_Model->sv_coef[0][i] < 0) ? -1 * libSVM_Model->sv_coef[0][i] : libSVM_Model->sv_coef[0][i];
	}

	DualSolution DS;
	DS.Bias = Bias;
	DS.Alpha = Alpha;

	Eigen::VectorXd tt = Alpha.cwiseProduct(Y);

	// objective value of dual solution
	DS.Value = Alpha.sum()    - 0.5* (double)(tt.transpose() * (Kernel*tt));


	return DS;

}

