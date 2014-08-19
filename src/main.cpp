/*
 * main.cpp
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *      Mail:   patrick@wieschollek.info
 */
#include <iostream>
#include <Eigen/Dense>
#include <list>
#include <fstream>
#include <string>
#include <sstream>

#include "CSVM.h"
#include "utils.h"
#include "FileHelper.h"

/********************* forward declarations *******************************************/

// Matlab-like function
Eigen::VectorXd logspace(double lmin, const double lmax, const int steps);
// computes exact kernel path
void ExactPathAlgorithm(CSVM &SVM, libSVMWrapper &libSVM);
// computes approx path
void ApproxPathAlgorithm(CSVM &SVM, libSVMWrapper &libSVM, double Epsilon);
// use approx path to reconstruct cool images
void ReconstructionAlgorithm(CSVM &SVM, std::list<DualSolution> &ApproximativePath);
// parse the options from the command line
void parse_command_line(int argc, char **argv, char *input_file_name);
// show help
void exit_with_help();

/******************** global variables ************************************************/
// parsed console options
Settings InputSettings;

// stores both paths
std::list<DualSolution> ApproximativePath;
std::list<DualSolution> ExactPath;

Eigen::MatrixXd X;		// training dataset (features)
Eigen::VectorXd Y;		// training dataset (labels)



int main(int argc, char **argv)
{

	// initialize default settings
	InputSettings.Algorithm = 0;
	InputSettings.Epsilon = 0.1;
	InputSettings.Costs = 0.1;
	InputSettings.Left = 0.0009;
	InputSettings.Right = 1024;
	InputSettings.FixedBias = 0;

	// load customs settings from console (as libsvm does)
	parse_command_line(argc, argv, InputSettings.input_file_name);

	// parse libSVM dataset and load the data into X,Y
	FileHelper::read_problem(InputSettings.input_file_name, X, Y);

	// create the SVM object to store kernel matrix and compute the primal solution
	// scale data and compute pairwise distances for faster kernel matrix computation
	CSVM SVMObject(X, Y, InputSettings.Costs);
	// wraps the libsvm cpp implementation to Eigen3 (malloc only once the space)
	libSVMWrapper libSVM(Y, InputSettings.Costs);

	// (simple menu) switch the routines
	switch(InputSettings.Algorithm){
		case 0:
			std::cout << "-> start exact path algorithm" << std::endl;
			ExactPathAlgorithm(SVMObject, libSVM);
			break;
		case 1:
			std::cout << "-> start approximate path algorithm" << std::endl;
			ApproxPathAlgorithm(SVMObject, libSVM, InputSettings.Epsilon);
			break;
		case 2:
			std::cout << "-> start approximate path algorithm" << std::endl;
			ApproxPathAlgorithm(SVMObject, libSVM, InputSettings.Epsilon);
			std::cout << "-> start reconstruction path algorithm" << std::endl;
			ReconstructionAlgorithm(SVMObject,ApproximativePath);
			break;
		case 3:
			std::cout << "-> start exact path algorithm" << std::endl;
			ExactPathAlgorithm(SVMObject, libSVM);
			std::cout << "-> start approximate path algorithm" << std::endl;
			ApproxPathAlgorithm(SVMObject, libSVM, InputSettings.Epsilon);
			std::cout << "-> start reconstruction path algorithm" << std::endl;
			ReconstructionAlgorithm(SVMObject,ApproximativePath);
			break;
	}

	exit(0);


/*  How to use this ?
 *  --------------------------
 *
 *  Eigen::MatrixXd X;												// training dataset (features)
 *  Eigen::VectorXd Y;												// training dataset (labels)
 *  FileHelper::read_problem("ionospshere_scale.train", X, Y);     	// load dataset
 *  std::cout << X.rows() << " "<< X.cols() << std::endl;			// shows size
 *
 *  CSVM SVMObject(X, Y, 0.1);									    // scale data and compute distances
 *  libSVMWrapper libSVM(Y, 0.1);
 *
 *  // set the kernel parameter and compute the kernel matrix
 *  SVMObject.SetGamma(10.0);
 *
 *  // get optimal solution in gamma = 10.0
 *  DualSolution DS = libSVM.Solve(SVMObject.Kernel);
 *
 *  // compute primal solution using the dual solution
 *  PrimalSolution PS = SVMObject.ObtainPrimalFromDual(SVMObject.Kernel, DS);
 *
 *  // estimate gap
 *
 *  double dv = SVMObject.DualValue(SVMObject.Kernel, DS.Alpha);
 *  double pv = SVMObject.PrimalValue(SVMObject.Kernel, PS.Beta, PS.Xi, PS.Bias);
 *
 *  double GAP = pv-dv;
 */

}


/// <summary>
/// use the approximate path to compute the bounds
/// </summary>
/// <param name="SVM">SVM object to compute Kernel</param>
/// <param name="ApproximativePath">path that should be use to compute the bounds</param>
void ReconstructionAlgorithm(CSVM &SVM, std::list<DualSolution> &ApproximativePath){

	// between every optimization call display "SampleSize" addition points
	int SampleSize = 10;

	// put the output into a text file (for plots)
	std::ofstream fout;
	std::ostringstream newfilename;
	newfilename << InputSettings.input_file_name << "_eps" << InputSettings.Epsilon << "_c"<<InputSettings.Costs;
	if(InputSettings.FixedBias == 1){
		newfilename << "_fixedbias";
	}
	newfilename <<"_reconstr.dat";
	fout.open (newfilename.str().c_str());

	std::list<DualSolution>::iterator it2 = ApproximativePath.begin();
	it2++;
	std::list<DualSolution>::iterator it = ApproximativePath.begin();


	// iterate the entire approximate path
	for (; it2 != ApproximativePath.end(); ){
		// set gamma to get kernel matrix
		SVM.SetGamma(it->Gamma);
		// use dual solution ...
		DualSolution DS = *it;
		// ... to get primal solution
		PrimalSolution PS = SVM.ObtainPrimalFromDual(SVM.Kernel,DS);

		// sample get additional points by pursuing both bounds
		// points to sample from:
		Eigen::VectorXd SamplePoints = logspace(log10(it->Gamma),log10((double)(it2->Gamma)-0.000002),SampleSize);
		for(int s=0;s<SampleSize;s++){
			// adapt kernel matrix
			SVM.SetGamma((double)SamplePoints(s));
			// get lower bound
			double dv = SVM.DualValue(SVM.Kernel, it->Alpha);
			// compute upper bound
			PS = SVM.ObtainPrimalFromDual(SVM.Kernel,*it);
			// store in file
			fout << (double)SamplePoints(s) << "," << dv << "," << PS.Value << std::endl;
		}
		// TikZ needs "NaN" for "unbounded coords=jump" for cool plots
		fout << (it2->Gamma-0.000001) << ",NaN,NaN"<< std::endl;

		it2++;
		it++;

	}

	fout.close();

}

/// <summary>
/// compute the approximate path
/// </summary>
/// <param name="SVM">SVM object to compute Kernel</param>
/// <param name="libSVM">wrapper to solve the svm</param>
/// <param name="Epsilon">approximation guarantee</param>
void ApproxPathAlgorithm(CSVM &SVM, libSVMWrapper &libSVM, double Epsilon)
{
	// aliases
	double CurrentGamma = InputSettings.Left;
	double MaxGamma = InputSettings.Right;

	// start with optimal solution pair in "t_current"
	SVM.SetGamma(CurrentGamma);
	// optimizer call for dual solution
	DualSolution DS = libSVM.Solve(SVM.Kernel);
	DS.Gamma = CurrentGamma;
	// save current update point
	ApproximativePath.push_back(DS);
	// use dual solution to get corresponding primal solution
	PrimalSolution PS = SVM.ObtainPrimalFromDual(SVM.Kernel, DS);
	PS.Gamma = CurrentGamma;

	// oops the given value for epsilon is to small. For pratical purposes try eps>= 0.125
	if(PS.Value-DS.Value> Epsilon){
		std::cerr << "ERROR: gap (" <<PS.Value-DS.Value <<") in start is too large!"<< std::endl;
		std::cerr << "dual bias: " << DS.Bias <<" primal bias: "<< PS.Bias << std::endl;
		exit(0);
	}

	double GAP = 0;
	// run algorithm
	while (CurrentGamma < MaxGamma)
	{
		// try to find interval bounds
		double LeftInterval = CurrentGamma;
		double RightInterval = CurrentGamma * 2;

		// find interval for next update
		while (RightInterval < MaxGamma)
		{
			SVM.SetGamma(RightInterval);
			// compute lower bound
			double dv = SVM.DualValue(SVM.Kernel, DS.Alpha);
			// obtain upper bound
			PS = SVM.ObtainPrimalFromDual(SVM.Kernel, DS);
			double pv = PS.Value;
			GAP = pv - dv;

			if (GAP < Epsilon)
			{
				// shift interval
				LeftInterval = RightInterval;
				RightInterval = 2 * RightInterval;
			}
			else
			{
				// this interval contains the next update point
				break;
			}

		}

		printf("search update point within [%.4f %.4f]\n", LeftInterval, RightInterval);

		double Middle = 0;
		double Diameter = RightInterval - LeftInterval;
		// search update points (kind of binary search, precision=0.0001)
		while (Diameter > 0.0001 || GAP > Epsilon)
		{
			Middle = (RightInterval + LeftInterval) / 2;

			SVM.SetGamma(Middle);
			// recompute upper bound (primal solution can become infeasible)
			PS = SVM.ObtainPrimalFromDual(SVM.Kernel, DS);
			GAP = PS.Value - SVM.DualValue(SVM.Kernel, DS.Alpha);

			if (GAP < Epsilon)
			{
				LeftInterval = Middle;
			}
			else
			{
				RightInterval = Middle;
			}

			Diameter = RightInterval - LeftInterval;

		}
		printf("update gamma in %.4f (gap = %f <-> eps = %f)\n", Middle, GAP, Epsilon);

		CurrentGamma = Middle;
		SVM.SetGamma(CurrentGamma);
		// optimization call
		DS = libSVM.Solve(SVM.Kernel);
		DS.Gamma = CurrentGamma;
		// store update point
		ApproximativePath.push_back(DS);

		PS = SVM.ObtainPrimalFromDual(SVM.Kernel, DS);

	}

	// write to file
	std::ofstream fout;
	std::ostringstream newfilename;
	newfilename << InputSettings.input_file_name << "_eps" << InputSettings.Epsilon << "_c"<<InputSettings.Costs;
	if(InputSettings.FixedBias == 1){
			newfilename << "_fixedbias";
		}
	newfilename <<".dat";
	fout.open (newfilename.str().c_str());
	for (std::list<DualSolution>::iterator it = ApproximativePath.begin(); it != ApproximativePath.end(); it++)
		fout << it->Gamma << "," << it->Value << std::endl;
	fout.close();
}


/// <summary>
/// naive grid search
/// </summary>
/// <param name="SVM">SVM object to compute Kernel</param>
/// <param name="libSVM">wrapper to solve the svm</param>
void ExactPathAlgorithm(CSVM &SVM, libSVMWrapper &libSVM)
{

	const int Steps = 40;
	Eigen::VectorXd S = logspace(log10(InputSettings.Left), log10(InputSettings.Right), Steps);

	// grid data
	Eigen::MatrixXd Objectives(Steps, 2);

	for (int i = 0; i < Steps; i++)
	{
		double Gamma = S(i);
		SVM.SetGamma(Gamma);
		DualSolution DS = libSVM.Solve(SVM.Kernel);
		DS.Gamma = Gamma;
		ExactPath.push_back(DS);
	}

	// write to file
	std::ofstream fout;
	std::ostringstream newfilename;
	newfilename << InputSettings.input_file_name << "_exact" << "_c"<<InputSettings.Costs <<".dat";
	fout.open (newfilename.str().c_str());
	for (std::list<DualSolution>::iterator it = ExactPath.begin(); it != ExactPath.end(); it++)
		fout << it->Gamma << "," << it->Value << std::endl;
	fout.close();

}

// matlab like function
Eigen::VectorXd logspace(double lmin, const double lmax, const int steps)
{
	Eigen::VectorXd R(steps);

	if (steps == 1)
	{
		R(0) = lmax;
		return R;
	}

	double dx = (lmax - lmin) / (steps - 1);

	for (int i = 0; i < steps; i++)
		R(i) = pow(10.0, (lmin + dx * i));

	return R;
}

void exit_with_help()
{
	printf(
	"Usage: ApproxPath [options] libsvm_training_set_file\n"
	"options:\n"
	"-a algorithm : set type of algorithm (default 0)\n"
	"        0 -- only naiv search (40 equi. points)	\n"
	"        1 -- only approx gap \n"
	"        2 -- approx gap (inclusive reconstruction) \n"
	"        3 -- all \n"
	"-e epsilon : set tolerance of gap guarantee (default 0.1)\n"
	"-c cost : set the parameter C of C-SVM  (default 0.1)\n"
	"-b bias : 1=fixed bias, 0=dynamic bias (default 0)\n"
	"-l left interval boundary  (default 0.0009)\n"
	"-r right interval boundary (default 1024)\n"
	//"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

// as in libSVM
void parse_command_line(int argc, char **argv, char *input_file_name){
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'a':
				InputSettings.Algorithm = atoi(argv[i]);
				if(InputSettings.Algorithm<0 || InputSettings.Algorithm > 3)
					exit_with_help();
				break;
			case 'e':
				InputSettings.Epsilon = atof(argv[i]);
				break;
			case 'c':
				InputSettings.Costs = atof(argv[i]);
				break;
			case 'l':
				InputSettings.Left = atof(argv[i]);
				break;
			case 'b':
				InputSettings.FixedBias = atoi(argv[i]);
				break;
			case 'r':
				InputSettings.Right = atof(argv[i]);
				break;
			/*case 'q':
				print_func = &print_null;
				i--;
				break;*/
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	if(i>=argc)
			exit_with_help();

		strcpy(input_file_name, argv[i]);
}
