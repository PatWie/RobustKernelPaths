/*
 * FileHelper.cpp
 *
 *  Created on: 26.01.2014
 *      Author: Patrick Wieschollek
 *        Mail: patrick@wieschollek.info
 *    based on: libsvmread
 */

#include "FileHelper.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "svm.h"
#include "utils.h"

#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif



static	char *line;
static int max_line_len;

char* FileHelper::readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in libsvm format)
void FileHelper::read_problem(const char *filename, Eigen::MatrixXd &XX, Eigen::VectorXd &Y)
{

	int max_index, min_index, inst_max_index, i;
	long elements, k;
	FILE *fp = fopen(filename,"r");
	int l = 0;
	char *endptr;



	if(fp == NULL)
	{
		printf("can't open input file %s\n",filename);
		exit(0);
	}

	max_line_len = 1024;
	line = (char *) malloc(max_line_len*sizeof(char));

	max_index = 0;
	min_index = 1; // our index starts from 1
	elements = 0;
	while(readline(fp) != NULL)
	{
		char *idx, *val;
		// features
		int index = 0;

		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		strtok(line," \t"); // label
		while (1)
		{
			idx = strtok(NULL,":"); // index:value
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			errno = 0;
			index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index)
			{
				printf("Wrong input format at line %d\n",l+1);
				exit(0);
			}
			else
				inst_max_index = index;

			min_index = min(min_index, index);
			elements++;
		}
		max_index = max(max_index, inst_max_index);
		l++;
	}
	rewind(fp);


	//std::cout << max_index<<std::endl;



	// y
	Y=Eigen::MatrixXd::Zero(l,1);
;
	// x^T
	Eigen::MatrixXd X;
	if (min_index <= 0){
		X = Eigen::MatrixXd::Zero(max_index-min_index+1,l);

	}else{
		X = Eigen::MatrixXd::Zero(max_index,l);

	}


	k=0;
	int jc=0;
	int LineCounter=0;
	for(i=0;i<l;i++)
	{
		char *idx, *val, *label;
		jc= k;

		readline(fp);

		label = strtok(line," \t\n");
		if(label == NULL)
		{
			std::cerr << "ERROR: Empty line at line "<< (i+1) << std::endl;
			exit(0);
		}
		Y[i] = strtod(label,&endptr);
		if((Y[i] != 1) && (Y[i] != -1)){
			std::cerr << "ERROR: labels have to be {+1,-1} but " << (double) Y[i] << " was given!"<< std::endl;
			exit(0);
		}
		if(endptr == label || *endptr != '\0')
		{
			std::cerr << "ERROR: Wrong input format at line "<< (i+1) << std::endl;
			exit(0);
		}

		// features
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			//std::cout << idx << std::endl;

			int ir = (strtol(idx,&endptr,10) - min_index); // precomputed kernel has <index> start from 0

			errno = 0;
			double vvv = strtod(val,&endptr);
			//std::cout << idx << " "<< i << " "<< vvv<<std::endl;
			//X(ir,jc/2) =  vvv;
			X(ir,i) = vvv;
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				std::cerr << "ERROR: Wrong input format at line "<< (i+1) << std::endl;
				exit(0);
			}
			++k;
		}
	}
	jc = k;

	fclose(fp);
	free(line);

	XX = X.transpose();
}
