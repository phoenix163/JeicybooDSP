/*
GMM Algorithm Auto Test Code.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

입력인자는 2개이며, TestFile .txt 및 GMM parameter가 저장된 file .txt이다.
*/
#include<stdio.h>
#include <conio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<fftw3.h>
#include<math.h>
#include<Eigen/Dense>
using namespace Eigen;
#include <Eigen/Eigenvalues>

//#define DIAGONAL
#define PCA_LEN 4
#define PI 3.141592
#define FEATURE_LEN 12
#define NUM_OF_MIXTURE 4
#define NUM_OF_CLASS 25
#define NUM_OF_TEST 25

typedef struct gParameter{
	double alpa[NUM_OF_MIXTURE];
	double mean[NUM_OF_MIXTURE][FEATURE_LEN];
	double covariance[NUM_OF_MIXTURE][FEATURE_LEN][FEATURE_LEN];
	double eigenVector[NUM_OF_MIXTURE][FEATURE_LEN][PCA_LEN];
}GMMParameter;

typedef struct gExpParameter{
	//double weights[TOTAL_LEN][NUM_OF_MIXTURE];
	double **weights;
	double nOfKey[NUM_OF_MIXTURE];
}GMMExpectation;
void ConvertMixtureDiagonalMatrix(double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]);
void DiagonalizeCovarianceMatrix(GMMParameter *pGmmParam);
double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]);
double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen);

void main(int argc, char** argv) {
	FILE *fpReadFile, *rgfpReadMfc, *fpReadClassFile[NUM_OF_CLASS], *fpReadParam;
	double **rgdMfcTestData[NUM_OF_CLASS] = { 0, };
	double dpAccumLogLikelihood[NUM_OF_CLASS] = { 0, };
	GMMParameter *rgGmmParameter[NUM_OF_CLASS] = { 0, };
	GMMExpectation *rgGmmExp[NUM_OF_CLASS] = { 0, };
	int rgiTestFileLen[NUM_OF_CLASS] = { 0, };
	int dArg = 0, iInitCount = 0;
	double dMax = 0;
	char rgcTempTextListRead[255], rgcTempMfcListRead[255];

	if (argc != 3) {
		printf("path를 2개 입력해야 합니다.\n"); // input path, output path                                                      
		return;
	}
	else {
		for (int i = 1; i < 3; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpReadFile = fopen(argv[1], "rb")) == NULL)
		printf("Read File Open Error\n");

	if ((fpReadParam = fopen(argv[2], "rb")) == NULL)
		printf("Read File Open Error\n");

	while (!feof(fpReadFile))
	{
		for (int i = 0; i < NUM_OF_CLASS; i++) {
			rgGmmParameter[i] = (GMMParameter*)malloc(sizeof(GMMParameter));
			fread(rgGmmParameter[i], sizeof(GMMParameter), 1, fpReadParam);
			//DiagonalizeCovarianceMatrix(rgGmmParameter[i]);
		}

		for (int i = 0; i < NUM_OF_CLASS; i++) {
			memset(rgcTempTextListRead, 0, sizeof(rgcTempTextListRead));
			fscanf(fpReadFile, "%s", rgcTempTextListRead);
			iInitCount = 0;

			if ((fpReadClassFile[i] = fopen(rgcTempTextListRead, "rb")) == NULL)
				printf("Read File Open Error\n");

			
			while (!feof(fpReadClassFile[i])) {

				memset(rgcTempMfcListRead, 0, sizeof(rgcTempMfcListRead));
				fscanf(fpReadClassFile[i], "%s", rgcTempMfcListRead);
				iInitCount++;
				if ((rgfpReadMfc = fopen(rgcTempMfcListRead, "rb")) == NULL)
					printf("Read File Open Error\n");

				// file pointer move to end
				fseek(rgfpReadMfc, 0L, SEEK_END);
				// get current file pointer position
				rgiTestFileLen[i] = ftell(rgfpReadMfc) / sizeof(double);
				rgiTestFileLen[i] /= FEATURE_LEN;
				fseek(rgfpReadMfc, 0, SEEK_SET);

				rgdMfcTestData[i] = (double**)malloc(sizeof(double*) * rgiTestFileLen[i]);
				for (int k = 0; k < rgiTestFileLen[i]; k++) {
					rgdMfcTestData[i][k] = (double*)malloc(sizeof(double) * FEATURE_LEN);
					if ((fread(rgdMfcTestData[i][k], sizeof(double), FEATURE_LEN, rgfpReadMfc)) == 0) {
						printf("Break! The buffer is insufficient.\n");
						continue;
					}
				}

				for (int u = 0; u < NUM_OF_CLASS; u++) {

					dpAccumLogLikelihood[u] = Recognition(rgdMfcTestData[i], rgGmmParameter[u], rgiTestFileLen[i]);
					//printf(" %d-th Test result %f prob. \n", u + 1, dpAccumLogLikelihood[u]);
					if (u == 0) {
						dMax = dpAccumLogLikelihood[0];
						dArg = 0;
					}
					else if (dMax < dpAccumLogLikelihood[u]) {
						dMax = dpAccumLogLikelihood[u];
						dArg = u;
					}
					printf(" %d-th class probability %f \n", u+1, dpAccumLogLikelihood[u]);
				}
				printf(" %d -th result %d \n", i + 1, dArg + 1);

				for (int k = 0; k < rgiTestFileLen[i]; k++) {
					free(rgdMfcTestData[i][k]);
					rgdMfcTestData[i][k] = NULL;
				}
				free(rgdMfcTestData[i]);
				rgdMfcTestData[i] = NULL;

			}
			// Save parameter
		}

	}

	for (int i = 0; i < NUM_OF_CLASS; i++) {
		fclose(fpReadClassFile[i]);
	}
	fclose(fpReadFile);
	fclose(rgfpReadMfc);
	fclose(fpReadParam);
	getchar();
}

double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen) {

	double dpAccumLogLikelihood = 0, dTemp = 0;
	for (int i = 0; i < iFileLen; i++) {
		for (int k = 0; k < NUM_OF_MIXTURE; k++) {
			dTemp += pGmmParameter->alpa[k] * probability(dpTestBuf[i], pGmmParameter->mean[k], pGmmParameter->covariance[k], pGmmParameter->eigenVector[k]);
		}
		dpAccumLogLikelihood += log(dTemp);
		dTemp = 0;
	}
	return dpAccumLogLikelihood / iFileLen;
}

double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]) {
#if 0
#ifndef DIAGONAL
#if 0
	for (int j = 0; j < FEATURE_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			printf(" %.3f ", rgdCovariance[j][m]);
		}
		printf("\n");
	}
#endif

	MatrixXd mxCovarianceMatrix = MatrixXd(FEATURE_LEN, FEATURE_LEN);
	MatrixXd mxMeanVector = MatrixXd(FEATURE_LEN, 1);
	MatrixXd mxTempVector = MatrixXd(1, FEATURE_LEN);
	double dTemp1 = 0, dTemp2 = 0, dTemp3 = 0;
	for (int j = 0; j < FEATURE_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			mxCovarianceMatrix(j, m) = rgdCovariance[j][m];
		}
	}
	for (int i = 0; i < FEATURE_LEN; i++) {
		mxMeanVector(i, 0) = pdFeature[i] - pdMean[i];
	}
	//printf(" covarianceMatrix.determinant() %f \n", ((double)pow((2 * PI), FEATURE_LEN / 2.0) * abs(sqrt(covarianceMatrix.determinant()))));
	dTemp1 = 1.0 / ((double)pow((2 * PI), FEATURE_LEN / 2.0) *sqrt(mxCovarianceMatrix.determinant()));
	dTemp2 = (mxMeanVector.transpose() * mxCovarianceMatrix.inverse() * mxMeanVector)(0, 0);
	dTemp3 = dTemp1 * (double)exp(-1 / 2.0 * dTemp2);
	//printf("dTemp3 %f \n", dTemp3);
	return dTemp3;
#else
	double dTempProb = 1.0, dTempEigen = 0;

	MatrixXd InputMatrx = MatrixXd(FEATURE_LEN, FEATURE_LEN);

	for (int j = 0; j < FEATURE_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			InputMatrx(j, m) = rgdCovariance[j][m];
		}
	}

	EigenSolver<MatrixXd> es(InputMatrx);
	MatrixXd OutputMatrx = es.pseudoEigenvalueMatrix();
	for (int i = 0; i < FEATURE_LEN; i++) {
		//printf("jc check ! %d \n", i);
		dTempEigen = OutputMatrx(i, i);
		dTempProb *= (1.0 / (pow(2.0 * PI * pow(dTempEigen, 2.0), 0.5))) *exp((-1 / 2.0) * (pdFeature[i] - pdMean[i]) / (pow(dTempEigen, 2.0)));
	}
	return dTempProb;
#endif
#endif

#if 1
	double dTempProb = 1.0;

	MatrixXd EigenMatrx = MatrixXd(FEATURE_LEN, PCA_LEN);
	MatrixXd InputMatrx = MatrixXd(1, FEATURE_LEN);

	for (int i = 0; i < FEATURE_LEN; i++) {
		InputMatrx(0, i) = pdFeature[i];
		for (int j = 0; j < PCA_LEN; j++) {
			EigenMatrx(i, j) = rgdEigenVector[i][j];
		}
	}
	InputMatrx = InputMatrx * EigenMatrx;

	for (int i = 0; i < PCA_LEN; i++) {
		//printf("jc check ! %d \n", i);
		dTempProb *= (1.0 / sqrt(2.0 * PI))* (1.0 / sqrt(rgdCovariance[i][i])) *exp((-1 / 2.0) * pow((InputMatrx(0,i) - pdMean[i]), 2.0) / (rgdCovariance[i][i]));
	}
	return dTempProb;
#endif
}
#if 0
void DiagonalizeCovarianceMatrix(GMMParameter *pGmmParam) {

	for (int k = 0; k < NUM_OF_MIXTURE; k++) {
		ConvertMixtureDiagonalMatrix(pGmmParam->mean[k], pGmmParam->covariance[k], pGmmParam->eigenVector[k]);
	}
}

void ConvertMixtureDiagonalMatrix(double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]) {

	double dTempProb = 1.0, dTempEigen = 0;

	MatrixXd CovMatrx = MatrixXd(FEATURE_LEN, FEATURE_LEN);
	MatrixXd MeanMatrx = MatrixXd(1, FEATURE_LEN);


	for (int j = 0; j < FEATURE_LEN; j++) {
		MeanMatrx(0, j) = pdMean[j];
		for (int m = 0; m < FEATURE_LEN; m++) {
			CovMatrx(j, m) = rgdCovariance[j][m];
		}
	}

	EigenSolver<MatrixXd> es(CovMatrx);
	MatrixXd EigenValueMtx = es.pseudoEigenvalueMatrix();
	MatrixXd EigenVectorMtx = es.pseudoEigenvectors();

	MeanMatrx = MeanMatrx * EigenVectorMtx;

	for (int i = 0; i < FEATURE_LEN; i++) {

		pdMean[i] = (MeanMatrx)(0, i);

		memset(rgdCovariance[i], 0, sizeof(double) * FEATURE_LEN);
		rgdCovariance[i][i] = EigenValueMtx(i, i);
		for (int j = 0; j < FEATURE_LEN; j++) {
			rgdEigenVector[i][j] = EigenVectorMtx(i, j);
		}
	}
	return;
}
#endif