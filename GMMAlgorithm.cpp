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
#define PI 3.141592
#define FEATURE_LEN 12
#define NUM_OF_MIXTURE 4
#define DIST(a, b) pow(a - b,2.0)
#define VAR(a, b)  pow(a - b,2.0)
#define NUM_OF_CLASS 5
#define NUM_OF_TEST 5

typedef struct gParameter{
	//double weights[TOTAL_LEN][NUM_OF_MIXTURE];
	double **weights;
	double alpa[NUM_OF_MIXTURE];
	double nOfKey[NUM_OF_MIXTURE];
	double mean[NUM_OF_MIXTURE][FEATURE_LEN];
	double covariance[NUM_OF_MIXTURE][FEATURE_LEN][FEATURE_LEN];
}GMMParameter;

double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN]);
void EmAlgorithmBasedGmmParameter(double **pdInBuf, GMMParameter *pGmmParam, int numOfFrames);
void KmeansAlogorithm(double **pdInBuf, GMMParameter *pGmmParam, int numOfFrames);
double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen);
void TrainingData(double **rgdMfcTrainData[NUM_OF_CLASS], GMMParameter *rggmmParameter[NUM_OF_CLASS], int iFileLen);
int TestData(double **rgdMfcTestData[NUM_OF_CLASS], GMMParameter *rggmmParameter[NUM_OF_CLASS], int iFileLen);

void main(int argc, char** argv) {
	FILE *rgfpReadTrain[NUM_OF_CLASS];
	FILE *rgfpReadTest[NUM_OF_CLASS];
	double **rgdMfcTrainData[NUM_OF_CLASS] = { 0, };
	double **rgdMfcTestData[NUM_OF_CLASS] = { 0, };
	double dpAccumLogLikelihood[NUM_OF_CLASS] = { 0, };
	GMMParameter *rgGmmParameter[NUM_OF_CLASS] = { 0, };
	int rgiTrainFileLen[NUM_OF_CLASS] = { 0, }, rgiTestFileLen[NUM_OF_CLASS] = { 0, };
	int dArg = 0;
	double dMax = 0;
	if (argc != 11) {
		printf("path를 10개 입력해야 합니다.\n"); // input path, output path
		printf("argc %d \n", argc);
		getchar();
		return;
	}
	else {
		for (int i = 1; i < 11; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	for (int i = 0; i < NUM_OF_CLASS; i++) {
		rgfpReadTrain[i] = fopen(argv[i + 1], "rb");
		// file pointer move to end
		fseek(rgfpReadTrain[i], 0L, SEEK_END);
		// get current file pointer position
		rgiTrainFileLen[i] = ftell(rgfpReadTrain[i]) / sizeof(double);
		rgiTrainFileLen[i] /= FEATURE_LEN;
		fseek(rgfpReadTrain[i], 0, SEEK_SET);


		rgdMfcTrainData[i] = (double**)malloc(sizeof(double*) * rgiTrainFileLen[i]);
		rgGmmParameter[i] = (GMMParameter*)malloc(sizeof(GMMParameter));

		rgGmmParameter[i]->weights = (double**)malloc(sizeof(double*) * rgiTrainFileLen[i]);
		for (int k = 0; k < rgiTrainFileLen[i]; k++) {
			rgGmmParameter[i]->weights[k] = (double*)malloc(sizeof(double) * NUM_OF_MIXTURE);
		}

		for (int k = 0; k < rgiTrainFileLen[i]; k++) {
			rgdMfcTrainData[i][k] = (double*)malloc(sizeof(double) * FEATURE_LEN);
			fread(rgdMfcTrainData[i][k], sizeof(double), FEATURE_LEN, rgfpReadTrain[i]);
		}
		KmeansAlogorithm(rgdMfcTrainData[i], rgGmmParameter[i], rgiTrainFileLen[i]);
		EmAlgorithmBasedGmmParameter(rgdMfcTrainData[i], rgGmmParameter[i], rgiTrainFileLen[i]);
		for (int k = 0; k < rgiTrainFileLen[i]; k++) {
			free(rgdMfcTrainData[i][k]);
			rgdMfcTrainData[i][k] = NULL;
		}
		free(rgdMfcTrainData[i]);
		rgdMfcTrainData[i] = NULL;
	}

	printf("test start!\n");

	for (int i = 0; i < NUM_OF_TEST; i++) {

		rgfpReadTest[i] = fopen(argv[NUM_OF_CLASS + i + 1], "rb");
		// file pointer move to end
		fseek(rgfpReadTest[i], 0L, SEEK_END);
		// get current file pointer position
		rgiTestFileLen[i] = ftell(rgfpReadTest[i]) / sizeof(double);
		rgiTestFileLen[i] /= FEATURE_LEN;
		rgdMfcTestData[i] = (double**)malloc(sizeof(double*) * rgiTestFileLen[i]);

		fseek(rgfpReadTest[i], 0, SEEK_SET);
		for (int k = 0; k < rgiTestFileLen[i]; k++) {
			rgdMfcTestData[i][k] = (double*)malloc(sizeof(double) * FEATURE_LEN);
			fread(rgdMfcTestData[i][k], sizeof(double), FEATURE_LEN, rgfpReadTest[i]);
		}

		for (int k = 0; k < NUM_OF_CLASS; k++) {

			dpAccumLogLikelihood[k] = Recognition(rgdMfcTestData[i], rgGmmParameter[k], rgiTestFileLen[i]);
			printf(" %d-th Test result %f prob. \n", k + 1, dpAccumLogLikelihood[k]);
			if (k == 0) {
				dMax = dpAccumLogLikelihood[0];
				dArg = 0;
			} else if (dMax < dpAccumLogLikelihood[k]) {
				dMax = dpAccumLogLikelihood[k];
				dArg = k;
			}
		}
		printf(" %d-th Test result %d select \n", i + 1, dArg + 1);
	}

	for (int i = 0; i < NUM_OF_CLASS; i++) {

		for (int k = 0; k < rgiTestFileLen[i]; k++) {
			free(rgdMfcTestData[i][k]);
			rgdMfcTestData[i][k] = NULL;
		}

		free(rgdMfcTestData[i]);
		rgdMfcTestData[i] = NULL;
		for (int k = 0; k < rgiTrainFileLen[i]; k++) {
			free(rgGmmParameter[i]->weights[k]);
			rgGmmParameter[i]->weights[k] = NULL;
		}
		free(rgGmmParameter[i]->weights);
		rgGmmParameter[i]->weights = NULL;
		free(rgGmmParameter[i]);
		rgGmmParameter[i] = NULL;
		fclose(rgfpReadTrain[i]);
		fclose(rgfpReadTest[i]);
	}
	getchar();
}
#if 0
void TrainingData(double **rgdMfcTrainData[NUM_OF_CLASS], GMMParameter *rgGmmParameter[NUM_OF_CLASS], int iFileLen) {
	for (int i = 0; i < NUM_OF_CLASS; i++) {
		KmeansAlogorithm(rgdMfcTrainData[i], rgGmmParameter[i], iFileLen);
		EmAlgorithmBasedGmmParameter(rgdMfcTrainData[i], rgGmmParameter[i], iFileLen);
	}
}
int TestData(double **rgdMfcTestData[NUM_OF_CLASS], GMMParameter *rgGmmParameter[NUM_OF_CLASS], int iFileLen) {

	double rgdAccumLogLikelihood[NUM_OF_CLASS] = { 0, };
	return Recognition(rgdMfcTestData, rgGmmParameter, rgdAccumLogLikelihood, iFileLen);
}
#endif

double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen) {

	double dpAccumLogLikelihood = 0, dTemp = 0;
	for (int i = 0; i < iFileLen; i++) {
		for (int k = 0; k < NUM_OF_MIXTURE; k++) {
			dTemp += pGmmParameter->alpa[k] * probability(dpTestBuf[i], pGmmParameter->mean[k], pGmmParameter->covariance[k]);
		}
		dpAccumLogLikelihood += log(dTemp);
		dTemp = 0;
	}
	return dpAccumLogLikelihood;
}

double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN]) {
#ifndef DIAGONAL
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
	dTemp1 = 1.0 / ((double)pow((2 * PI), FEATURE_LEN / 2.0) * abs(sqrt(mxCovarianceMatrix.determinant())));
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
}

void EmAlgorithmBasedGmmParameter(double **pdInBuf, GMMParameter *pGmmParam, int numOfFrames) {
	int count_ = 0;
	double dTemp1 = 0, dTemp2 = 0, dTempAft = 0, dTempBf = 0;
	MatrixXd mxMeanVector = MatrixXd(FEATURE_LEN, 1);
	MatrixXd mxTransVector = MatrixXd(1, FEATURE_LEN);
	MatrixXd mxResultVector = MatrixXd(FEATURE_LEN, FEATURE_LEN);
	for (int i = 0; i < NUM_OF_MIXTURE; i++) {
		pGmmParam->alpa[i] = 1.0 / NUM_OF_MIXTURE;
	}
	for (int i = 0; i < numOfFrames; i++) {
		for (int k = 0; k < NUM_OF_MIXTURE; k++) {
			pGmmParam->weights[i][k] = 1.0 / NUM_OF_MIXTURE;
		}
	}
	double alpha = 0.000000001;
	while (1) {
		count_++;
		dTemp1 = 0;
		dTemp2 = 0;
		dTempAft = 0;
		printf("count_ %d \n", count_);
		//printf("jcboo numof %d \n", numOfFrames);
		for (int i = 0; i < numOfFrames; i++) {
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				pGmmParam->weights[i][k] = probability(pdInBuf[i], pGmmParam->mean[k], pGmmParam->covariance[k]) * pGmmParam->alpa[k];
				//printf(" gm->weights[i][k] %f \n", gm->weights[i][k]);
				dTemp2 += pGmmParam->weights[i][k];
			}
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				pGmmParam->weights[i][k] = pGmmParam->weights[i][k] / dTemp2;
			}
			dTemp2 = 0;
			//printf("jcboo  %d-th \n", i);
		}
		//printf("jcboo 1 \n");
		for (int k = 0; k < NUM_OF_MIXTURE; k++) {
			for (int i = 0; i < numOfFrames; i++) {
				pGmmParam->alpa[k] += pGmmParam->weights[i][k];
			}
			pGmmParam->nOfKey[k] = pGmmParam->alpa[k];
			pGmmParam->alpa[k] /= (double)numOfFrames;
		}
		for (int d = 0; d < FEATURE_LEN; d++) {
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				for (int i = 0; i < numOfFrames; i++) {
					pGmmParam->mean[k][d] += pGmmParam->weights[i][k] * pdInBuf[i][d];
				}
				pGmmParam->mean[k][d] /= pGmmParam->nOfKey[k];
			}
		}
		for (int k = 0; k < NUM_OF_MIXTURE; k++) {
			mxResultVector.setZero(FEATURE_LEN, FEATURE_LEN);
			mxMeanVector.setZero(FEATURE_LEN, 1);
			//printf(" mean vector %f \n", meanVector(3, 0));
			for (int i = 0; i < numOfFrames; i++) {
				for (int d = 0; d < FEATURE_LEN; d++) {
					mxMeanVector(d, 0) = pdInBuf[i][d] - pGmmParam->mean[k][d];
				}
				mxResultVector += mxMeanVector* mxMeanVector.transpose() * pGmmParam->weights[i][k];
			}
			mxResultVector /= pGmmParam->nOfKey[k]; // 상수값 나눌 시 원소 전체 나눠지는지 확인.
			for (int j = 0; j < FEATURE_LEN; j++) {
				for (int m = 0; m < FEATURE_LEN; m++) {
					pGmmParam->covariance[k][j][m] = mxResultVector(j, m); // 상수값 나눌 시 원소 전체 나눠지는지 확인.
				}
			}
		}
		for (int i = 0; i < numOfFrames; i++) {
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				dTemp2 += pGmmParam->alpa[k] * probability(pdInBuf[i], pGmmParam->mean[k], pGmmParam->covariance[k]);
			}
			dTempAft += log(dTemp2);
		}
		printf(" before %.5f after %.5f \n", dTempBf, dTempAft);
		if (count_ > 2) //&& abs(temp_before - temp_after) < 1)
			break;

		dTempBf = dTempAft;
	}
	printf("training end! \n");

}

void KmeansAlogorithm(double **pdInBuf, GMMParameter *pGmmParam, int numOfFrames) {
	double cost = 0, cost_before = 0, count = 0;
	double dTempDist = 0;
	int arg_temp = 0, count_ = 0;
	MatrixXd covarianceMatrix_k = MatrixXd(FEATURE_LEN, FEATURE_LEN);
	MatrixXd meanVector_k = MatrixXd(FEATURE_LEN, 1);
	int(*Selection)[FEATURE_LEN][NUM_OF_MIXTURE] = { 0, };

	Selection = (int(*)[FEATURE_LEN][NUM_OF_MIXTURE])malloc(sizeof(int[FEATURE_LEN][NUM_OF_MIXTURE]) * numOfFrames);

	while (1) {
		//2)
		printf(" go ! \n");
		count_++;
		for (int i = 0; i < numOfFrames; i++) {
			for (int k = 0; k < FEATURE_LEN; k++) {
				for (int j = 0; j < NUM_OF_MIXTURE; j++) {
					if (dTempDist > DIST(pdInBuf[i][k], pGmmParam->mean[j][k]))
						arg_temp = j;

					dTempDist = DIST(pdInBuf[i][k], pGmmParam->mean[j][k]);
				}
				Selection[i][k][arg_temp] = 1;
				arg_temp = 0;
				dTempDist = 0;
			}
		}

		//3)
		cost = 0;
		for (int i = 0; i < numOfFrames; i++) {
			for (int k = 0; k < FEATURE_LEN; k++) {

				for (int j = 0; j < NUM_OF_MIXTURE; j++) {
					if (Selection[i][k][j])
						cost += DIST(pdInBuf[i][k], pGmmParam->mean[j][k]);
				}
			}
		}
		printf(" 1 prev cost %f, current cost %f \n", cost_before, cost);
		if (count_ == 1 || cost_before > cost) {
			printf(" prev cost %f, current cost %f \n", cost_before, cost);
			cost_before = cost;

		}
		else {
			//calc var;
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				covarianceMatrix_k.setZero(FEATURE_LEN, FEATURE_LEN);
				for (int i = 0; i < numOfFrames; i++) {

					for (int z = 0; z < FEATURE_LEN; z++)
						meanVector_k(z, 0) = pdInBuf[i][z] - pGmmParam->mean[k][z];

					covarianceMatrix_k += meanVector_k * meanVector_k.transpose();
				}
				covarianceMatrix_k /= numOfFrames;

				for (int m = 0; m < FEATURE_LEN; m++) {
					for (int u = 0; u< FEATURE_LEN; u++) {
						pGmmParam->covariance[k][m][u] = covarianceMatrix_k(m, u);
					}
					pGmmParam->mean[k][m] = pGmmParam->mean[k][m];
				}
			}
			break;
		}

		//1)
		memset(pGmmParam->mean, 0, sizeof(double) * NUM_OF_MIXTURE);
		count = 0;
		for (int j = 0; j < NUM_OF_MIXTURE; j++) {
			for (int k = 0; k < FEATURE_LEN; k++) {

				for (int i = 0; i < numOfFrames; i++) {
					if (Selection[i][k][j]) {
						count++;
						pGmmParam->mean[j][k] += pdInBuf[i][k];
					}
				}
				if (count == 0) {
					pGmmParam->mean[j][k] = 0;
				}
				else {
					pGmmParam->mean[j][k] /= count;
				}
				count = 0;
			}
		}
	}
	free(Selection);
	Selection = NULL;
}
