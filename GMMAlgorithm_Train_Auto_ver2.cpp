/*
GMM Algorithm Auto Training Code.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

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

#define PCA_LEN 8
#define PI 3.141592
#define FEATURE_LEN 12
#define NUM_OF_MIXTURE 4
#define NUM_OF_CLASS 25
#define THRESHOLD_OF_DISTANCE 1.0

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

void UpdateMean(double *pdInBuf, double *pMean);
double DistanceToCenter(double *pdInBuf, double *pMean);
void PCAConvertMixtureDiagonalMatrix(double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]);
void PCADiagonalizeCovarianceMatrix(GMMParameter *pGmmParam);
double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN]);
void EmAlgorithmBasedGmmParameter(double **pdInBuf, GMMParameter *pGmmParam, GMMExpectation *pGmmExp, int numOfFrames);
void KmeansAlogorithm(double **pdInBuf, GMMParameter *pGmmParam, int numOfFrames);
double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, GMMExpectation *pGmmExp, int iFileLen);

void main(int argc, char** argv) {
	FILE *fpReadFile, *rgfpReadMfc, *fpReadClassFile[NUM_OF_CLASS], *fpGMMParamWrite;
	double **rgdMfcTrainData[NUM_OF_CLASS] = { 0, };
	double dpAccumLogLikelihood[NUM_OF_CLASS] = { 0, };
	GMMParameter *rgGmmParameter[NUM_OF_CLASS] = { 0, };
	GMMExpectation *rgGmmExp[NUM_OF_CLASS] = { 0, };
	int rgiTrainFileLen[NUM_OF_CLASS] = { 0, }, rgiTestFileLen[NUM_OF_CLASS] = { 0, };
	int dArg = 0, iInitCount = 0;
	double dMax = 0;
	char rgcTempTextListRead[255], rgcTempMfcListRead[255];
	int iDebuggingCountFile = 0, iDebuggingCountList = 0;

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

	if ((fpGMMParamWrite = fopen(argv[2], "wb")) == NULL)
		printf("Write File Open Error\n");

	while (!feof(fpReadFile))
	{
		for (int i = 0; i < NUM_OF_CLASS; i++) {

			memset(rgcTempTextListRead, 0, sizeof(rgcTempTextListRead));
			fscanf(fpReadFile, "%s", rgcTempTextListRead);
			printf(" ReadTExt -------------------------------------------***********************\n");
			printf(" DebuggingCountFile %d ***********************\n", iDebuggingCountFile++);
			if ((fpReadClassFile[i] = fopen(rgcTempTextListRead, "rb")) == NULL)
				printf("Read File Open Error\n");
			iInitCount = 0;
			while (!feof(fpReadClassFile[i])) {
				printf(" iDebuggingCountList %d ++++++++++++++++++++\n", iDebuggingCountList++);
				memset(rgcTempMfcListRead, 0, sizeof(rgcTempMfcListRead));
				fscanf(fpReadClassFile[i], "%s", rgcTempMfcListRead);
				iInitCount++;
				if ((rgfpReadMfc = fopen(rgcTempMfcListRead, "rb")) == NULL)
					printf("Read File Open Error\n");

					// file pointer move to end
					fseek(rgfpReadMfc, 0L, SEEK_END);
					// get current file pointer position
					rgiTrainFileLen[i] = ftell(rgfpReadMfc) / sizeof(double);
					rgiTrainFileLen[i] /= FEATURE_LEN;
					fseek(rgfpReadMfc, 0, SEEK_SET);

					rgdMfcTrainData[i] = (double**)malloc(sizeof(double*) * rgiTrainFileLen[i]);
					if (iInitCount == 1) {
						rgGmmParameter[i] = (GMMParameter*)malloc(sizeof(GMMParameter));
						rgGmmExp[i] = (GMMExpectation*)malloc(sizeof(GMMExpectation));
						rgGmmExp[i]->weights = (double**)malloc(sizeof(double*) * rgiTrainFileLen[i]);
						for (int k = 0; k < rgiTrainFileLen[i]; k++) {
							rgGmmExp[i]->weights[k] = (double*)malloc(sizeof(double) * NUM_OF_MIXTURE);
						}
					}

					for (int k = 0; k < rgiTrainFileLen[i]; k++) {
						rgdMfcTrainData[i][k] = (double*)malloc(sizeof(double) * FEATURE_LEN);
						if ((fread(rgdMfcTrainData[i][k], sizeof(double), FEATURE_LEN, rgfpReadMfc)) == 0) {
							printf("Break! The buffer is insufficient.\n");
							continue;
						}
					}
					if (iInitCount == 1) {

						for (int j = 0; j < NUM_OF_MIXTURE; j++) {
							for (int k = 0; k < FEATURE_LEN; k++) {

								rgGmmParameter[i]->mean[j][k] = rgdMfcTrainData[i][j * 4][k];
							}
						}
						KmeansAlogorithm(rgdMfcTrainData[i], rgGmmParameter[i], rgiTrainFileLen[i]);

						for (int j = 0; j < NUM_OF_MIXTURE; j++) {
							rgGmmParameter[i]->alpa[j] = 1.0 / NUM_OF_MIXTURE;
						}
						for (int j = 0; j < rgiTrainFileLen[i]; j++) {
							for (int k = 0; k < NUM_OF_MIXTURE; k++) {
								rgGmmExp[i]->weights[j][k] = 1.0 / NUM_OF_MIXTURE;
							}
						}
					}
					EmAlgorithmBasedGmmParameter(rgdMfcTrainData[i], rgGmmParameter[i], rgGmmExp[i], rgiTrainFileLen[i]);

					for (int k = 0; k < rgiTrainFileLen[i]; k++) {
						free(rgdMfcTrainData[i][k]);
						rgdMfcTrainData[i][k] = NULL;
					}
					free(rgdMfcTrainData[i]);
					rgdMfcTrainData[i] = NULL;


			}
			// Save parameter
			PCADiagonalizeCovarianceMatrix(rgGmmParameter[i]);
			for (int k = 0; k < rgiTrainFileLen[i]; k++) {
				free(rgGmmExp[i]->weights[k]);
				rgGmmExp[i]->weights[k] = NULL;
			}
			free(rgGmmExp[i]->weights);
			rgGmmExp[i]->weights = NULL;
			free(rgGmmExp[i]);
			rgGmmExp[i] = NULL;

			fwrite(rgGmmParameter[i], sizeof(GMMParameter), 1, fpGMMParamWrite);
			free(rgGmmParameter[i]);
			rgGmmParameter[i] = NULL;
		}

	}

	for (int i = 0; i < NUM_OF_CLASS; i++) {
		fclose(fpReadClassFile[i]);
	}
	fclose(fpReadFile);
	fclose(rgfpReadMfc);
	fclose(fpGMMParamWrite);
	getchar();
}

double Recognition(double **dpTestBuf, GMMParameter *pGmmParameter, GMMExpectation *pGmmExp, int iFileLen) {

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

	MatrixXd mxCovarianceMatrix = MatrixXd(FEATURE_LEN, FEATURE_LEN);

	MatrixXd mxSortedEigenValueMtx = MatrixXd(PCA_LEN, PCA_LEN);
	MatrixXd mxSortedEigenVectorMtx = MatrixXd(FEATURE_LEN, PCA_LEN);
	MatrixXd mxInputVector = MatrixXd(FEATURE_LEN, 1);
	MatrixXd mxMeanVector = MatrixXd(FEATURE_LEN, 1);
	MatrixXd mxPCAInput = MatrixXd(PCA_LEN, 1);
	MatrixXd mxPCAMean = MatrixXd(PCA_LEN, 1);
	int iSortingIdx[FEATURE_LEN] = { 0, };
	int iTempArg = 0;
	double dTempProb = 1;

	mxSortedEigenValueMtx.setZero(PCA_LEN, PCA_LEN);
	mxSortedEigenVectorMtx.setZero(FEATURE_LEN, PCA_LEN);

	double dTemp1 = 0, dTemp2 = 0, dTemp3 = 0;
	double dTempArray[FEATURE_LEN] = { 0, };
	for (int j = 0; j < FEATURE_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			mxCovarianceMatrix(j, m) = rgdCovariance[j][m];
		}
	}

	EigenSolver<MatrixXd> es(mxCovarianceMatrix);
	MatrixXd EigenValueMtx = es.pseudoEigenvalueMatrix();
	MatrixXd EigenVectorMtx = es.pseudoEigenvectors();

	for (int j = 0; j < FEATURE_LEN; j++) {
		//printf("EigenValueMtx(m, m) %f , ", EigenValueMtx(j, j));
		for (int m = 0; m < FEATURE_LEN; m++) {
			if (EigenValueMtx(j, j) < EigenValueMtx(m, m))
				iSortingIdx[j]++; // 작을수록 큰 index를 갖음.
		}
	}

	for (int j = 0; j < PCA_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			if (iSortingIdx[m] == j) {
				iTempArg = m;
				break;
			}
		}
		//printf("iTEmpArg %d , j %d  ,%f , %f \n", iTempArg, j, EigenValueMtx(iTempArg, iTempArg), mxSortedEigenValueMtx(j, j));
		mxSortedEigenValueMtx(j, j) = EigenValueMtx(iTempArg, iTempArg);
		for (int m = 0; m < FEATURE_LEN; m++) {
			mxSortedEigenVectorMtx(m, j) = EigenVectorMtx(m, iTempArg);
		}
	}

	for (int i = 0; i < FEATURE_LEN; i++) {
		mxInputVector(i,0) = pdFeature[i];
		mxMeanVector(i,0) = pdMean[i];
	}

	mxPCAInput = (mxInputVector.transpose() * mxSortedEigenVectorMtx).transpose();
	mxPCAMean = (mxMeanVector.transpose() * mxSortedEigenVectorMtx).transpose();

	for (int i = 0; i < PCA_LEN; i++) {
		dTempProb *= (1.0 / sqrt(2.0 * PI))* (1.0 / sqrt(mxSortedEigenValueMtx(i,i))) *exp((-1 / 2.0) * pow((mxPCAInput(i, 0) - mxPCAMean(i, 0)), 2.0) / (mxSortedEigenValueMtx(i, i)));
	}
	return dTempProb;

}

void EmAlgorithmBasedGmmParameter(double **pdInBuf, GMMParameter *pGmmParam, GMMExpectation *pGmmExp, int numOfFrames) {
	int count_ = 0;
	double dTemp1 = 0, dTemp2 = 0, dTempAft = 0, dTempBf = 0;
	MatrixXd mxMeanVector = MatrixXd(FEATURE_LEN, 1);
	MatrixXd mxTransVector = MatrixXd(1, FEATURE_LEN);
	MatrixXd mxResultVector = MatrixXd(FEATURE_LEN, FEATURE_LEN);

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
					pGmmExp->weights[i][k] = probability(pdInBuf[i], pGmmParam->mean[k], pGmmParam->covariance[k]) * pGmmParam->alpa[k];
					
					//printf(" probability(pdInBuf[i], pGmmParam->mean[k], pGmmParam->covariance[k]) %f \n", probability(pdInBuf[i], pGmmParam->mean[k], pGmmParam->covariance[k]));
					dTemp2 += pGmmExp->weights[i][k];
				}
				for (int k = 0; k < NUM_OF_MIXTURE; k++) {
					pGmmExp->weights[i][k] = pGmmExp->weights[i][k] / dTemp2;
					//printf("pGmmExp->weights[i][k] %d-th %f \n", k, pGmmExp->weights[i][k]);
				}
				
				dTemp2 = 0;
				//printf("jcboo  %d-th \n", i);
			}
			//printf("jcboo 1 \n");

			memset(pGmmExp->nOfKey, 0, sizeof(double) * NUM_OF_MIXTURE);

			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				for (int i = 0; i < numOfFrames; i++) {
					pGmmParam->alpa[k] += pGmmExp->weights[i][k];
				}
				pGmmExp->nOfKey[k] = pGmmParam->alpa[k];
				pGmmParam->alpa[k] /= (double)numOfFrames;
			}

		for (int d = 0; d < FEATURE_LEN; d++) {
			for (int k = 0; k < NUM_OF_MIXTURE; k++) {
				for (int i = 0; i < numOfFrames; i++) {
					pGmmParam->mean[k][d] += pGmmExp->weights[i][k] * pdInBuf[i][d];
				}
				pGmmParam->mean[k][d] /= pGmmExp->nOfKey[k];
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
				mxResultVector += mxMeanVector* mxMeanVector.transpose() * pGmmExp->weights[i][k];
			}
			mxResultVector /= pGmmExp->nOfKey[k]; // 상수값 나눌 시 원소 전체 나눠지는지 확인.
			for (int j = 0; j < FEATURE_LEN; j++) {
				for (int m = 0; m < FEATURE_LEN; m++) {
					pGmmParam->covariance[k][j][m] = mxResultVector(j, m); // 상수값 나눌 시 원소 전체 나눠지는지 확인.
					//printf("%f  ", pGmmParam->covariance[k][j][m]);
				}
				//printf(" \n ");
			}
			//printf(" End \n ");

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
	int(*Selection)[NUM_OF_MIXTURE] = { 0, };
	int iCountofSelected[NUM_OF_MIXTURE] = { 0, };

	Selection = (int(*)[NUM_OF_MIXTURE])malloc(sizeof(int[NUM_OF_MIXTURE]) * numOfFrames);
	memset(Selection, 0, sizeof(int[NUM_OF_MIXTURE]) * numOfFrames);
	while (1) {
		//2)
		printf(" go ! \n");
		count_++;
		for (int i = 0; i < numOfFrames; i++) {
			dTempDist = DistanceToCenter(pdInBuf[i], pGmmParam->mean[0]);
			for (int j = 0; j < NUM_OF_MIXTURE; j++) {
				if (dTempDist >= DistanceToCenter(pdInBuf[i], pGmmParam->mean[j])) {
					arg_temp = j;
					dTempDist = DistanceToCenter(pdInBuf[i], pGmmParam->mean[j]);
				}
			}
			Selection[i][arg_temp] = 1;
			arg_temp = 0;
		}

		//3)
		cost = 0;
		for (int i = 0; i < numOfFrames; i++) {
				for (int j = 0; j < NUM_OF_MIXTURE; j++) {
					if (Selection[i][j])
						cost += DistanceToCenter(pdInBuf[i], pGmmParam->mean[j]);
				}
		}
		printf(" 1 prev cost %f, current cost %f \n", cost_before, cost);

		if (count_ == 1 || abs(cost - cost_before) >= THRESHOLD_OF_DISTANCE) {
			printf(" prev cost %f, current cost %f \n", cost_before, cost);
			cost_before = cost;
		}
		else {

			//calc var;
			memset(iCountofSelected, 0, sizeof(int) * NUM_OF_MIXTURE);
			for (int j = 0; j < NUM_OF_MIXTURE; j++) {
				covarianceMatrix_k.setZero(FEATURE_LEN, FEATURE_LEN);
				for (int i = 0; i < numOfFrames; i++) {

					if (Selection[i][j]) {
						iCountofSelected[j]++;
						for (int z = 0; z < FEATURE_LEN; z++)
							meanVector_k(z, 0) = pdInBuf[i][z] - pGmmParam->mean[j][z];

						covarianceMatrix_k += meanVector_k * meanVector_k.transpose();
					}
				}
				covarianceMatrix_k /= (double)iCountofSelected[j];

				for (int m = 0; m < FEATURE_LEN; m++) {
					for (int u = 0; u< FEATURE_LEN; u++) {
						pGmmParam->covariance[j][m][u] = covarianceMatrix_k(m, u);
						//printf("%f ", pGmmParam->covariance[j][m][u]);
					}
					//printf("\n");
				}
				//printf(" end!!! \n");
			}
			break;
		}
		printf("update mean! \n");
		//1)
		for (int i = 0; i < NUM_OF_MIXTURE; i++) {
			memset(pGmmParam->mean[i], 0, sizeof(double) * FEATURE_LEN);
		}
		memset(iCountofSelected, 0, sizeof(int) * NUM_OF_MIXTURE);
		for (int i = 0; i < numOfFrames; i++) {
			for (int j = 0; j < NUM_OF_MIXTURE; j++) {
				if (Selection[i][j]) {
					UpdateMean(pdInBuf[i], pGmmParam->mean[j]);
					iCountofSelected[j]++;
				}
			}
		}
		for (int j = 0; j < NUM_OF_MIXTURE; j++) {

			if (iCountofSelected[j] == 0)
				continue;

			for (int k = 0; k < FEATURE_LEN; k++) {
				pGmmParam->mean[j][k] /= (double)iCountofSelected[j];
			}
		}
	}
	free(Selection);
	Selection = NULL;
}

double DistanceToCenter(double *pdInBuf, double *pMean) {
	
	double dDistance = 0;
	for (int i = 0; i < FEATURE_LEN; i++) {
		dDistance += pow(pdInBuf[i] - pMean[i], 2.0);
	}
	return dDistance;
}

void UpdateMean(double *pdInBuf, double *pMean) {

	for (int k = 0; k < FEATURE_LEN; k++) {
		pMean[k] += pdInBuf[k];
	}
}

void PCADiagonalizeCovarianceMatrix(GMMParameter *pGmmParam) {

	for (int k = 0; k < NUM_OF_MIXTURE; k++) {
		PCAConvertMixtureDiagonalMatrix(pGmmParam->mean[k], pGmmParam->covariance[k], pGmmParam->eigenVector[k]);
	}
}

void PCAConvertMixtureDiagonalMatrix(double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]) {

	double dTempProb = 1.0, dTempEigen = 0;
	MatrixXd CovMatrx = MatrixXd(FEATURE_LEN, FEATURE_LEN);
	MatrixXd MeanMatrx = MatrixXd(1, FEATURE_LEN);
	MatrixXd mxSortedMean = MatrixXd(1, PCA_LEN);
	MatrixXd mxSortedEigenValueMtx = MatrixXd(PCA_LEN, PCA_LEN);
	MatrixXd mxSortedEigenVectorMtx = MatrixXd(FEATURE_LEN, PCA_LEN);
	int iSortingIdx[FEATURE_LEN] = { 0, };
	int iTempArg = 0;
	mxSortedEigenValueMtx.setZero(PCA_LEN, PCA_LEN);
	mxSortedEigenVectorMtx.setZero(FEATURE_LEN, PCA_LEN);

	for (int j = 0; j < FEATURE_LEN; j++) {
		MeanMatrx(0, j) = pdMean[j];
		for (int m = 0; m < FEATURE_LEN; m++) {
			CovMatrx(j, m) = rgdCovariance[j][m];
		}
	}

	EigenSolver<MatrixXd> es(CovMatrx);
	MatrixXd EigenValueMtx = es.pseudoEigenvalueMatrix();
	MatrixXd EigenVectorMtx = es.pseudoEigenvectors();

	for (int j = 0; j < FEATURE_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			if (EigenValueMtx(j, j) < EigenValueMtx(m, m))
				iSortingIdx[j]++; // 작을수록 큰 index를 갖음.
		}
	}

	for (int j = 0; j < PCA_LEN; j++) {
		for (int m = 0; m < FEATURE_LEN; m++) {
			if (iSortingIdx[m] == j) {
				iTempArg = m;
				break;
			}
		}
		//printf("iTEmpArg %d , j %d  ,%f , %f \n", iTempArg, j, EigenValueMtx(iTempArg, iTempArg), mxSortedEigenValueMtx(j, j));
		mxSortedEigenValueMtx(j, j) = EigenValueMtx(iTempArg, iTempArg);
		for (int m = 0; m < FEATURE_LEN; m++) {
			mxSortedEigenVectorMtx(m, j) = EigenVectorMtx(m, iTempArg);
		}
	}

	mxSortedMean = MeanMatrx * mxSortedEigenVectorMtx;
	memset(pdMean, 0, sizeof(double) * FEATURE_LEN);
	for (int i = 0; i < PCA_LEN; i++) {
		pdMean[i] = (mxSortedMean)(0, i);
		memset(rgdCovariance[i], 0, sizeof(double) * FEATURE_LEN);
		rgdCovariance[i][i] = mxSortedEigenValueMtx(i, i);
		for (int j = 0; j < FEATURE_LEN; j++) {
			rgdEigenVector[j][i] = mxSortedEigenVectorMtx(j, i);
		}
	}
	return;
}