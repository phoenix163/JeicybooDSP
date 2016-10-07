/*
Viterbi algorithm implementation.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

refer to http://www1.icsi.berkeley.edu/Speech/docs/HTKBook/node104_ct.html
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
#define FEATURE_LEN 12 // 특징 벡터 1개에 대한 길이.
#define NUM_OF_MIXTURE 4 // GMM 1개에 대한 Mixture 갯수.
#define NUM_OF_CLASS 1 // 1개 단어에 대해서 인식.
#define NUM_OF_TEST 25 // 총 25개의 Test
#define NUM_OF_STATE 6 // 1개 단어를 구성하는 state는 6개.

typedef struct gGmmParameter{
	double alpa[NUM_OF_MIXTURE];
	double mean[NUM_OF_MIXTURE][FEATURE_LEN];
	double covariance[NUM_OF_MIXTURE][FEATURE_LEN][FEATURE_LEN];
	double eigenVector[NUM_OF_MIXTURE][FEATURE_LEN][PCA_LEN];
}GMMParameter;

typedef struct gHmmParameter{
	GMMParameter gMMParam[NUM_OF_STATE];
	double transProb[NUM_OF_STATE][NUM_OF_STATE];
}HMMParameter;

typedef struct gSaveProb{
	double **dHMMProb;
	int **prevIdx;
}SaveProb;

double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]);
double GMMRecognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen);
double HMMRecognition(double **dpTestBuf, HMMParameter *pHmmParameter, int iFileLen);

void main(int argc, char** argv) {
	FILE *fpReadFile, *rgfpReadMfc, *fpReadClassFile[NUM_OF_CLASS], *fpReadParam;
	double **rgdMfcTestData[NUM_OF_CLASS] = { 0, };
	double dpAccumLogLikelihood[NUM_OF_CLASS] = { 0, };
	HMMParameter *rgHmmParameter[NUM_OF_CLASS] = { 0, };
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
			rgHmmParameter[i] = (HMMParameter*)malloc(sizeof(HMMParameter));
			fread(rgHmmParameter[i], sizeof(HMMParameter), 1, fpReadParam);
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
					//dpAccumLogLikelihood[u] = GMMRecognition(rgdMfcTestData[i], rgGmmParameter[u], rgiTestFileLen[i]); GMM recognition
					dpAccumLogLikelihood[u] = HMMRecognition(rgdMfcTestData[i], rgHmmParameter[u], rgiTestFileLen[i]);
					//printf(" %d-th Test result %f prob. \n", u + 1, dpAccumLogLikelihood[u]);
					if (u == 0) {
						dMax = dpAccumLogLikelihood[0];
						dArg = 0;
					}
					else if (dMax < dpAccumLogLikelihood[u]) {
						dMax = dpAccumLogLikelihood[u];
						dArg = u;
					}
					printf(" %d-th class probability %f \n", u + 1, dpAccumLogLikelihood[u]);
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

		for (int i = 0; i < NUM_OF_CLASS; i++) {
			free(rgHmmParameter[i]);
			rgHmmParameter[i] = NULL;
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

double HMMRecognition(double **dpTestBuf, HMMParameter *pHmmParameter, int iFileLen) {

	double dpAccumLogLikelihood = 0, dTempProb = 0, dTempCalc = 0;
	int dTempIndx = 0;
	SaveProb sProb;
	double *dDecodingReslt = NULL;

	dDecodingReslt = (double*)malloc(sizeof(double) * (iFileLen - 1));
	memset(dDecodingReslt, 0, sizeof(double) * (iFileLen - 1));

	sProb.dHMMProb = (double**)malloc(sizeof(double*) * NUM_OF_STATE);
	sProb.prevIdx = (int**)malloc(sizeof(int*) * NUM_OF_STATE);
	memset(sProb.dHMMProb, 0, sizeof(double*) * NUM_OF_STATE);
	memset(sProb.prevIdx, 0, sizeof(int*) * NUM_OF_STATE);

	for (int u = 0; u < NUM_OF_STATE; u++) {
		sProb.dHMMProb[u] = (double*)malloc(sizeof(double) * iFileLen);
		memset(sProb.dHMMProb[u], 0, sizeof(double) * iFileLen);
		sProb.prevIdx[u] = (int*)malloc(sizeof(int) * iFileLen);
		memset(sProb.prevIdx[u], 0, sizeof(int) * iFileLen);
	}

	for (int i = 0; i < iFileLen; i++) {

		if (i == 0) {
			for (int m = 0; m < NUM_OF_STATE; m++) { // current
				for (int k = 0; k < NUM_OF_MIXTURE; k++) {
					sProb.dHMMProb[m][0] += pHmmParameter->gMMParam[m].alpa[k] * probability(dpTestBuf[i], pHmmParameter->gMMParam[m].mean[k], pHmmParameter->gMMParam[m].covariance[k], pHmmParameter->gMMParam[m].eigenVector[k]);
				}
				sProb.dHMMProb[m][0] = log(sProb.dHMMProb[m][0]) + log(1.0 / (double)NUM_OF_STATE);
			}
			continue;
		}
		for (int m = 0; m < NUM_OF_STATE; m++) { // current
			for (int u = 0; u < NUM_OF_STATE; u++) { // prev
				dTempCalc = 0;
				for (int k = 0; k < NUM_OF_MIXTURE; k++) {
					dTempCalc += pHmmParameter->gMMParam[m].alpa[k] * probability(dpTestBuf[i], pHmmParameter->gMMParam[m].mean[k], pHmmParameter->gMMParam[m].covariance[k], pHmmParameter->gMMParam[m].eigenVector[k]);
				}
				dTempCalc = log(sProb.dHMMProb[u][i - 1]) + log(pHmmParameter->transProb[u][m]) + log(dTempCalc); // *가 +로 바뀜.(로그를 취했으므로)

				if (u == 0) {
					sProb.dHMMProb[m][i] = dTempCalc;
					sProb.prevIdx[m][i] = 0;
				} else if (sProb.dHMMProb[m][i] < dTempCalc) {
					sProb.dHMMProb[m][i] = dTempCalc;
					sProb.prevIdx[m][i] = u;
				}
			}
		}
	}

	for (int i = iFileLen - 1; i > 0; i--) {

		// find max accumulated probability
		for (int m = 0; m < NUM_OF_STATE; m++) { // 

			if (m == 0) {
				dTempProb = sProb.dHMMProb[m][i];
				dTempIndx = 0;
			}else if (sProb.dHMMProb[m][i] > dTempProb) {
				dTempProb = sProb.dHMMProb[m][i];
				dTempIndx = m;
			}
		}
		printf("max accumulated prob %f \n", dTempProb);
		dDecodingReslt[i] = dTempIndx;
		dTempIndx = sProb.prevIdx[dTempIndx][i];
	}
	// decoding result !
	printf("decoding result ! \n");
	for (int i = 0; i < iFileLen - 1; i++) {
		printf("%d ,", dDecodingReslt[i]);
	}
	printf("\n");

	for (int u = 0; u < NUM_OF_STATE; u++) {
		free(sProb.dHMMProb[u]);// = (double*)malloc(sizeof(double) * iFileLen);
		free(sProb.prevIdx[u]);
		sProb.dHMMProb[u] = NULL;
		sProb.prevIdx[u] = NULL;
	}
	free(sProb.dHMMProb);
	free(sProb.prevIdx);
	free(dDecodingReslt);
	sProb.dHMMProb = NULL;
	sProb.prevIdx = NULL;
	dDecodingReslt = NULL;
	return dTempProb;
}

double probability(double *pdFeature, double *pdMean, double rgdCovariance[FEATURE_LEN][FEATURE_LEN], double rgdEigenVector[FEATURE_LEN][PCA_LEN]) {
	
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
		dTempProb *= (1.0 / sqrt(2.0 * PI))* (1.0 / sqrt(rgdCovariance[i][i])) *exp((-1 / 2.0) * pow((InputMatrx(0, i) - pdMean[i]), 2.0) / (rgdCovariance[i][i]));
	}
	return dTempProb;
}


#if 0
double GMMRecognition(double **dpTestBuf, GMMParameter *pGmmParameter, int iFileLen) {
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
		dTempProb *= (1.0 / sqrt(2.0 * PI))* (1.0 / sqrt(rgdCovariance[i][i])) *exp((-1 / 2.0) * pow((InputMatrx(0, i) - pdMean[i]), 2.0) / (rgdCovariance[i][i]));
	}
	return dTempProb;
}
#endif