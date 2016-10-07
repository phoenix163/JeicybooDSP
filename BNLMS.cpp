/*
LMS(Least Mean Square) Filter. (Adaptive Filter)
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

구현내용 정리.

Reference Signal과 Estimation한 Signal의 차이(Least Mean Square error)가 최소가 되도록
Estimation System의 FilterCoefficient를 실시간 update하는 filter이다.
현재 구현내용은 가장 기본적인 LMS version이다.
(매 Sample 마다 coefficient가 update, 8kHz의 경우 1초에 8000번 update)
Haykin의 Adaptive Filter Theory 책 참고 할 것.

Tunning Value인 MU value조절을 잘 해서, Filter의 출력이 발산하지 않도록, 함.

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
//#include<Eigen/Dense>
//#include<Eigen/Eigenvalues>
//using Eigen::MatrixXd;

#define THRESHOLD 0
#define FALSE 0
#define TRUE 1
#define PI 3.141592
#define BLOCK_LEN 1024 // 1000*(512/16000)= 32ms.
#define FILTER_LEN 128
#define KEEP_LEN 127
#define MU 0.01
#define COMPENSATION 0.00001


bool DoubleTalkState(short *rgsInputBuffer, short *rgsRefBuffer);
bool BlockLMSFilter(short *rgsInputBuffer, short *rgsRefBuffer, short *rgdLMSFilteringBuffer, short *rgsError);
//double CalcMaxMu(short* rgsInputBuffer);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpReadInput, *fpReadRef, *fpWriteEst, *fpWriteErr;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, }, rgsRefBuffer[BLOCK_LEN] = { 0, }, rgdLMSFilteringBuffer[BLOCK_LEN] = { 0, }, rgsError[BLOCK_LEN] = { 0, };

	if (argc != 5) {
		printf("path를 4개 입력해야 합니다.\n"); // input path, output path
		return;
	} else {
		for (int i = 1; i < 5; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpReadInput = fopen(argv[1], "rb")) == NULL)
		printf("Read Input File Open Error\n");

	if ((fpReadRef = fopen(argv[2], "rb")) == NULL)
		printf("Read Ref File Open Error\n");

	if ((fpWriteEst = fopen(argv[3], "wb")) == NULL)
		printf("Est Write File Open Error\n");

	if ((fpWriteErr = fopen(argv[4], "wb")) == NULL)
		printf("Err Write File Open Error\n");

	// Read한 Wav 파일 맨 앞의 Header 44Byte 만큼 Write할 Wav파일에 write.
	fread(rgcHeader, 1, 44, fpReadInput);
	//fread(rgcHeader, 1, 44, fpReadRef);
	//fwrite(rgcHeader, 1, 44, fpWrite);

	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpReadInput)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}

		if ((fread(rgsRefBuffer, sizeof(short), BLOCK_LEN, fpReadRef)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}

		if (BlockLMSFilter(rgsInputBuffer, rgsRefBuffer, rgdLMSFilteringBuffer, rgsError)) { // TRUE가 return되었을 경우에만 file에 쓴다.

			fwrite(rgdLMSFilteringBuffer, sizeof(short), BLOCK_LEN, fpWriteEst);
			fwrite(rgsError, sizeof(short), BLOCK_LEN, fpWriteErr);
		}
	}
	printf("Processing End\n");
	fclose(fpReadInput);
	fclose(fpReadRef);
	fclose(fpWriteEst);
	fclose(fpWriteErr);
	getchar();
	return;
}

bool BlockLMSFilter(short *rgsInputBuffer, short *rgsRefBuffer, short *rgsLMSFilteringBuffer, short *rgsError) {

	static short rgssKeepInput[KEEP_LEN] = { 0, };
	static short rgssKeepRefer[KEEP_LEN] = { 0, };
	static double rgsdCoefficient[FILTER_LEN] = { 0, };
	short rgsProcessingInput[BLOCK_LEN + KEEP_LEN] = { 0, };
	short rgsProcessingRefer[BLOCK_LEN + KEEP_LEN] = { 0, };
	double dNormalizingFactor = 0, dTempAcc[FILTER_LEN] = { 0, }; // using Normalizing Factor.
	static int iNumOfIteration = 0;
	double dTempCalc = 0.0; // short로 바로 담기보다는 계산을 다 하고, 맨 마지막에 short 인자에 넣는다.

	iNumOfIteration++;
	//CalcMaxMu(rgsInputBuffer);
	// copy to processing input.
	memcpy(rgsProcessingInput, rgssKeepInput, KEEP_LEN * sizeof(short));
	memcpy(rgsProcessingInput + KEEP_LEN, rgsInputBuffer, BLOCK_LEN * sizeof(short));
	// copy to processing refer.
	memcpy(rgsProcessingRefer, rgssKeepRefer, KEEP_LEN * sizeof(short));
	memcpy(rgsProcessingRefer + KEEP_LEN, rgsRefBuffer, BLOCK_LEN * sizeof(short));

	for (int i = 0; i < BLOCK_LEN; i++) {
		// Estimation output
		dTempCalc = 0;
		for (int j = 0; j < FILTER_LEN; j++) {
			dTempCalc += rgsdCoefficient[FILTER_LEN - 1 - j] * rgsProcessingInput[j + i];
		}
		rgsLMSFilteringBuffer[i] = dTempCalc; //dTempCalc 란 double인자에 누적값을 넣고, 마지막에 short형 output buffer에 넣는다.
		rgsError[i] = rgsRefBuffer[i] - rgsLMSFilteringBuffer[i];
	}

	if (!DoubleTalkState(rgsProcessingInput, rgsProcessingRefer)) {
		// Estimation Coefficient
		printf("It is not double talk state! \n");
		memset(dTempAcc, 0, sizeof(dTempAcc));
		for (int i = 0; i < BLOCK_LEN; i++) {
			dNormalizingFactor = 0;
			for (int j = 0; j < FILTER_LEN; j++) {
				// Calc Normalizing Factor
				dNormalizingFactor += pow((double)rgsProcessingInput[i + j], 2.0);
			}
			for (int j = 0; j < FILTER_LEN; j++) {
				dTempAcc[j] += 2.0 * (double)rgsProcessingInput[j + i] * MU * (rgsRefBuffer[i] - rgsLMSFilteringBuffer[i]) / (dNormalizingFactor + COMPENSATION);
			}
		}

		for (int i = 0; i < FILTER_LEN; i++) {
			dTempAcc[i] /= BLOCK_LEN;
			rgsdCoefficient[i] += dTempAcc[i];
		}
	}

	//Keeping Buffer
	memcpy(rgssKeepInput, &rgsInputBuffer[BLOCK_LEN - KEEP_LEN], sizeof(short) * KEEP_LEN);
	memcpy(rgssKeepRefer, &rgsRefBuffer[BLOCK_LEN - KEEP_LEN], sizeof(short) * KEEP_LEN);

	if (iNumOfIteration > 1)
		return TRUE;
	else
		return FALSE;
}

bool DoubleTalkState(short *rgsProcessingInput, short *rgsProcessingRefer) {
	double dMax = 0;
	double dAutoCorrelation[BLOCK_LEN] = { 0, };

	for (int k = 0; k < BLOCK_LEN; k++) {
		for (int i = 0; i < 2 * BLOCK_LEN - k; i++) {
			dAutoCorrelation[k] += rgsProcessingInput[i] * rgsProcessingRefer[i + k];
		}
		dAutoCorrelation[k] /= (double)(2 * BLOCK_LEN - k);
	}
	//Calculate max value of CrossCorrelation
	for (int i = 0; i < BLOCK_LEN; i++) {
		if (dAutoCorrelation[i] > dMax) {
			dMax = dAutoCorrelation[i];
		}
	}
	if (dMax > THRESHOLD)
		return FALSE;
	else
		return TRUE;


}

#if 0
double CalcMaxMu(short* rgsInputBuffer) {
	static short rgssKeepInput[BLOCK_LEN] = { 0, };
	short rgsProcessingInput[BLOCK_LEN * 2] = { 0, };
	double rgdResult[BLOCK_LEN] = { 0, };
	double rgdEigenValues[BLOCK_LEN] = { 0, };
	MatrixXd mxAutoCorMaxtrix = MatrixXd(BLOCK_LEN, BLOCK_LEN);
	MatrixXd mxEigenValues = MatrixXd(BLOCK_LEN, 1);
	Eigen::EigenSolver<Eigen::MatrixXcd> es;
	double dTemp = 0;
	MatrixXd D = es.eigenvalues().asDiagonal();

	memcpy(rgsProcessingInput, rgssKeepInput, BLOCK_LEN * sizeof(short));
	memcpy(rgsProcessingInput + BLOCK_LEN, rgsInputBuffer, BLOCK_LEN * sizeof(short));

	for (int i = 0; i < BLOCK_LEN; i++) {
		for (int j = 0; j < BLOCK_LEN * 2 - i; j++) {
			rgdResult[i] += rgsProcessingInput[j] * rgsProcessingInput[j + i];
		}
		rgdResult[i] /= BLOCK_LEN * 2 - i;
	}
	// Calc mxToeplitzMatrix
	for (int i = 0; i < BLOCK_LEN; i++) {
		for (int j = 0; j < BLOCK_LEN; j++) {
			mxAutoCorMaxtrix(i, j) = rgdResult[abs(i - j)];
		}
	}
	
#if 0
	for (int i = 0; i < BLOCK_LEN; i++) {
		if (dTemp < mxEigenValues(i, 1))
			dTemp = mxEigenValues(i, 1);
	}
#endif
	memcpy(rgssKeepInput, rgsInputBuffer, sizeof(short) * BLOCK_LEN);
	//printf(" Maximum Boundary of Mu %f \n", 2.0 / dTemp);
	return  2.0 / dTemp; // Maximum Boundary of Mu   2 / Maxof(EigenValues)
}
#endif