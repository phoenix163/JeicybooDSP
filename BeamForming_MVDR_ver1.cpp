/*
Beamforming using 2 smartphone mic.

This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 48kHz SamplingRate, stereo channel, 16bit per sample.

구현 시 고려사항.

Steering Vector가 복소수 vector이므로, https://eigen.tuxfamily.org/dox/group__matrixtypedefs.html
Steering Vector변수형은 MatrixXcd를 사용하였다.

결국 MVDR기준의 Filter Coefficient는 w = R−1c(cHR−1c)−1 로 구해지고, 이를 통해 각 bin 별 복소수 w(k)를 구할 수 있다.
https://www.google.co.kr/#q=Two_Sensor_Beamforming_Algorithm

MVDR는 noise의 출력을 최소화 해야 하기 때문에, noise인지 speech인지 detection이 필요하다, VAD function을 사용해야 한다.

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
#include<Eigen/Dense>
using namespace Eigen;
#include <Eigen/Eigenvalues>

#define FALSE 0
#define TRUE 1
#define THRESHOLD_OF_ENERGY 700.0
#define THRESHOLD_OF_ZCR 200.0
#define SAMPLING_RATE 16000.0
#define FFT_PROCESSING_LEN 1024
#define BLOCK_LEN 512 // 1000*(512/16000)= 32ms.
#define KEEP_LEN 511
#define CHANNEL 2
#define PI 3.141592
#define SPEED_OF_SOUND 34000.0 // (cm/sec) ,340.29(m/s)
#define DISTANCE_OF_MIC 800.0 //

bool ProcessMVDR(short *rgsInputBufferL, short *rgsInputBufferR, int iBlockLen, short *rgsOutputBuffer, double dTime, double(*rgdSpatialCorr)[CHANNEL]);
bool VoiceActivityDetection(short *rgsInputBuffer, int iFrameCount);
void EstimateSpatialCorrMtx(short *rgsTempBufferL, short *rgsTempBufferR, int iNumOfIteration, double(*rgdSpatialCorr)[CHANNEL], int iFrameCount);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead1, *fpRead2, *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBufferL[BLOCK_LEN] = { 0, };
	short rgsInputBufferR[BLOCK_LEN] = { 0, };
	short rgsOutputBuffer[BLOCK_LEN] = { 0, };
	short rgsTempBufferL[BLOCK_LEN * 2] = { 0, }, rgsTempBufferR[BLOCK_LEN * 2] = { 0, };
	double rgdSpatialCorr[CHANNEL][CHANNEL] = { 0, };
	double dAngle = 0 , dTime = 0;
	int iNumOfIteration = 0;

	dTime = (DISTANCE_OF_MIC / SPEED_OF_SOUND) * sin(dAngle);

	if (argc != 4) {
		printf("path를 3개 입력해야 합니다.\n"); // input path, output path
		return;
	}
	else {
		for (int i = 1; i < 4; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpRead1 = fopen(argv[1], "rb")) == NULL)
		printf("Read File Open Error\n");

	if ((fpRead2 = fopen(argv[2], "rb")) == NULL)
		printf("Read File Open Error\n");

	if ((fpWrite = fopen(argv[3], "wb")) == NULL)
		printf("Write File Open Error\n");

	// Read한 Wav 파일 맨 앞의 Header 44Byte 만큼 Write할 Wav파일에 write.
	fread(rgcHeader, 1, 44, fpRead1);
	fread(rgcHeader, 1, 44, fpRead2);

	while (true)
	{
		if ((fread(rgsInputBufferL, sizeof(short), BLOCK_LEN, fpRead1)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		if ((fread(rgsInputBufferR, sizeof(short), BLOCK_LEN, fpRead2)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}

		if (!VoiceActivityDetection(rgsInputBufferL, BLOCK_LEN)) {
			iNumOfIteration++;
			if (iNumOfIteration > 1) {
				memcpy(rgsTempBufferL + BLOCK_LEN, rgsInputBufferL, sizeof(rgsInputBufferL));
				memcpy(rgsTempBufferR + BLOCK_LEN, rgsInputBufferR, sizeof(rgsInputBufferR));
				EstimateSpatialCorrMtx(rgsTempBufferL, rgsTempBufferR, iNumOfIteration, rgdSpatialCorr, BLOCK_LEN * 2);
			}
			memcpy(rgsTempBufferL, rgsInputBufferL, sizeof(rgsInputBufferL));
			memcpy(rgsTempBufferR, rgsInputBufferR, sizeof(rgsInputBufferR));

		}
		else {

			iNumOfIteration = 0;
		}

		if (ProcessMVDR(rgsInputBufferL, rgsInputBufferR, BLOCK_LEN, rgsOutputBuffer, dTime, rgdSpatialCorr))
			fwrite(rgsOutputBuffer, sizeof(short), BLOCK_LEN, fpWrite);


	}
	printf("Processing End\n");
	fclose(fpRead1);
	fclose(fpRead2);
	fclose(fpWrite);
	getchar();
	return;
}

bool ProcessMVDR(short *rgsInputBufferL, short *rgsInputBufferR, int iBlockLen, short *rgsOutputBuffer, double dTime, double(*rgdSpatialCorr)[CHANNEL]) {

	fftw_complex fcLeftBefFFT[FFT_PROCESSING_LEN] = { 0, }, fcLeftAftFFT[FFT_PROCESSING_LEN] = { 0, };
	fftw_complex fcRightBefFFT[FFT_PROCESSING_LEN] = { 0, }, fcRightAftFFT[FFT_PROCESSING_LEN] = { 0, };
	fftw_complex fcMergeBefFFT[FFT_PROCESSING_LEN] = { 0, }, fcMergeAftFFT[FFT_PROCESSING_LEN] = { 0, };
	static fftw_complex fcLKeepBuf[KEEP_LEN] = { 0, };
	static fftw_complex fcRKeepBuf[KEEP_LEN] = { 0, };
	fftw_plan fpLeft_p, fpRight_p, fpInverse_p;
	MatrixXcd mxAutoCorr = MatrixXcd(CHANNEL, CHANNEL);
	MatrixXcd mxWeight = MatrixXcd(CHANNEL, 1);
	MatrixXcd mxSteerVec = MatrixXcd(CHANNEL, 1);
	MatrixXcd mxConjuSteerVec = MatrixXcd(CHANNEL, 1);
	double rgdLeftWeightFun[FFT_PROCESSING_LEN][2] = { 0, };
	double rgdRightWeightFun[FFT_PROCESSING_LEN][2] = { 0, };
	static int iNumOfCount = 0;
	iNumOfCount++;
	memcpy(fcLeftBefFFT, fcLKeepBuf, sizeof(fftw_complex) * KEEP_LEN);
	memcpy(fcRightBefFFT, fcRKeepBuf, sizeof(fftw_complex) * KEEP_LEN);
	for (int i = 0; i < iBlockLen; i++) {
		fcLeftBefFFT[i + KEEP_LEN][0] = rgsInputBufferL[i];
		fcRightBefFFT[i + KEEP_LEN][0] = rgsInputBufferR[i];
	}
	// Left channel FFT processing
	fpLeft_p = fftw_plan_dft_1d(FFT_PROCESSING_LEN, fcLeftBefFFT, fcLeftAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpLeft_p);

	// Right channel FFT processing
	fpRight_p = fftw_plan_dft_1d(FFT_PROCESSING_LEN, fcRightBefFFT, fcRightAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpRight_p);

	mxAutoCorr(0, 0) = rgdSpatialCorr[0][0];
	mxAutoCorr(0, 1) = rgdSpatialCorr[0][1];
	mxAutoCorr(1, 0) = rgdSpatialCorr[1][0];
	mxAutoCorr(1, 1) = rgdSpatialCorr[1][1];

	//printf("autocorr %f , %f , %f  %f \n", rgdSpatialCorr[0][0], rgdSpatialCorr[0][1], rgdSpatialCorr[1][0], rgdSpatialCorr[1][1]);

	for (int i = 0; i < FFT_PROCESSING_LEN; i++) {
		mxSteerVec(0, 0).real(1);
		mxSteerVec(0, 0).imag(0);
		mxSteerVec(1, 0).real(cos(2 * PI * i * (SAMPLING_RATE / FFT_PROCESSING_LEN) * dTime));
		mxSteerVec(1, 0).imag(sin(2 * PI * i * (SAMPLING_RATE / FFT_PROCESSING_LEN) * dTime));
		mxConjuSteerVec = mxSteerVec.conjugate();
		//printf("steering vector %f , %f , %f, %f \n ", mxSteerVec(0, 0).real(), mxSteerVec(0, 0).imag(), mxSteerVec(1, 0).real(), mxSteerVec(1, 0).imag());
		//printf("conjugate vector %f , %f, %f, %f \n ", mxConjuSteerVec(0, 0).real(), mxConjuSteerVec(0, 0).imag(), mxConjuSteerVec(1, 0).real(), mxConjuSteerVec(1, 0).imag());

		mxWeight = mxAutoCorr.inverse() * mxSteerVec;
		mxWeight = mxWeight / ((mxConjuSteerVec.transpose() * mxWeight)(0, 0));

		//printf("weight vector %f , %f, %f, %f \n ", mxWeight(0, 0).real(), mxWeight(0, 0).imag(), mxWeight(0, 1).real(), mxWeight(0, 1).imag());

		rgdLeftWeightFun[i][0] = mxWeight(0, 0).real();
		rgdLeftWeightFun[i][1] =  - mxWeight(0, 0).imag(); // complex conjugate
		rgdRightWeightFun[i][0] = mxWeight(1, 0).real();
		rgdRightWeightFun[i][1] =  - mxWeight(1, 0).imag(); // complex conjugate

		fcLeftAftFFT[i][0] = fcLeftAftFFT[i][0] * rgdLeftWeightFun[i][0] - fcLeftAftFFT[i][1] * rgdLeftWeightFun[i][1];
		fcLeftAftFFT[i][1] = fcLeftAftFFT[i][0] * rgdLeftWeightFun[i][1] + fcLeftAftFFT[i][1] * rgdLeftWeightFun[i][0];
		fcRightAftFFT[i][0] = fcRightAftFFT[i][0] * rgdRightWeightFun[i][0] - fcRightAftFFT[i][1] * rgdRightWeightFun[i][1];
		fcRightAftFFT[i][1] = fcRightAftFFT[i][0] * rgdRightWeightFun[i][1] + fcRightAftFFT[i][1] * rgdRightWeightFun[i][0];
		fcMergeAftFFT[i][0] = fcLeftAftFFT[i][0] + fcRightAftFFT[i][0];
		fcMergeAftFFT[i][1] = fcLeftAftFFT[i][1] + fcRightAftFFT[i][1];
	}

	// iFFT processing
	fpInverse_p = fftw_plan_dft_1d(FFT_PROCESSING_LEN, fcMergeAftFFT, fcMergeBefFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fpInverse_p);

	for (int i = 0; i < iBlockLen; i++) {
		rgsOutputBuffer[i] = fcMergeBefFFT[i + KEEP_LEN][0] * 1. / FFT_PROCESSING_LEN;
	}
	memcpy(fcLKeepBuf, &fcLeftBefFFT[KEEP_LEN], sizeof(fftw_complex) * KEEP_LEN);
	memcpy(fcRKeepBuf, &fcRightBefFFT[KEEP_LEN], sizeof(fftw_complex) * KEEP_LEN);
	fftw_destroy_plan(fpRight_p);
	fftw_destroy_plan(fpLeft_p);
	fftw_destroy_plan(fpInverse_p);

	if (iNumOfCount > 1)
		return true;
	else
		return false;
}

bool VoiceActivityDetection(short *rgsInputBuffer, int iFrameCount) {

	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	short rgsProcessingBuffer[FFT_PROCESSING_LEN] = { 0, };
	double dEnergy = 0.0;
	int dZCR = 0;
	memcpy(rgsProcessingBuffer, rgssKeepBuffer, sizeof(rgssKeepBuffer));
	memcpy(rgsProcessingBuffer + KEEP_LEN, rgsInputBuffer, sizeof(short) * iFrameCount);

	for (int i = 0; i < FFT_PROCESSING_LEN; i++) {
		rgsProcessingBuffer[i] *= (0.54 - 0.46 * cos(2 * PI * i / (FFT_PROCESSING_LEN - 1))); // Windowing 
		// 조금 조심 할 부분, short가 되어서, 소수점 짤림.

		//Calc Energy
		dEnergy += pow(rgsProcessingBuffer[i], 2.0);

		//Calc Zero Crossing Rate
		if (i != FFT_PROCESSING_LEN) {
			if (rgsProcessingBuffer[i] * rgsProcessingBuffer[i + 1] < 0)
				dZCR++;
		}
	}
	dEnergy /= FFT_PROCESSING_LEN;
	//dZCR /= (FFT_PROCESSING_SIZE - 1);

	printf(" dEnergy %f , dZCR %d \n", dEnergy, dZCR);
	if (dEnergy > THRESHOLD_OF_ENERGY) {
		return TRUE;
	}
	else {
		return FALSE;
	}

	memcpy(rgssKeepBuffer, &rgsInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));

}

void EstimateSpatialCorrMtx(short *rgsTempBufferL, short *rgsTempBufferR, int iNumOfIteration, double(*rgdSpatialCorr)[CHANNEL], int iFrameCount) {

	fftw_complex fcLeftBefFFT[FFT_PROCESSING_LEN] = { 0, }, fcLeftAftFFT[FFT_PROCESSING_LEN] = { 0, };
	fftw_complex fcRightBefFFT[FFT_PROCESSING_LEN] = { 0, }, fcRightAftFFT[FFT_PROCESSING_LEN] = { 0, };
	fftw_plan fpLeft_p, fpRight_p;

	for (int i = 0; i < iFrameCount; i++) {
		fcLeftBefFFT[i][0] = rgsTempBufferL[i];
		fcRightBefFFT[i][0] = rgsTempBufferR[i];
	}

	// Left channel FFT processing
	fpLeft_p = fftw_plan_dft_1d(FFT_PROCESSING_LEN, fcLeftBefFFT, fcLeftAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpLeft_p);

	// Right channel FFT processing
	fpRight_p = fftw_plan_dft_1d(FFT_PROCESSING_LEN, fcRightBefFFT, fcRightAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpRight_p);

	for (int i = 0; i < FFT_PROCESSING_LEN; i++) {
		rgdSpatialCorr[0][0] += (pow(fcLeftAftFFT[i][0], 2.0) + pow(fcLeftAftFFT[i][1], 2.0)) / FFT_PROCESSING_LEN;
		rgdSpatialCorr[0][1] += (-fcLeftAftFFT[i][0] * fcRightAftFFT[i][1] + fcLeftAftFFT[i][1] * fcRightAftFFT[i][0]) / FFT_PROCESSING_LEN;
		rgdSpatialCorr[1][0] += (-fcRightAftFFT[i][0] * fcLeftAftFFT[i][1] + fcRightAftFFT[i][1] * fcLeftAftFFT[i][0]) / FFT_PROCESSING_LEN;
		rgdSpatialCorr[1][1] += (pow(fcRightAftFFT[i][0], 2.0) + pow(fcRightAftFFT[i][1], 2.0)) / FFT_PROCESSING_LEN;
	}

}