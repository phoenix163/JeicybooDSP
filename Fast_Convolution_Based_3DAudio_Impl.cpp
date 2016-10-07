/*
3D Audio Implementation using room impulse response.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

FFT 변환을 통한 Frequency Domain 해석
https://ko.wikipedia.org/wiki/%EA%B3%A0%EC%86%8D_%ED%91%B8%EB%A6%AC%EC%97%90_%EB%B3%80%ED%99%98

Uring Matlab function [h]=rir(fs, mic, n, r, rm, src);
%RIR   Room Impulse Response.
%   [h] = RIR(FS, MIC, N, R, RM, SRC) performs a room impulse
%         response calculation by means of the mirror image method.
%
%      FS  = sample rate.
%      MIC = row vector giving the x,y,z coordinates of
%            the microphone.
%      N   = The program will account for (2*N+1)^3 virtual sources
%      R   = reflection coefficient for the walls, in general -1<R<1.
%      RM  = row vector giving the dimensions of the room.
%      SRC = row vector giving the x,y,z coordinates of
%            the sound source.
%
%   EXAMPLE:
%
%      >>fs=44100;
%      >>mic=[19 18 1.6];
%      >>n=12;
%      >>r=0.3;
%      >>rm=[20 19 21];
%      >>src=[5 2 1];
%      >>h=rir(fs, mic, n, r, rm, src);

-------------------------------------------------------------
*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
#include"FilterCoefficient.h"
#include <vector>
#include <queue>

#define BLOCK_SIZE 1024 // 1000*(512/16000)= 32ms.
#define FFT_PROCESSING_SIZE 8192// 2의 n승값이 되어야 한다.
#define SAVE_LENGTH 7168
using namespace std;
bool AnalySisFreqDomain(short *psInputBuffer, short *psOutputBuffer, int iFrameCount, fftw_complex *fcFilterBefFFT);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_SIZE] = { 0, };
	short rgsOutputBuffer[BLOCK_SIZE] = { 0, };
	fftw_complex fcFilterBefFFT[FFT_PROCESSING_SIZE] = { 0, };

	if (argc != 3) {
		printf("path를 2개 입력해야 합니다.\n"); // input path, output path
		return;
	}
	else {
		for (int i = 1; i < 3; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpRead = fopen(argv[1], "rb")) == NULL)
		printf("Read File Open Error\n");

	if ((fpWrite = fopen(argv[2], "wb")) == NULL)
		printf("Write File Open Error\n");

	// Read한 Wav 파일 맨 앞의 Header 44Byte 만큼 Write할 Wav파일에 write.
	fread(rgcHeader, 1, 44, fpRead);
	//fwrite(rgcHeader, 1, 44, fpWrite);
	//memcpy(fcFilterBefFFT[i][0], rgdFirLPF_coefficients[i], sizeof(double) * FILTER_LENGTH);
	for (int i = 0; i < FILTER_LENGTH; i++) {
		fcFilterBefFFT[i][0] = rgdFirLPF_coefficients[i];
	}
	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_SIZE, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		if (AnalySisFreqDomain(rgsInputBuffer, rgsOutputBuffer, BLOCK_SIZE, fcFilterBefFFT))
			fwrite(rgsOutputBuffer, sizeof(short), BLOCK_SIZE, fpWrite);
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();

	return;
}

bool AnalySisFreqDomain(short *psInputBuffer, short *psOutputBuffer, int iFrameCount, fftw_complex *fcFilterBefFFT) {

	fftw_complex fcInputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcInputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcFilterAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcOutputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcOutputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_plan fpInput_p, fpFilter_p, fpOutput_p;
	double rgdTempFilterAmp[FFT_PROCESSING_SIZE] = { 0, };
	double rgdTempFilteredAmp[FFT_PROCESSING_SIZE] = { 0, };
	double rgdArcTan[FFT_PROCESSING_SIZE] = { 0, };
	static short rgssKeepBuffer[FILTER_LENGTH - 1] = { 0, };
	static int siNumOfCount = 0;
	static queue<short *> Queue;
	queue<short *> TempQueue;
	short *dTempBuffer = NULL, *dTempAddress = NULL;
	int dTempCal = 0;
	siNumOfCount++;

	if (siNumOfCount < (MAX_QUEUE_SIZE + 1)) {
		dTempBuffer = (short*)malloc(sizeof(short) * iFrameCount);
		Queue.push(dTempBuffer);
		return false;
	}

	for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
		dTempAddress = Queue.front();
		dTempCal = i * iFrameCount;
		for (int j = 0; j < iFrameCount; j++)
			fcInputBefFFT[dTempCal + j][0] = dTempAddress[j];

		TempQueue.push(Queue.front());
		Queue.pop();
	}
	dTempCal = MAX_QUEUE_SIZE * iFrameCount;
	for (int j = 0; j < iFrameCount; j++) {
		fcInputBefFFT[dTempCal + j][0] = psInputBuffer[j];
	}

	fpInput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcInputBefFFT, fcInputAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fpFilter_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcFilterBefFFT, fcFilterAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fpOutput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcOutputAftFFT, fcOutputBefFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fpInput_p);
	fftw_execute(fpFilter_p);

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgdArcTan[i] = atan2(fcInputAftFFT[i][1], fcInputAftFFT[i][0]);
	}

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		fcOutputAftFFT[i][0] = fcInputAftFFT[i][0] * fcFilterAftFFT[i][0] - fcInputAftFFT[i][1] * fcFilterAftFFT[i][1];
		fcOutputAftFFT[i][1] = fcInputAftFFT[i][0] * fcFilterAftFFT[i][1] + fcInputAftFFT[i][1] * fcFilterAftFFT[i][0];
	}

	fftw_execute(fpOutput_p);

	for (int i = 0; i < iFrameCount; i++) {
		psOutputBuffer[i] = fcOutputBefFFT[i + FILTER_LENGTH - 1][0] * 1. / FFT_PROCESSING_SIZE;
	}

	for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
		if (i == 0) {
			free(TempQueue.front());
			TempQueue.pop();
			continue;
		}
		Queue.push(TempQueue.front());
		TempQueue.pop();
	}
	dTempBuffer = (short*)malloc(sizeof(short) * iFrameCount);
	memcpy(dTempBuffer, psInputBuffer, sizeof(short) * iFrameCount);
	Queue.push(dTempBuffer);

	fftw_destroy_plan(fpInput_p);
	fftw_destroy_plan(fpFilter_p);
	fftw_destroy_plan(fpOutput_p);
	return true;
}