/*
Bass Treble Booster
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 48kHz SamplingRate, mono channel, 16bit per sample.
https://www.dsprelated.com/showcode/170.php
https://www.dsprelated.com/showcode/169.php

아래와 같은 코드로, python에서 filter계수에 대한 frequency response를 구할 수 있습니다.
2차 iir filter라서 그런지, ideal 한 모양이 안나오네요.. 그만큼 필터가 정교하지 않습니다.

from pylab import *
import scipy.signal as signal
import matplotlib.pyplot as plt
b = [ 1.000000 ,  -1.307183,  0.848636]
a = [ 1.000000, -1.307183, 0.900263]
w, h = signal.freqz(b , a)
plot(w/(2*pi), 20*log10(abs(h)))
show()
#plot(w/(2*pi), abs(h))
#show()

*/
#include <stdlib.h>
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
#define PI 3.141592
#define SAMPLING_RATE 48000.0
#define SHELVING_KEEP_LEN 2
#define PEAKING_KEEP_LEN 2
//#define DEBUG
#define DEBUG_NUM 6

#define TOTAL_BANDS 7
//#define NUM_OF_PEAK_BANDS 5
#define PEAK_FILTER_LENGTH 3
#define SHELVING_FILTER_LENGTH 3
#define BLOCK_LEN 512 // 1000*(512/16000)= 32ms.
#define KEEP_LENGTH 12 // 2(keep length)*5(number of bandpass filter) + 1(keep length)*(number of low/high pass filter)
#define Q 4.318 // 1/3 Octave
//double rgdCenterFreqs[TOTAL_BANDS] = { 31.0, 125.0, 250.0, 500.0, 2000.0, 6000.0, 16000.0 };
double rgdCenterFreqs[TOTAL_BANDS] = { 44.0, 125.0, 250.0, 500.0, 2000.0, 6000.0, 11313.0 };
//double rgdUpperCutOffFreqs[TOTAL_BANDS] = { 44, 177, 354, 707, 2829, 8486, 22629 };
//double rgdLowerCutOffFreqs[TOTAL_BANDS] = { 22, 88, 177, 354, 1414, 4242, 11313 };
//double rgdGainOfBand[TOTAL_BANDS] = { 1, 1, 1, 1, 1, 1, 1 };
#define GAIN_DB_BAND1 12.0
#define GAIN_DB_BAND2 12.0
#define GAIN_DB_BAND3 0.0
#define GAIN_DB_BAND4 0.0
#define GAIN_DB_BAND5 3.0
#define GAIN_DB_BAND6 0.0
#define GAIN_DB_BAND7 -12.0

#define ROOT2 (1.0 / Q) //sqrt(2.0)//

#define V_BAND1 (pow(10, GAIN_DB_BAND1/20.0)) // %Invert gain if a cut
#define V_BAND2 (pow(10, GAIN_DB_BAND2/20.0))
#define V_BAND3 (pow(10, GAIN_DB_BAND3/20.0))
#define V_BAND4 (pow(10, GAIN_DB_BAND4/20.0))
#define V_BAND5 (pow(10, GAIN_DB_BAND5/20.0))
#define V_BAND6 (pow(10, GAIN_DB_BAND6/20.0))
#define V_BAND7 (pow(10, GAIN_DB_BAND7/20.0))

//#define H_0 (V_0 - 1.0)
#define K_BAND1 tan(PI*rgdCenterFreqs[0]/SAMPLING_RATE)
#define K_BAND2 tan(PI*rgdCenterFreqs[1]/SAMPLING_RATE)
#define K_BAND3 tan(PI*rgdCenterFreqs[2]/SAMPLING_RATE)
#define K_BAND4 tan(PI*rgdCenterFreqs[3]/SAMPLING_RATE)
#define K_BAND5 tan(PI*rgdCenterFreqs[4]/SAMPLING_RATE)
#define K_BAND6 tan(PI*rgdCenterFreqs[5]/SAMPLING_RATE)
#define K_BAND7 tan(PI*rgdCenterFreqs[6]/SAMPLING_RATE)

double K_band[TOTAL_BANDS] = { K_BAND1, K_BAND2, K_BAND3, K_BAND4, K_BAND5, K_BAND6, K_BAND7 };
double V_band[TOTAL_BANDS] = { V_BAND1, V_BAND2, V_BAND3, V_BAND4, V_BAND5, V_BAND6, V_BAND7 };
double G_band[TOTAL_BANDS] = { GAIN_DB_BAND1, GAIN_DB_BAND2, GAIN_DB_BAND3, GAIN_DB_BAND4, GAIN_DB_BAND5, GAIN_DB_BAND6, GAIN_DB_BAND7 };

// Coefficients
// b0 b1 b2
//  1 a1 a2
// for iir filtering

double rgdBandCoeff[TOTAL_BANDS][2][PEAK_FILTER_LENGTH] = { 0, };

void CalcCoefficient();
void ApplyIirGEQ(short *psInputBuffer, short *psOutputBuffer, int iFrameCount);
void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };
	short rgsOutputBuffer[BLOCK_LEN] = { 0, };

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
	CalcCoefficient();
	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		ApplyIirGEQ(rgsInputBuffer, rgsOutputBuffer, BLOCK_LEN);
		fwrite(rgsOutputBuffer, sizeof(short), BLOCK_LEN, fpWrite);
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;
}


void CalcCoefficient() {
	double dTemp = 0.0;

	for (int k = 0; k < TOTAL_BANDS; k++) {
		if (V_band[k] < 1)
			V_band[k] = 1.0 / V_band[k];
	}

	if (GAIN_DB_BAND1 > 0) {
		// Bass Booster
		//b0 = (1 + sqrt(V0)*root2*K + V0*K^2) / (1 + root2*K + K^2);
		//b1 =             (2 * (V0*K^2 - 1) ) / (1 + root2*K + K^2);
		//b2 = (1 - sqrt(V0)*root2*K + V0*K^2) / (1 + root2*K + K^2);
		//a1 =                (2 * (K^2 - 1) ) / (1 + root2*K + K^2);
		//a2 =             (1 - root2*K + K^2) / (1 + root2*K + K^2);
		dTemp = (1 + ROOT2*K_band[0] + pow(K_band[0], 2.0));

		rgdBandCoeff[0][0][0] = (1 + sqrt(V_band[0])*ROOT2*K_band[0] + V_band[0] * pow(K_band[0], 2.0)) / dTemp;
		rgdBandCoeff[0][0][1] = (2 * (V_band[0] * pow(K_band[0], 2.0) - 1)) / dTemp;
		rgdBandCoeff[0][0][2] = (1 - sqrt(V_band[0])*ROOT2*K_band[0] + V_band[0] * pow(K_band[0], 2.0)) / dTemp;
		rgdBandCoeff[0][1][0] = 0.0;
		rgdBandCoeff[0][1][1] = (2 * (pow(K_band[0], 2.0) - 1)) / dTemp;
		rgdBandCoeff[0][1][2] = (1 - ROOT2*K_band[0] + pow(K_band[0], 2.0)) / dTemp;
	}
	else {
		// Bass Cut
		//b0 =             (1 + root2*K + K^2) / (1 + root2*sqrt(V0)*K + V0*K^2);
		//b1 =                (2 * (K^2 - 1) ) / (1 + root2*sqrt(V0)*K + V0*K^2);
		//b2 =             (1 - root2*K + K^2) / (1 + root2*sqrt(V0)*K + V0*K^2);
		//a1 =             (2 * (V0*K^2 - 1) ) / (1 + root2*sqrt(V0)*K + V0*K^2);
		//a2 = (1 - root2*sqrt(V0)*K + V0*K^2) / (1 + root2*sqrt(V0)*K + V0*K^2);
		dTemp = (1 + ROOT2*sqrt(V_band[0])*K_band[0] + V_band[0] * pow(K_band[0], 2.0));

		rgdBandCoeff[0][0][0] = (1 + ROOT2*K_band[0] + pow(K_band[0], 2.0)) / dTemp;
		rgdBandCoeff[0][0][1] = (2 * (pow(K_band[0], 2.0) - 1)) / dTemp;
		rgdBandCoeff[0][0][2] = (1 - ROOT2*K_band[0] + pow(K_band[0], 2.0)) / dTemp;
		rgdBandCoeff[0][1][0] = 0.0;
		rgdBandCoeff[0][1][1] = (2 * (K_band[0] * pow(K_band[0], 2.0) - 1)) / dTemp;
		rgdBandCoeff[0][1][2] = (1 - ROOT2*sqrt(K_band[0])*K_band[0] + K_band[0] * pow(K_band[0], 2.0)) / dTemp;
	}

	if (GAIN_DB_BAND7 > 0) {
		// Treble Booster
		//b0 = (V0 + root2*sqrt(V0)*K + K^2) / (1 + root2*K + K^2);
		//b1 =             (2 * (K^2 - V0) ) / (1 + root2*K + K^2);
		//b2 = (V0 - root2*sqrt(V0)*K + K^2) / (1 + root2*K + K^2);
		//a1 =              (2 * (K^2 - 1) ) / (1 + root2*K + K^2);
		//a2 =           (1 - root2*K + K^2) / (1 + root2*K + K^2);
		dTemp = (1 + ROOT2*K_band[6] + pow(K_band[6], 2.0));

		rgdBandCoeff[6][0][0] = (V_band[6] + ROOT2*sqrt(V_band[6])*K_band[6] + pow(K_band[6], 2.0)) / dTemp;
		rgdBandCoeff[6][0][1] = (2 * (pow(K_band[6], 2.0) - V_band[6])) / dTemp;
		rgdBandCoeff[6][0][2] = (V_band[6] - ROOT2*sqrt(V_band[6])*K_band[6] + pow(K_band[6], 2.0)) / dTemp;
		rgdBandCoeff[6][1][0] = 0.0;
		rgdBandCoeff[6][1][1] = (2 * (pow(K_band[6], 2.0) - 1)) / dTemp;
		rgdBandCoeff[6][1][2] = (1 - ROOT2*K_band[6] + pow(K_band[6], 2.0)) / dTemp;
	}
	else {
		// Treble Cut
		//b0 =               (1 + root2*K + K^2) / (V0 + root2*sqrt(V0)*K + K^2);
		//b1 =                  (2 * (K^2 - 1) ) / (V0 + root2*sqrt(V0)*K + K^2);
		//b2 =               (1 - root2*K + K^2) / (V0 + root2*sqrt(V0)*K + K^2);
		//a1 =             (2 * ((K^2)/V0 - 1) ) / (1 + root2/sqrt(V0)*K + (K^2)/V0);
		//a2 = (1 - root2/sqrt(V0)*K + (K^2)/V0) / (1 + root2/sqrt(V0)*K + (K^2)/V0);

		dTemp = (V_band[6] + ROOT2*sqrt(V_band[6])*K_band[6] + pow(K_band[6], 2.0));
		rgdBandCoeff[6][0][0] = (1 + ROOT2*K_band[6] + pow(K_band[6], 2.0)) / dTemp;
		rgdBandCoeff[6][0][1] = (2 * (pow(K_band[6], 2.0) - 1)) / dTemp;
		rgdBandCoeff[6][0][2] = (1 - ROOT2*K_band[6] + pow(K_band[6], 2.0)) / dTemp;

		dTemp = (1 + ROOT2 / sqrt(V_band[6])*K_band[6] + (pow(K_band[6], 2.0)) / V_band[6]);
		rgdBandCoeff[6][1][0] = 0.0;
		rgdBandCoeff[6][1][1] = (2 * ((pow(K_band[6], 2.0)) / V_band[6] - 1)) / dTemp;
		rgdBandCoeff[6][1][2] = (1 - ROOT2 / sqrt(V_band[6])*K_band[6] + (pow(K_band[6], 2.0)) / V_band[6]) / dTemp;
	}

	for (int k = 0; k < TOTAL_BANDS; k++) {

		if (k == 0 || k == 6) {
			continue;
		}
		if (G_band[k] > 0) {
			// Boost Peak
			//b0 = (1 + ((V0/Q)*K) + K^2) / (1 + ((1/Q)*K) + K^2);
			//b1 =        (2 * (K^2 - 1)) / (1 + ((1/Q)*K) + K^2);
			//b2 = (1 - ((V0/Q)*K) + K^2) / (1 + ((1/Q)*K) + K^2);
			//a1 = b1;
			//a2 =  (1 - ((1/Q)*K) + K^2) / (1 + ((1/Q)*K) + K^2);
			dTemp = (1 + ((1 / Q)*K_band[k]) + pow(K_band[k], 2.0));
			rgdBandCoeff[k][0][0] = (1 + ((V_band[k] / Q)*K_band[k]) + pow(K_band[k], 2.0)) / dTemp;
			rgdBandCoeff[k][0][1] = (2 * (pow(K_band[k], 2.0) - 1)) / dTemp;
			rgdBandCoeff[k][0][2] = (1 - ((V_band[k] / Q)*K_band[k]) + pow(K_band[k], 2.0)) / dTemp;

			rgdBandCoeff[k][1][0] = 0.0;
			rgdBandCoeff[k][1][1] = rgdBandCoeff[k][0][1];
			rgdBandCoeff[k][1][2] = (1 - ((1 / Q)*K_band[k - 1]) + pow(K_band[k], 2.0)) / dTemp;
		}
		else {
			// Cut Peak
			//b0 = (1 + ((1/Q)*K) + K^2) / (1 + ((V0/Q)*K) + K^2);
			//b1 =       (2 * (K^2 - 1)) / (1 + ((V0/Q)*K) + K^2);
			//b2 = (1 - ((1/Q)*K) + K^2) / (1 + ((V0/Q)*K) + K^2);
			//a1 = b1;
			//a2 = (1 - ((V0/Q)*K) + K^2) / (1 + ((V0/Q)*K) + K^2);
			dTemp = (1 + ((V_band[k] / Q)*K_band[k]) + pow(K_band[k], 2.0));
			rgdBandCoeff[k][0][0] = (1 + ((1.0 / Q)*K_band[k]) + pow(K_band[k], 2.0)) / dTemp;
			rgdBandCoeff[k][0][1] = (2 * (pow(K_band[k], 2.0) - 1)) / dTemp;
			rgdBandCoeff[k][0][2] = (1 - ((1.0 / Q)*K_band[k]) + pow(K_band[k], 2.0)) / dTemp;

			rgdBandCoeff[k][1][0] = 0.0;
			rgdBandCoeff[k][1][1] = rgdBandCoeff[k][0][1];
			rgdBandCoeff[k][1][2] = (1 - ((V_band[k] / Q)*K_band[k - 1]) + pow(K_band[k], 2.0)) / dTemp;
		}
	}

	for (int k = 0; k < TOTAL_BANDS; k++) {

		printf("%d -th  b0 %f , b1 %f, b2 %f,  a0 %f, a1 %f, a2 %f \n", 
		k, rgdBandCoeff[k][0][0], rgdBandCoeff[k][0][1], rgdBandCoeff[k][0][2], 1.0 , rgdBandCoeff[k][1][1], rgdBandCoeff[k][1][2]);
	}

}

void ApplyIirGEQ(short *psInputBuffer, short *psOutputBuffer, int iFrameCount) {

	static short rgssKeepInputBufferBand[TOTAL_BANDS][SHELVING_KEEP_LEN] = { 0, };
	static short rgssKeepOutputBufferBand[TOTAL_BANDS][SHELVING_KEEP_LEN] = { 0, };

	short **pSInBufBand = (short **)calloc(TOTAL_BANDS, sizeof(short*));
	short **pSOutBufBand = (short **)calloc(TOTAL_BANDS, sizeof(short*));
	short rgsKeepLen[TOTAL_BANDS] = { 2, 2, 2, 2, 2, 2, 2 };

	double dTemp = 0.0;
	pSInBufBand[0] = (short*)calloc((iFrameCount + rgsKeepLen[0]), sizeof(short));
	memcpy(pSInBufBand[0], rgssKeepInputBufferBand[0], sizeof(short) * rgsKeepLen[0]);
	memcpy(pSInBufBand[0] + rgsKeepLen[0], psInputBuffer, iFrameCount * sizeof(short));

	pSOutBufBand[0] = (short*)calloc((iFrameCount + rgsKeepLen[0]), sizeof(short));
	memcpy(pSOutBufBand[0], rgssKeepOutputBufferBand[0], sizeof(short) * rgsKeepLen[0]);

	for (int k = 0; k < TOTAL_BANDS; k++) {

		// 1 Band Processing convolution.
		for (int j = 0; j < iFrameCount; j++) {
			for (int i = 0; i < rgsKeepLen[k] + 1; i++) {
				dTemp += rgdBandCoeff[k][0][rgsKeepLen[k] - i] * pSInBufBand[k][i + j];
				dTemp -= rgdBandCoeff[k][1][rgsKeepLen[k] - i] * pSOutBufBand[k][i + j];
			}
			pSOutBufBand[k][j + rgsKeepLen[k]] = dTemp;
			dTemp = 0;
		}

		memcpy(rgssKeepInputBufferBand[k], pSInBufBand[k] + iFrameCount, sizeof(short) * rgsKeepLen[k]);
		memcpy(rgssKeepOutputBufferBand[k], pSOutBufBand[k] + iFrameCount, sizeof(short) * rgsKeepLen[k]);
#ifdef DEBUG
		if (k == (TOTAL_BANDS - 1)) {
#else
		if (k == DEBUG_NUM) {
#endif
			break;
		}

		pSInBufBand[k + 1] = (short*)calloc((iFrameCount + rgsKeepLen[k + 1]), sizeof(short));
		memcpy(pSInBufBand[k + 1], rgssKeepInputBufferBand[k + 1], sizeof(short) * rgsKeepLen[k + 1]);
		memcpy(pSInBufBand[k + 1] + rgsKeepLen[k + 1], pSOutBufBand[k] + rgsKeepLen[k], iFrameCount * sizeof(short));
		pSOutBufBand[k + 1] = (short*)calloc((iFrameCount + rgsKeepLen[k + 1]), sizeof(short));
		memcpy(pSOutBufBand[k + 1], rgssKeepOutputBufferBand[k + 1], sizeof(short) * rgsKeepLen[k + 1]);

		free(pSInBufBand[k]);
		free(pSOutBufBand[k]);
		pSInBufBand[k] = NULL;
		pSOutBufBand[k] = NULL;
	}
	//output Buffer로 결과 copy
#ifdef DEBUG
	memcpy(psOutputBuffer, pSOutBufBand[TOTAL_BANDS - 1] + rgsKeepLen[TOTAL_BANDS -1], sizeof(short) * iFrameCount);
	free(pSInBufBand[TOTAL_BANDS - 1]);
	pSInBufBand[TOTAL_BANDS - 1] = NULL;
	free(pSOutBufBand[TOTAL_BANDS - 1]);
	pSOutBufBand[TOTAL_BANDS - 1] = NULL;
	free(pSInBufBand);
	free(pSOutBufBand);
	pSInBufBand = NULL;
	pSOutBufBand = NULL;
#else
	memcpy(psOutputBuffer, pSOutBufBand[DEBUG_NUM] + rgsKeepLen[DEBUG_NUM], sizeof(short) * iFrameCount);
	free(pSInBufBand[DEBUG_NUM]);
	pSInBufBand[DEBUG_NUM] = NULL;
	free(pSOutBufBand[DEBUG_NUM]);
	pSOutBufBand[DEBUG_NUM] = NULL;
	free(pSInBufBand);
	free(pSOutBufBand);
	pSInBufBand = NULL;
	pSOutBufBand = NULL;
#endif
	return;
}