/*
This program is made by jongcheol boo.

FFT implementation.
http://blog.naver.com/PostView.nhn?blogId=horgan&logNo=40011894016&parentCategoryNo=1&viewDate=&currentPage=1&listtype=0

*/
#include<fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define MAX(a,b)    (((a) > (b)) ? (a) : (b))
//#define PI  3.141592
#define PI 3.14159265358
#define BLOCK_LEN 512
#define TRUE 1
#define FALSE 0

typedef struct {
	double real, imag;
} COMPLEX;

void FFTProcess(COMPLEX *cpFftInput, COMPLEX *cpFftOutput, int iFFTLen, bool bDir);
void IFFTProcess(COMPLEX *cpFftOutput, COMPLEX *cpFftInput, int iFFTLen);
void DFTProcess(short *spInputBuffer, COMPLEX *cpFftOutput, int iFFTLen);
void IDFTProcess(COMPLEX *cpFftOutput, COMPLEX *cpFftInput, int iFFTLen);
void Bitrev(COMPLEX *cpFftInput, short* psBit, int iFFTLen, COMPLEX *cpFftBitRevInput);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };
	COMPLEX rgcFftOutput[BLOCK_LEN] = { 0, };
	COMPLEX rgcFftInput[BLOCK_LEN] = { 0, };
	short rgsResultBuffer[BLOCK_LEN] = { 0, };
	//fftw_complex fcInputBefFFT[BLOCK_LEN] = { 0, }, fcInputAftFFT[BLOCK_LEN] = { 0, };
	//fftw_plan fpInput_p;

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

	while (TRUE)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		for (int i = 0; i < BLOCK_LEN; i++) {
			rgcFftInput[i].real = rgsInputBuffer[i];
		}

		//fpInput_p = fftw_plan_dft_1d(BLOCK_LEN, fcInputBefFFT, fcInputAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
		//fftw_execute(fpInput_p);
		//DFTProcess(rgsInputBuffer, rgcFftOutput, BLOCK_LEN);
		FFTProcess(rgcFftInput, rgcFftOutput, BLOCK_LEN, TRUE /* forward */);
		//IDFTProcess(rgcFftOutput, rgcFftInput, BLOCK_LEN);
		FFTProcess(rgcFftOutput, rgcFftInput, BLOCK_LEN, FALSE /* backward */); //inverse fft

		for (int i = 0; i < BLOCK_LEN; i++) {
			rgsResultBuffer[i] = rgcFftInput[i].real / (double)BLOCK_LEN;
			//printf("--------- rgcFftInput[i].real %f \n", rgcFftInput[i].real / (double)BLOCK_LEN);
		}
		fwrite(rgsResultBuffer, sizeof(short), BLOCK_LEN, fpWrite);
		memset(rgcFftInput, 0, sizeof(COMPLEX) * BLOCK_LEN);
		memset(rgcFftOutput, 0, sizeof(COMPLEX) * BLOCK_LEN);
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;
}

void FFTProcess(COMPLEX *cpFftInput, COMPLEX *cpFftOutput, int iFFTLen, bool bDir) {

	short rgsRevBit[BLOCK_LEN] = { 0, };
	int iNpoint = iFFTLen / 2, iN2 = 0, iN1 = 0, iN3 = 0, iIndex = 0;
	int iCountAdd = 0, iCountMulply = 0;
	COMPLEX cTemp = { 0, };
	// bit reverse algorithm refer to http://www.katjaas.nl/bitreversal/bitreversal.html
	Bitrev(cpFftInput, rgsRevBit, iFFTLen, cpFftOutput);

	// FFTDecimation - in - Time, In - order output, Radix - 2 FFT
	while (TRUE) {
		//iNpoint; //8,  4 , 2 , 1 --> max count for add
		//iNpoint/2; // 4 , 2 , 1 ,0 --> max count for multiply
		//printf(" iNPoint %d \n", iNpoint);
		iN2 = iFFTLen / iNpoint; // 2, 4, 8, 16 --> add offset
		iN1 = iN2 / 2; // 1, 2, 4 ,8 --> half multiply offset
		iN3 = iN2 * 2; // 4, 8 ,16, 32 --> multiply offset
		for (int i = 0; i < iNpoint; i++) {
			for (int m = 0; m < iN1; m++) {
				iIndex = iN2 * i + m;
				cTemp.real = cpFftOutput[iIndex].real;
				cTemp.imag = cpFftOutput[iIndex].imag;
				cpFftOutput[iIndex].real = cpFftOutput[iN2 * i + m].real + cpFftOutput[iIndex + iN1].real;
				cpFftOutput[iIndex].imag = cpFftOutput[iN2 * i + m].imag + cpFftOutput[iIndex + iN1].imag;
				cpFftOutput[iIndex + iN1].real = cTemp.real - cpFftOutput[iIndex + iN1].real;
				cpFftOutput[iIndex + iN1].imag = cTemp.imag - cpFftOutput[iIndex + iN1].imag;
				iCountAdd++;
			}
		}

		if (iNpoint == 1) {// end calculation.
			break;
		}

		for (int k = 0; k < (iNpoint/2); k++) { // 4, 2, 1. end
			for (int n = 0; n < iN2; n++) {
				iIndex = k * iN3 + iN2 + n;
				iCountMulply++;
				if (bDir == TRUE) {
					cTemp.real = cpFftOutput[iIndex].real;
					cTemp.imag = cpFftOutput[iIndex].imag;
					cpFftOutput[iIndex].real = cos(-2 * PI * n / (double)iN3) * cTemp.real - sin(-2 * PI * n / (double)iN3) * cTemp.imag;
					cpFftOutput[iIndex].imag = cos(-2 * PI * n / (double)iN3) * cTemp.imag + sin(-2 * PI * n / (double)iN3) * cTemp.real;
				}
				else {
					cTemp.real = cpFftOutput[iIndex].real;
					cTemp.imag = cpFftOutput[iIndex].imag;
					cpFftOutput[iIndex].real = cos(2 * PI * n / (double)iN3) * cTemp.real - sin(2 * PI * n / (double)iN3) * cTemp.imag;
					cpFftOutput[iIndex].imag = cos(2 * PI * n / (double)iN3) * cTemp.imag + sin(2 * PI * n / (double)iN3) * cTemp.real;
				}
			}
		}
		iNpoint /= 2;
	}
	printf("%d-point FFT Calculation add %d multiply %d \n ", iFFTLen, iCountAdd, iCountMulply);
}

void IFFTProcess(COMPLEX *cpFftOutput, COMPLEX *cpFftInput, int iFFTLen) {
	for (int k = 0; k < iFFTLen; k++) {
		for (int i = 0; i < iFFTLen; i++) {
			cpFftInput[k].real += (cpFftOutput[i].real * cos(2 * PI * i * k / (double)iFFTLen) - cpFftOutput[i].imag * sin(2 * PI * i * k / (double)iFFTLen)) * 1 / (double)iFFTLen;
			cpFftInput[k].imag += (cpFftOutput[i].real * sin(2 * PI * i * k / (double)iFFTLen) + cpFftOutput[i].imag * cos(2 * PI * i * k / (double)iFFTLen)) * 1 / (double)iFFTLen;
		}
		//printf(" real++++++++ output %f \n", cpFftInput[k].real);
		//printf(" image-------- output %f \n", cpFftInput[k].imag);
	}
}

void DFTProcess(short *spInputBuffer, COMPLEX *cpFftOutput, int iFFTLen) {
	int iCountAdd = 0, iCountMulply = 0;
	for (int k = 0; k < iFFTLen; k++) {
		for (int i = 0; i < iFFTLen; i++) {
			iCountMulply++;
			iCountAdd++;
			cpFftOutput[k].real += spInputBuffer[i] * cos(2 * PI * i * k / (double)iFFTLen);
			cpFftOutput[k].imag += spInputBuffer[i] * -sin(2 * PI * i * k / (double)iFFTLen);
		}
	}
	printf("%d-point DFT Calculation add %d multiply %d \n ", iFFTLen, iCountAdd, iCountMulply);
}

void IDFTProcess(COMPLEX *cpFftOutput, COMPLEX *cpFftInput, int iFFTLen) {
	for (int k = 0; k < iFFTLen; k++) {
		for (int i = 0; i < iFFTLen; i++) {
			cpFftInput[k].real += (cpFftOutput[i].real * cos(2 * PI * i * k / (double)iFFTLen) - cpFftOutput[i].imag * sin(2 * PI * i * k / (double)iFFTLen));// *1 / (double)iFFTLen;
			cpFftInput[k].imag += (cpFftOutput[i].real * sin(2 * PI * i * k / (double)iFFTLen) + cpFftOutput[i].imag * cos(2 * PI * i * k / (double)iFFTLen));// *1 / (double)iFFTLen;
		}
		//printf(" real++++++++ output %f \n", cpFftInput[k].real);
		//printf(" image-------- output %f \n", cpFftInput[k].imag);
	}
}

void Bitrev(COMPLEX *cpFftInput, short* psBit, int iFFTLen, COMPLEX *cpFftBitRevInput) {

	int iBits = (int)log2((double)BLOCK_LEN);
	short iTemp = 0;

	for (int k = 0; k < iFFTLen; k++) {
		iTemp = k;
		psBit[k] = iTemp;

		for (int i = 1; i < iBits; i++)
		{
			iTemp >>= 1;
			psBit[k] <<= 1;
			psBit[k] |= iTemp & 1;   // give LSB of n to nrev
		}

		psBit[k] &= iFFTLen - 1;         // clear all bits more significant than N-1

		cpFftBitRevInput[k].real = cpFftInput[psBit[k]].real;
		cpFftBitRevInput[k].imag = cpFftInput[psBit[k]].imag;
	}
}