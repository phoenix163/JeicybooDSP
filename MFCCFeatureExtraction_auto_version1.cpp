/*
Mel Frequency Cepstral Coefficient
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

refer to http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>

#include <string>
using namespace std;

//#define LOPASS_FREQ 300
//#define HIPASS_FREQ 3400
#define MFCC_LEN 12
#define FALSE 0
#define TRUE 1
#define PI 3.141592
#define BLOCK_LEN 1024 // 1000*(1024/16000)= 64ms. 
#define WINDOW_LEN 1024// But it spends 23 ms so that 1 MFCC Feature is extracted
#define KEEP_LEN 512
#define NUM_OF_FEATURE 2 // 1024/256 * 2(overlapping)
#define CHANNEL 38
#define LIFTER_LEN 22
#define HALF_SAMPLING_RATE 22050.0
double rgdFilterBank[KEEP_LEN] = { 0, }; // Filterbank's weighting value.
int rgdFiBins[KEEP_LEN] = { 0, }; // it contains channel information. (0~38)
double rgdMelFreqs[CHANNEL + 1] = { 0, }; // frequencys of linearly divided mel scale.

void MelFilterBankInit();
bool MFCCFeatureExtraction(short *rgsInputBuffer, double(*dMFCCFeature)[MFCC_LEN]);
void MelFilterBank(double *dAbs, double *dMelFiltered);
void DCT(double *dMelFiltered, double *dMFCCFeature);
void Liftering(double *dMFCCFeature);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpReadFile, *fpRead;
	FILE *fpWrite;
	//string stFilePath = "D:\\delete\\구자료\\RingtoneTest\\MFCC_File.txt";
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };
	double dMFCCFeature[NUM_OF_FEATURE][MFCC_LEN] = { 0, };
	int iNumOfIteration = 0, iCount = 0;
	char rgcTempRead[255], rgcTempWrite[255];

	if (argc != 2) {
		printf("path를 1개 입력해야 합니다.\n"); // input path, output path
		return;
	}
	else {
		for (int i = 1; i < 2; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpReadFile = fopen(argv[1], "rb")) == NULL)
		printf("Read File Open Error\n");

	while (!feof(fpReadFile))
	{
		memset(rgcTempRead, 0, sizeof(rgcTempRead));
		memset(rgcTempWrite, 0, sizeof(rgcTempWrite));
		fscanf(fpReadFile, "%s %s", rgcTempRead, rgcTempWrite);
		printf(" rgcTempRead %s \n", rgcTempRead);
		printf(" rgcTempWrite %s \n", rgcTempWrite);

		if ((fpRead = fopen(rgcTempRead, "rb")) == NULL)
			printf("Read File Open Error\n");

		if ((fpWrite = fopen(rgcTempWrite, "wb")) == NULL)
			printf("Write File Open Error\n");

		// Read한 Wav 파일 맨 앞의 Header 44Byte 만큼 Write할 Wav파일에 write.
		fread(rgcHeader, 1, 44, fpRead);
		//fwrite(rgcHeader, 1, 44, fpWrite);
		MelFilterBankInit();
		while (true)
		{
			if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
				printf("Break! The buffer is insufficient.\n");
				break;
			}

			if (MFCCFeatureExtraction(rgsInputBuffer, dMFCCFeature)) { // TRUE가 return되었을 경우에만 file에 쓴다.
				for (int i = 0; i < NUM_OF_FEATURE; i++) {
					if (iNumOfIteration == 0 && i == 0) {
						printf("skip init feature \n");
					}
					else {
						fwrite(dMFCCFeature[i], sizeof(double), MFCC_LEN, fpWrite);
					}
				}
			}
			iNumOfIteration++;
		}

	}
	printf("Processing End\n");
	fclose(fpReadFile);
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;

}

// Calculate rgdFilterBank, rgdFiBins, rgdMelFreqs.

void MelFilterBankInit() {

	double dUnitFreq = 0;
	int dTempBin = 0;
	double dTempFreq = 0;

	dUnitFreq = 1127.0 * log(1 + (HALF_SAMPLING_RATE / 700.0)) / (CHANNEL + 1); // dividing channel+1 for calc unit scale of melfreq.

	for (int i = 1; i <= CHANNEL + 1; i++) {
		rgdMelFreqs[i - 1] = dUnitFreq * i;
		rgdMelFreqs[i - 1] = 700 * (exp(rgdMelFreqs[i - 1] / 1127.0) - 1.0);
	}

	for (int i = 0, k = 0; i < KEEP_LEN; i++) {
		if ((i / (double)(KEEP_LEN - 1)) * HALF_SAMPLING_RATE > rgdMelFreqs[k]) {
			if (k < CHANNEL)
				k++; // k가 몇까지 올라가는지 확인 0~38의 값이어야 함.
		}
		rgdFiBins[i] = k;
	}

	for (int i = 0; i < KEEP_LEN; i++) {
		dTempBin = rgdFiBins[i];
		dTempFreq = (i / (double)(KEEP_LEN - 1)) * HALF_SAMPLING_RATE;
		if (dTempBin == 0) {
			rgdFilterBank[i] = (rgdMelFreqs[dTempBin] - dTempFreq) / (rgdMelFreqs[dTempBin] - 0);
		}
		else {
			rgdFilterBank[i] = (rgdMelFreqs[dTempBin] - dTempFreq) / (rgdMelFreqs[dTempBin] - rgdMelFreqs[dTempBin - 1]);
		}
		if (rgdFilterBank[i] < 0)
			rgdFilterBank[i] = 0;
	}
	return;
}

void MelFilterBank(double *dAbs, double *dMelFiltered) {

	double rgdMelFiltered[CHANNEL] = { 0, };
	for (int i = 0, k = 0; i < KEEP_LEN; i++) {
		k = rgdFiBins[i];

		if (k == 0) {
			rgdMelFiltered[k] += (1 - rgdFilterBank[i]) * dAbs[i];
		}
		else {
			rgdMelFiltered[k - 1] += rgdFilterBank[i] * dAbs[i];
			if (k != CHANNEL)
				rgdMelFiltered[k] += (1 - rgdFilterBank[i]) * dAbs[i];
		}
	}

	for (int i = 0; i < CHANNEL; i++) {
		dMelFiltered[i] = log(rgdMelFiltered[i]);
	}

}

void DCT(double *dMelFiltered, double *dMFCCFeature) {

	for (int i = 1; i <= MFCC_LEN; i++) {
		for (int k = 1; k <= CHANNEL; k++) {
			dMFCCFeature[i - 1] += sqrt(2.0 / CHANNEL) * dMelFiltered[k - 1] * cos(PI * i * (k - 0.5) / (double)CHANNEL);
		}
	}
}

void Liftering(double *dMFCCFeature) {

	for (int i = 1; i <= MFCC_LEN; i++) {
		//printf("i-th %d,  %f \n", i, (1 + 0.5 * LIFTER_LEN *sin(PI * i / LIFTER_LEN)));
		dMFCCFeature[i - 1] = dMFCCFeature[i - 1] * (1 + 0.5 * LIFTER_LEN *sin(PI * i / LIFTER_LEN));
	}

}

bool MFCCFeatureExtraction(short *rgsInputBuffer, double(*dMFCCFeature)[MFCC_LEN]) {

	fftw_complex fcInBefFFT[KEEP_LEN + BLOCK_LEN] = { 0, }, fcInAftFFT[KEEP_LEN + BLOCK_LEN] = { 0, };
	fftw_plan fpInFFT;
	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	short rgsProcessingBuffer[KEEP_LEN + BLOCK_LEN] = { 0, };
	double dAbs[WINDOW_LEN] = { 0, };
	double dMelFiltered[CHANNEL] = { 0, };

	memcpy(rgsProcessingBuffer, rgssKeepBuffer, KEEP_LEN * sizeof(short));
	memcpy(rgsProcessingBuffer + KEEP_LEN, rgsInputBuffer, BLOCK_LEN * sizeof(short));
	for (int k = 0; k < NUM_OF_FEATURE; k++) {

		// Pre-Emphasis
		for (int i = 1; i < WINDOW_LEN; i++) {
			fcInBefFFT[i][0] = rgsProcessingBuffer[i + k * KEEP_LEN] - 0.96 * rgsProcessingBuffer[i - 1 + k * KEEP_LEN];
		}
		// Windowing
		for (int i = 0; i < WINDOW_LEN; i++) {
			fcInBefFFT[i][0] *= (0.54 - 0.46 * cos(2 * PI * i / (WINDOW_LEN - 1)));
		}

		fpInFFT = fftw_plan_dft_1d(WINDOW_LEN, fcInBefFFT, fcInAftFFT, FFTW_FORWARD, FFTW_ESTIMATE); // ready
		fftw_execute(fpInFFT);
		for (int i = 0; i < WINDOW_LEN; i++) {
			dAbs[i] = sqrt(pow(fcInAftFFT[i][0], 2) + pow(fcInAftFFT[i][1], 2));
		}

		MelFilterBank(dAbs, dMelFiltered);
		memset(dMFCCFeature[k], 0, sizeof(double) * MFCC_LEN);
		DCT(dMelFiltered, dMFCCFeature[k]);
		Liftering(dMFCCFeature[k]);
		fftw_destroy_plan(fpInFFT);
	}
	memcpy(rgssKeepBuffer, &rgsInputBuffer[BLOCK_LEN - KEEP_LEN], KEEP_LEN * sizeof(short));
	//printf(" mfcc 0 %f, mfcc 1 %f, mfcc 2 %f \n", dMFCCFeature[3][0], dMFCCFeature[3][1], dMFCCFeature[3][2]);
	return TRUE;
}