/*
AnalysisAWGN
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

C++11에서 제공하는 API를 이용하여 normali distribution의 noise를 generation하였고,
이를 기존 clean 신호에 더해줌으로써, noisy신호를 만들 수 있다.

White Noise는 sample값들의 분포가 normal distribution을 따르는데 자세한 내용은 아래싸이트 참고바랍니다.
https://en.wikipedia.org/wiki/White_noise

AutoCorrelation과 PowerSpectralDensity의 Fourier Transform관계가 있고,
White Noise는 다른시간의 dependency가 없기 때문에, Rxx(0) = variance이고, Rxx(m) = 0 (단, m은 0이 아님)
Digital 신호에서 1 0 0 0 0 ... 이 되는 신호는 impulse이고 이를 FT.하면 상수값이 된다.
결국 PowerSpectralDensity는 상수가 되고, 전 대역에 일정한 에너지를 갖게 된다.
https://cnx.org/contents/cillqc8i@5/Autocorrelation-of-Random-Proc
실제 White noise의 Spectrogram을 보면 전 대역에 에너지가 골고루 일정하게 존재한다.

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
#include<stdlib.h> // for using rand() function.
#include<Windows.h>
#include<random>
#include<chrono>

#define BLOCK_SIZE 512 // 1000*(512/16000)= 32ms.
#define FFT_PROCESSING_SIZE 1024// 2의 n승값이 되어야 한다.
#define KEEP_LENGTH 512
#define DEFAULT_SAMPLINGRATE 16000.0

void GaussianRandom(short* rgdNoise, double dAverage, double dStddev);
void AnalysisAdditiveWhiteGaussianNoise(short *psNoiseBuffer, int iFrameCount);
void AddAdditiveWhiteGaussianNoise(short *psInputBuffer, short *psOutputBuffer, int iFrameCount);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_SIZE] = { 0, };
	short rgsOutputBuffer[BLOCK_SIZE] = { 0, };

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

	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_SIZE, fpRead)) != BLOCK_SIZE) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		AddAdditiveWhiteGaussianNoise(rgsInputBuffer, rgsOutputBuffer, BLOCK_SIZE);
		fwrite(rgsOutputBuffer, sizeof(short), BLOCK_SIZE, fpWrite);
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;
}

void GaussianRandom(short* rgdNoise, double dAverage, double dStddev) {
	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	std::normal_distribution<double> distribution(dAverage, dStddev);

	for (int i = 0; i < BLOCK_SIZE; i++) {
		rgdNoise[i] = distribution(generator);
	}
}

void AnalysisAdditiveWhiteGaussianNoise(short *psNoiseBuffer, int iFrameCount) {

	fftw_complex fcInputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcInputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcOutputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcOutputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_plan fpInput_p, fpOutput_p;
	static short rgssKeepBuffer[KEEP_LENGTH] = { 0, };
	double dAutoCorrelation[BLOCK_SIZE] = { 0, };

	for (int i = 0; i < KEEP_LENGTH; i++) {
		fcInputBefFFT[i][0] = rgssKeepBuffer[i];
	}
	for (int i = 0; i < iFrameCount; i++) {
		fcInputBefFFT[KEEP_LENGTH + i][0] = psNoiseBuffer[i];
	}
	fpInput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcInputBefFFT, fcInputAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);

	fpOutput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcOutputAftFFT, fcOutputBefFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fpInput_p);

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		fcOutputAftFFT[i][0] = fcInputAftFFT[i][0] * fcInputAftFFT[i][0] + fcInputAftFFT[i][1] * fcInputAftFFT[i][1];
		fcOutputAftFFT[i][1] = 0;
	}
	fftw_execute(fpOutput_p);
	for (int i = 0; i < iFrameCount; i++) {
		dAutoCorrelation[i] = fcOutputBefFFT[i][0] * 1. / FFT_PROCESSING_SIZE;
	}
	//Check AutoCorrelation Vector


	// 버퍼의 맨 마지막 부분을 KEEP_LENGTH 만큼 Keep함.
	memcpy(rgssKeepBuffer, &psNoiseBuffer[iFrameCount - KEEP_LENGTH], sizeof(rgssKeepBuffer));
	fftw_destroy_plan(fpInput_p);
	fftw_destroy_plan(fpOutput_p);
	return;
}

void AddAdditiveWhiteGaussianNoise(short *psInputBuffer, short *psOutputBuffer, int iFrameCount) {
	short rgdNoise[BLOCK_SIZE] = { 0, };
	double dTargetMean = 0.0, dTargetStd = 10.0;
	srand(GetTickCount());
	GaussianRandom(rgdNoise, dTargetMean, dTargetStd);
	for (int i = 0; i < iFrameCount; i++) {
		psOutputBuffer[i] = rgdNoise[i] + psInputBuffer[i];
	}
	AnalysisAdditiveWhiteGaussianNoise(rgdNoise, iFrameCount);
}