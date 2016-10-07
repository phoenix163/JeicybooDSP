/*
VAD based Noise Spectrum Estimation.
Apply to Spectral Subtraction/Wiener Filter.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

구현 시 고려사항.

이전 Study에서 언급했던 방법은. 초기의 수백 ms정도를 Noise라고 가정하고, Estimation하였다.
그러나, recording이 시작 된 이후에 Noise update문제가 발생 할 수 있으며, 녹음 초반에 음성이 들어 갈 수 있기 때문에
Voice Activity Detection이 필요하다.

VAD에 오래전부터 사용되어 왔던 특징인 ZCR과 Energy를 이용하여 음성구간/비음성 구간을 분별하고,
비음성 구간의 경우에는 Estimation하는 Noise Spectrum을 update시킨다.
https://www.asee.org/documents/zones/zone1/2008/student/ASEE12008_0044_paper.pdf


1) Zero Crossing Rate
: Time domain에서 Amplitude 0값을 기준으로 얼마나 자주 교차되는가를 측정한 feature이다.
잡음일수록 영교차율 값은 큰 값을 나타내며, 음성인 경우 영교차율 값은 작은 경향이 있다.
2) Energy
: WAV파형의 Envelope와 비슷하다.

ZCR과 Energy두개에 대한 각각의 Threadhold기반의 Decision이 이뤄져야 하는데,
음성구간을 비음성이라 판단하는 것이, 비음성구간을 음성이라 판단하는 것보다 risk가 더 크다.
왜냐하면, 음성구간을 비음성이라고 판단하면, 잡음 추정에 음성특성이 들어가서, 차감된 결과는 왜곡되게 되기 때문이다.
연구결과들을 검색해 보면, Energy에서 검출하지 못하는 음성구간이 있고, ZCR에서 검출하지 못하는 음성구간이 있다.
따라서, 음성검출 조건은 다음과 같이 or 연산을 적용하기로 한다.
조건 : (ZCR < Threadhold(ZCR) || Energy > Threadhold(Energy))

참고로 이 잡음제거는 음성인식이 아닌 Voice Recorder에 적용되는 경우를 고려한다면
음성구간이든 비음성구간이든 항상 노이즈를 제거해야한다.

아래 두 특징 모두 SNR이 낮은 환경에서는 Detection성능이 떨어진다.
추후 GMM Study가 끝나면, Voice / UnVoiced 의 GMM 확률모델링을 한후 Likelihood Ratio Test를 통해 Decision하는 방법도 생각 해보자.

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>


#define THRESHOLD_OF_ENERGY 700.0
#define THRESHOLD_OF_ZCR 200.0
#define FALSE 0
#define TRUE 1
#define PI 3.141592
#define KEEP_LEN 512
#define BLOCK_LEN 512 // 1000*(512/16000)= 32ms.
#define FFT_PROCESSING_SIZE 1024// 2의 n승값이 되어야 한다.
#define NOISE_ESTIMATION_FRAMECOUNT 10.0 

bool VoiceActivityDetection(short *rgsInputBuffer, int iFrameCount);
void EstimateNoiseSpectrum(short *rgsTempBuffer, int iNumOfIteration, short *psInputBuffer, double *pdEstimatedNoiseSpec, int iFrameCount);
bool SpectralSubtraction(short *psInputBuffer, double *pdEstimatedNoiseSpec, short *psOutputBuffer, int iFrameCount);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };
	short rgsTempBuffer[BLOCK_LEN] = { 0, };
	double rgdEstimatedNS[FFT_PROCESSING_SIZE] = { 0, };
	short rgsOutputBuffer[BLOCK_LEN] = { 0, };
	int iNumOfIteration = 0;
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
	//fread(rgcHeader, 1, 44, fpRead);
	//fwrite(rgcHeader, 1, 44, fpWrite);

	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		if (!VoiceActivityDetection(rgsInputBuffer, BLOCK_LEN)) {
			iNumOfIteration++;
			if (iNumOfIteration == 1) {
				memcpy(rgsTempBuffer, rgsInputBuffer, sizeof(rgsInputBuffer));
			}
			else if (iNumOfIteration > 1) {
				EstimateNoiseSpectrum(rgsTempBuffer, iNumOfIteration, rgsInputBuffer, rgdEstimatedNS, BLOCK_LEN);
			}
		}
		else {
			iNumOfIteration = 0;
		}

		if (SpectralSubtraction(rgsInputBuffer, rgdEstimatedNS, rgsOutputBuffer, BLOCK_LEN)) // TRUE가 return되었을 경우에만 file에 쓴다.
			fwrite(rgsOutputBuffer, sizeof(short), BLOCK_LEN, fpWrite);
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;
}

bool VoiceActivityDetection(short *rgsInputBuffer, int iFrameCount) {

	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	short rgsProcessingBuffer[FFT_PROCESSING_SIZE] = { 0, };
	double dEnergy = 0.0;
	int dZCR = 0;
	memcpy(rgsProcessingBuffer, rgssKeepBuffer, sizeof(rgssKeepBuffer));
	memcpy(rgsProcessingBuffer + KEEP_LEN, rgsInputBuffer, sizeof(short) * iFrameCount);

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgsProcessingBuffer[i] *= (0.54 - 0.46 * cos(2 * PI * i / (FFT_PROCESSING_SIZE - 1))); // Windowing 
		// 조금 조심 할 부분, short가 되어서, 소수점 짤림.

		//Calc Energy
		dEnergy += pow(rgsProcessingBuffer[i], 2.0);

		//Calc Zero Crossing Rate
		if (i != FFT_PROCESSING_SIZE) {
			if (rgsProcessingBuffer[i] * rgsProcessingBuffer[i + 1] < 0)
				dZCR++;
		}
	}
	dEnergy /= FFT_PROCESSING_SIZE;
	//dZCR /= (FFT_PROCESSING_SIZE - 1);

	printf(" dEnergy %f , dZCR %d \n", dEnergy, dZCR);
	if (dEnergy > THRESHOLD_OF_ENERGY || dZCR < THRESHOLD_OF_ZCR) {
		return TRUE;
	}
	else {
		return FALSE;
	}

	memcpy(rgssKeepBuffer, &rgsInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));

}


void EstimateNoiseSpectrum(short *rgsTempBuffer, int iNumOfIteration, short *psInputBuffer, double *pdEstimatedNoiseSpec, int iFrameCount) {

	static double rgsdAveragedNS[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcInputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcInputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_plan fpInput_p;
	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	if (iNumOfIteration == 2) {
		memcpy(rgssKeepBuffer, rgsTempBuffer, sizeof(rgssKeepBuffer));
	}
	for (int i = 0; i < KEEP_LEN; i++) {
		fcInputBefFFT[i][0] = rgssKeepBuffer[i]; // copy keepBuffer to processing buffer.
	}
	for (int i = 0; i < iFrameCount; i++) {
		fcInputBefFFT[KEEP_LEN + i][0] = psInputBuffer[i]; // copy inputBuffer to processing buffer.
	}

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		fcInputBefFFT[i][0] *= (0.54 - 0.46 * cos(2 * PI * i / (FFT_PROCESSING_SIZE - 1))); // Windowing
	}

	fpInput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcInputBefFFT, fcInputAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpInput_p);

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgsdAveragedNS[i] += sqrt(fcInputAftFFT[i][0] * fcInputAftFFT[i][0] + fcInputAftFFT[i][1] * fcInputAftFFT[i][1]); // FFT변환 후 각 magnitude값을 구함.
		if (iNumOfIteration >= 3) { // 3번째 버퍼부터 2개가 되므로, 이때부터 나누기 2 할 것.
			rgsdAveragedNS[i] /= 2.0;
		}
	}

	if (iNumOfIteration == NOISE_ESTIMATION_FRAMECOUNT) {
		for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
			pdEstimatedNoiseSpec[i] = rgsdAveragedNS[i]; // / (NOISE_ESTIMATION_FRAMECOUNT - 1); // output Buffer로의 copy
		}
	}
	//printf("pdEstimatedNoiseSpec[0] %f, pdEstimatedNoiseSpec[1] %f pdEstimatedNoiseSpec[2] %f \n", pdEstimatedNoiseSpec[0], pdEstimatedNoiseSpec[1], pdEstimatedNoiseSpec[2]);
	memcpy(rgssKeepBuffer, &psInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));
	fftw_destroy_plan(fpInput_p);
	return;
}


bool SpectralSubtraction(short *psInputBuffer, double *pdEstimatedNoiseSpec, short *psOutputBuffer, int iFrameCount) {
	static int iNumOfIteration = 0;
	double rgdSubtractedAmp[FFT_PROCESSING_SIZE] = { 0, };
	double rgdSaveAngle[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcInputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcInputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_complex fcOutputBefFFT[FFT_PROCESSING_SIZE] = { 0, }, fcOutputAftFFT[FFT_PROCESSING_SIZE] = { 0, };
	fftw_plan fpInput_p, fpOutput_p;
	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	static double rgsdOveraped[FFT_PROCESSING_SIZE] = { 0, }; // Overlap and Saved method
	iNumOfIteration++;
	if (iNumOfIteration == 1) {
		memcpy(rgssKeepBuffer, &psInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));
		//fftw_destroy_plan(fpInput_p);
		//fftw_destroy_plan(fpOutput_p);
		return FALSE; // 첫번째 frame은 2개의 버퍼가 준비가 안됐기 때문에 연산하지 않음.
	}

	for (int i = 0; i < KEEP_LEN; i++) {
		fcInputBefFFT[i][0] = rgssKeepBuffer[i];
	}
	for (int i = 0; i < iFrameCount; i++) {
		fcInputBefFFT[KEEP_LEN + i][0] = psInputBuffer[i];
	}

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		fcInputBefFFT[i][0] *= (0.54 - 0.46 * cos(2 * PI * i / (FFT_PROCESSING_SIZE - 1)));
	}

	fpInput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcInputBefFFT, fcInputAftFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fpInput_p);

	// Save phase information.
	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgdSaveAngle[i] = atan2(fcInputAftFFT[i][1], fcInputAftFFT[i][0]); // 복소수의 angle 정보 저장.
	}

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgdSubtractedAmp[i] = sqrt(fcInputAftFFT[i][0] * fcInputAftFFT[i][0] + fcInputAftFFT[i][1] * fcInputAftFFT[i][1]) - pdEstimatedNoiseSpec[i]; // Noise의 Spectrum Magnitude를 신호에서 차감한다.
		// 혹시 음수가 저장되진 않았는지 확인.
		fcOutputAftFFT[i][0] = rgdSubtractedAmp[i] * cos(rgdSaveAngle[i]); // 저장한 angle정보를 통해 복소수 복원.
		fcOutputAftFFT[i][1] = rgdSubtractedAmp[i] * sin(rgdSaveAngle[i]);
	}

	fpOutput_p = fftw_plan_dft_1d(FFT_PROCESSING_SIZE, fcOutputAftFFT, fcOutputBefFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(fpOutput_p);

	for (int i = 0; i < FFT_PROCESSING_SIZE; i++) {
		rgsdOveraped[i] += 1. / FFT_PROCESSING_SIZE * fcOutputBefFFT[i][0]; // FFTW lib.의 IFFT 연산은 1/N연산이 생략되었기 때문에 iFFT 시 꼭 해줘야 함. 
	}

	for (int i = 0; i < iFrameCount; i++) {
		psOutputBuffer[i] = rgsdOveraped[i];
	}
	printf("psOutputBuffer[0] %d, psOutputBuffer[1] %d psOutputBuffer[2] %d \n", psOutputBuffer[0], psOutputBuffer[1], psOutputBuffer[2]);
	memcpy(rgsdOveraped, rgsdOveraped + KEEP_LEN, sizeof(rgsdOveraped) / 2); // Overlap 버퍼에서 뒷쪽 버퍼를 앞으로 당겨 옴.
	memset(rgsdOveraped + KEEP_LEN, 0, sizeof(rgsdOveraped) / 2); // Overlap 버퍼에서 절반은 비움.
	memcpy(rgssKeepBuffer, &psInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));
	fftw_destroy_plan(fpInput_p);
	fftw_destroy_plan(fpOutput_p);
	if (iNumOfIteration >= 3)
		return TRUE;
	else
		return FALSE;
}