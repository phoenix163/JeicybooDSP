/*
Linear Prediction Coding Estimation.
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

We are targetting format of 16kHz SamplingRate, mono channel, 16bit per sample.

구현내용 정리.

자기자신을 Estimation하는 Filter의 Coefficient를 구하면 그것이 LPC가 된다.
값에 의미는 음성에서는 Vocal tract(성도)에 해당하는 성분이다.
나온결과를 FFT하면, Speech를 FFT 했을때의 Evelope와 같은 결과가 나오고,
이 Evelope의 peak가 되는 Frequency를 Formant가 된다.

Yule-Walker Equation을 풀어야 하기 때문에 Matrix inverse연산이 필요하다.
Matrix 연산을 해야되어서, Eigen lib를 미리 설치해 둘 것.(http://eigen.tuxfamily.org/index.php?title=Main_Page
http://blog.daum.net/pg365/156 참고)

LPC를 이해하는데 http://slideplayer.com/slide/2390999/ 자료 참고할 것.

*/
#include<stdio.h>
#include<string.h>
#include<fftw3.h>
#include<math.h>
#include<Eigen/Dense>
using Eigen::MatrixXd;

#define LPC_LEN 12
#define FALSE 0
#define TRUE 1
#define PI 3.141592
#define BLOCK_LEN 256 // 1000*(1024/44100)= 23ms.

bool LPCEstimation(short *rgsInputBuffer, double *dLPCFeature);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	FILE *fpWrite;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };
	double dLPCFeature[LPC_LEN] = { 0, };

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
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}

		if (LPCEstimation(rgsInputBuffer, dLPCFeature)) { // TRUE가 return되었을 경우에만 file에 쓴다.
			printf("dLPCFeature[0] %f, dLPCFeature[1] %f, dLPCFeature[2] %f \n", dLPCFeature[0], dLPCFeature[1], dLPCFeature[2]);

			fwrite(dLPCFeature, sizeof(double), LPC_LEN, fpWrite);
		}
	}
	printf("Processing End\n");
	fclose(fpRead);
	fclose(fpWrite);
	getchar();
	return;
}

bool LPCEstimation(short *rgsInputBuffer, double *dLPCFeature) {

	MatrixXd mxToeplitzMatrix = MatrixXd(LPC_LEN, LPC_LEN);
	MatrixXd mxToeplitzVector = MatrixXd(LPC_LEN, 1);
	MatrixXd LPCFeature = MatrixXd(LPC_LEN, 1);
	static int iNumOfIteration = 0;
	static short rgssKeepBuffer[BLOCK_LEN] = { 0, };
	short rgsProcessingBuffer[BLOCK_LEN * 2] = { 0, };
	double rgdAftWindow[BLOCK_LEN * 2] = { 0, };
	double rgdResult[LPC_LEN + 1] = { 0, };

	iNumOfIteration++;
	// copy to processingbuffer (keepBuffer + InputBuffer)
	memcpy(rgsProcessingBuffer, rgssKeepBuffer, sizeof(short) * BLOCK_LEN);
	memcpy(rgsProcessingBuffer + BLOCK_LEN, rgsInputBuffer, sizeof(short) * BLOCK_LEN);

	// multiply window.
	for (int i = 0; i < BLOCK_LEN * 2; i++) {
		rgdAftWindow[i] = rgsProcessingBuffer[i] * (double)(0.54 - 0.46 * cos(2 * PI * i / (double)(2 * BLOCK_LEN - 1)));
	}

	for (int i = 0; i < LPC_LEN + 1; i++) {
		for (int j = 0; j < BLOCK_LEN * 2 - i; j++) {
			rgdResult[i] += rgdAftWindow[j] * rgdAftWindow[j + i];
		}
		rgdResult[i] /= BLOCK_LEN * 2 - i;
	}
	// Calc mxToeplitzMatrix
	for (int i = 0; i < LPC_LEN; i++) {
		for (int j = 0; j < LPC_LEN; j++) {
			mxToeplitzMatrix(i, j) = rgdResult[abs(i - j)];
		}
	}
	// Calc mxToeplitzVector
	for (int i = 0; i < LPC_LEN; i++) {
		mxToeplitzVector(i, 0) = -rgdResult[i + 1];
	}

	// Calc LPC Feature
	LPCFeature = mxToeplitzMatrix.inverse() * mxToeplitzVector;
	// Copy to output buffer.
	for (int i = 0; i < LPC_LEN; i++) {
		dLPCFeature[i] = LPCFeature(i);
	}
	// Save KeepBuffer
	memcpy(rgssKeepBuffer, rgsInputBuffer, sizeof(short) * BLOCK_LEN);
	if (iNumOfIteration > 1)
		return TRUE;
	else
		return FALSE;
}