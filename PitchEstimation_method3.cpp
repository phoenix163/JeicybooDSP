/*
Pitch Estimation
This program is made by jongcheol boo.

변수 이름은 헝가리안 표기법을따랐다.
http://jinsemin119.tistory.com/61 , https://en.wikipedia.org/wiki/Hungarian_notation , http://web.mst.edu/~cpp/common/hungarian.html

음성데이터의 PCM을 time scale을 늘려서 보면, 주기적으로 반복되는 형태임을 관찰 할 수 있다.
이 반복되는 시간이 주기이고, 이 주기의 역수가 바로 목소리의 기본주파수 pitch가 된다.

Pitch Estimation의 세가지 방법을 구현.
1) Method 1 : Using relation of (Autocorrelation --->(FFT) PowerSpectralDensity)
http://ktword.co.kr/abbr_view.php?m_temp1=3726&m_search=S

2) Method 2 : Using AMDF(Average Magnitude Difference Function)
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.590.5684&rep=rep1&type=pdf

3) Method 3 : Using AutoCorrelation
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.590.5684&rep=rep1&type=pdf

*/
#include<stdio.h>
#include<string.h>
#include<math.h>

#define BLOCK_LEN 512 // 1000*(512/16000)= 32ms.
#define ROCESSING_LEN 1024// 2의 n승값이 되어야 한다.
#define KEEP_LEN 512
#define DEFAULT_SAMPLINGRATE 16000.0

void CalcPitch(short *psInputBuffer, int iFrameCount);

void main(int argc, char** argv) {

	// fRead는 input, fWrite는 processing이후 write 될 output file pointer.
	FILE *fpRead;
	char rgcHeader[44] = { '\0', }; // header를 저장할 배열.
	short rgsInputBuffer[BLOCK_LEN] = { 0, };

	if (argc != 2) {
		printf("path를 1개 입력해야 합니다.\n"); // input path, output path
		return;
	}
	else {
		for (int i = 1; i < 2; i++)
			printf("%d-th path %s \n", i, argv[i]);
	}

	if ((fpRead = fopen(argv[1], "rb")) == NULL)
		printf("Read File Open Error\n");

	// Read한 Wav 파일 맨 앞의 Header 44Byte 만큼 Write할 Wav파일에 write.
	fread(rgcHeader, 1, 44, fpRead);

	while (true)
	{
		if ((fread(rgsInputBuffer, sizeof(short), BLOCK_LEN, fpRead)) == 0) {
			printf("Break! The buffer is insufficient.\n");
			break;
		}
		CalcPitch(rgsInputBuffer, BLOCK_LEN);
	}
	printf("Processing End\n");
	fclose(fpRead);
	getchar();
	return;
}

void CalcPitch(short *psInputBuffer, int iFrameCount) {

	static short rgssKeepBuffer[KEEP_LEN] = { 0, };
	double dAutoCorrelation[BLOCK_LEN] = { 0, };
	short rgsProcessingBuffer[ROCESSING_LEN] = { 0, };
	double dMax = 0;
	int iArg = 0;
	memcpy(rgsProcessingBuffer, rgssKeepBuffer, sizeof(rgssKeepBuffer));
	memcpy(rgsProcessingBuffer + KEEP_LEN, psInputBuffer, sizeof(short) * iFrameCount);

	for (int k = 0; k < BLOCK_LEN; k++) {
		for (int i = 0; i < ROCESSING_LEN - k; i++) {
			dAutoCorrelation[k] += rgsProcessingBuffer[i] * rgsProcessingBuffer[i + k];
		}
		dAutoCorrelation[k] /= (double)(ROCESSING_LEN - k);
	}

	//Calculate pitch : find min arg of AutoCorrelation
	dMax = dAutoCorrelation[iFrameCount - 1];
	printf(" dMin %f \n", dMax);
	for (int i = iFrameCount - 1; i > 100; i--) {

		if (dAutoCorrelation[i] >= dMax) {
			iArg = i;
			dMax = dAutoCorrelation[i];
		}
	}
	printf("Estimation arg %d , dMin %f pitch %f \n", iArg, dMax, (DEFAULT_SAMPLINGRATE / (double)iArg));

	// 버퍼의 맨 마지막 부분을 KEEP_LENGTH 만큼 Keep함. 
	memcpy(rgssKeepBuffer, &psInputBuffer[iFrameCount - KEEP_LEN], sizeof(rgssKeepBuffer));
	return;
}
