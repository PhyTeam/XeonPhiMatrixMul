#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]){
	int len = 0;
	len = atoi(argv[1]);

	int i;
	FILE* file = fopen("A","wb");
	FILE* file2 = fopen("B","wb");
	printf("Len %d\n", len);

	for ( i = 0 ; i < len; ++i ){
		float *buff = malloc(len*sizeof(float));
		float *buff2 = malloc(len*sizeof(float));
		int j;
		for(j = 0; j < len; ++j){
			buff[j] = i + j;
			buff2[j] = j * i;
			fwrite(buff, sizeof(float), len, file);
			fwrite(buff2, sizeof(float), len, file2);
		}
	free(buff); free(buff2);
	}
	fclose(file);
	fclose(file2);
	return 0;
}
