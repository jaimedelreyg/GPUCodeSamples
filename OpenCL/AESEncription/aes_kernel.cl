//OPENCL KERNEL FOR AES ENCRIPTION CALCULATION

void printWindow(double* window){

printf("%lf ",window[0]);
printf("%lf ",window[1]);
printf("%lf ",window[2]);
printf("%lf \n",window[3]);
printf("%lf ",window[4]);
printf("%lf ",window[5]);
printf("%lf ",window[6]);
printf("%lf \n",window[7]);
printf("%lf ",window[8]);
printf("%lf ",window[9]);
printf("%lf ",window[10]);
printf("%lf \n",window[11]);
printf("%lf ",window[12]);
printf("%lf ",window[13]);
printf("%lf ",window[14]);
printf("%lf \n\n",window[15]);

}

__kernel
void aes_encription(__global double* input, __global char *keys, __global uint *lookup_table, uint width)
{

int round=0;
int i = 0;
int j = 0;
int z = 0;
double res;
double window_a[16];
double window_b[16];
double mix_col_matrix[16] = {2,3,1,1,1,2,3,1,1,1,2,3,3,1,1,2};
double mix_aux[4];
int x = (int)get_global_id(0);
int y = (int)get_global_id(1);
x = x*4;
y = y*4;

//Fill the window
for(i=0;i<4;i++){
  for(j=0;j<4;j++){
    window_a[i*4+j]=input[(x+i)*width+(y+j)];
  }
}

if((x==0)&&(y==0)){
  printf("Encript process!\n");
}
while(round < 1){
    
  if((x==0)&&(y==0)){
  printf("ROUND %i\n",round);
  printWindow(window_a);
  }
  
  //SubBytes step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      window_b[lookup_table[i*4+j]]=window_a[i*4+j];
    }
  }
  
  if((x==0)&&(y==0)){
  printf("Subbytes\n");
  printWindow(window_b);
  }
      
  //Shift Rows step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
	if(j-i < 0) window_a[i*4+((j-i)+4)] = window_b[i*4+j];
	else window_a[i*4+(j-i)] = window_b[i*4+j];
    }
  }
      
  if((x==0)&&(y==0)){
  printf("ShiftRows\n");
  printWindow(window_a);
  }
  
  //MixColumns step
  for(i=0;i<4;i++){
    mix_aux[0] = window_a[i]; 
    mix_aux[1] = window_a[4+i];
    mix_aux[2] = window_a[8+i];
    mix_aux[3] = window_a[12+i];

    for(j=0;j<4;j++){
    	res=0;
        res += mix_col_matrix[j*4+0]*mix_aux[0];
        res += mix_col_matrix[j*4+1]*mix_aux[1];
        res += mix_col_matrix[j*4+2]*mix_aux[2];
        res += mix_col_matrix[j*4+3]*mix_aux[3];
        
    window_a[2*j+i] = res;
    }
  } 

  if((x==0)&&(y==0)){
  printf("MixCols\n");
  printWindow(window_a);
  }
  

  //Add round key step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
	window_b[i*4+j] = (int)window_a[i*4+j] ^ (int)keys[round*10+(i*4+j)];	  
    }
  }
  

  if((x==0)&&(y==0)){
  printf("AddRoundKey\n");
  printWindow(window_b);
  }
round++;

}

//Return the window to the input
for(i=0;i<4;i++){
  for(j=0;j<4;j++){
    input[(x+i)*width+(y+j)] = window_b[i*4+j];
  }
}
}

__kernel
void aes_decription(__global double* input, __global char *keys, __global uint *lookup_table, uint width)
{

int round=0;
uint i = 0;
uint j = 0;
uint z = 0;
double res;
double window_a[16];
double window_b[16];
double mix_col_matrix[16] = {2,3,1,1,1,2,3,1,1,1,2,3,3,1,1,2};
double mix_aux[4];
uint x = (int)get_global_id(0);
uint y = (int)get_global_id(1);
x = x*4;
y = y*4;

////Fill the window
for(i=0;i<4;i++){
  for(j=0;j<4;j++){
    window_a[i*4+j]=input[(x+i)*width+(y+j)];
  }
}

if((x==0)&&(y==0)){
  printf("Decript process!\n");
}

while(round < 1){
    
  if((x==0)&&(y==0)){
  printf("ROUND %i\n",round);
  printWindow(window_a);
  }
  
  //Shift Rows step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
	if(j+i > 4) window_a[i*4+((j+i)-4)] = window_b[i*4+j];
	else window_a[i*4+(j+i)] = window_b[i*4+j];
    }
  }
      
  if((x==0)&&(y==0)){
  printf("ShiftRows\n");
  printWindow(window_a);
  }
  
  //SubBytes step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      window_a[i*4+j]=window_b[lookup_table[i*4+j]];
    }
  }
  
  if((x==0)&&(y==0)){
  printf("Subbytes\n");
  printWindow(window_b);
  }
  
  //MixColumns step
  for(i=0;i<4;i++){
    mix_aux[0] = window_a[i]; 
    mix_aux[1] = window_a[4+i];
    mix_aux[2] = window_a[8+i];
    mix_aux[3] = window_a[12+i];

    for(j=0;j<4;j++){
    	res=0;
        res += mix_col_matrix[j*4+0]/mix_aux[0];
        res += mix_col_matrix[j*4+1]/mix_aux[1];
        res += mix_col_matrix[j*4+2]/mix_aux[2];
        res += mix_col_matrix[j*4+3]/mix_aux[3];
        
    window_a[2*j+i] = res;
    }
  } 

  if((x==0)&&(y==0)){
  printf("MixCols\n");
  printWindow(window_a);
  }
  //Add round key step
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
	window_b[i*4+j] = (int)window_a[i*4+j] ^ (int)keys[round*10+(i*4+j)];	  
    }
  }
  

  if((x==0)&&(y==0)){
  printf("AddRoundKey\n");
  printWindow(window_b);
  }
  

      
round++;

}

//Return the window to the input
for(i=0;i<4;i++){
  for(j=0;j<4;j++){
    input[(x+i)*width+(y+j)] = window_b[i*4+j];
  }
}
}

