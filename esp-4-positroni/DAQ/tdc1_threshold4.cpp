/* 
 * Continuous TDC read
 * MSS 11.2015
*/

#include <libxxusb.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <string.h>
#include <iostream>

using namespace std;

char *append_txt(char *str) {
  size_t len = strlen(str);
  size_t newlen = len + strlen(".txt");
  char *newstr = (char *)malloc(sizeof(char) * (newlen + 1));
  strcpy(newstr, str);
  strcat(newstr, ".txt");
  return newstr;
}


int main (int argc,  char *argv[]) {
  int CamN, CamA, CamF;
  long int CamD;
  int CamQ, CamX;
  //  char nafin[20];
  char nafinx[20];
  
  int ret, i, nDevices;
  xxusb_device_type devices[100]; 
  struct usb_device *dev;
  usb_dev_handle *udev;       // Device Handle 
  //  int inhibit;
  time_t current_time;
  char* c_time_string;
  FILE *OutF;
  char OutputFileName[] = "TDC-PETData.txt";
  int NEvt=0;

  // Open the output file
  if (argc>0) 
    OutF = fopen(append_txt(argv[1]),"a"); // non cancella il file
  else
    OutF = fopen(OutputFileName,"a");
  
   setvbuf(OutF,NULL,_IOLBF,0);
  
  // Find XX_USB devices and open the first one found
  nDevices = xxusb_devices_find(devices);
  if (nDevices<0) {
    printf ("Privilegi insufficienti - root?\n");
    return 1;
  }
  //  cout << "Trovati " << nDevices << " controller" << endl;
  dev = devices[0].usbdev;
  udev = xxusb_device_open(dev); 
  // Make sure CC_USB opened OK
  if (!udev) {
    printf ("Errore nell'inizializzazione del CC\n");
    return 1;
  }

  // Initialize
  CAMAC_Z(udev);
  CAMAC_I(udev,true);
  CAMAC_write(udev,1,0,16,0xaaaaaa,&CamQ,&CamX);
  CAMAC_read(udev,1,0,0,&CamD,&CamQ,&CamX);
  CAMAC_C(udev);
  CAMAC_I(udev,false);
  CAMAC_Z(udev);
  //  inhibit=0;

  CamN=8;/*
  while (CamN==0)  {
    printf("TDC slot? ");
    fflush(stdin);
    scanf("%s",nafinx);
    sscanf(nafinx,"%i",&CamN);
    fflush(stdin);
    if (CamN<1 || CamN>24) CamN=0;
  }*/
  CamA=1;
  CamF=26;
  ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
  printf("Enabled LAM at Add 1\n");
  if (ret<0 || CamX==0) {
    printf("LAM enable fallito\n");
    return 1;
  }
  
  int Nevt_4 = 0;
  int letture[4];
  int letture_fallite;
  int letture_4;
  while(1) {
    CamA=0;
    CamF=8;
    //printf("Test LAM at Add 0\n");    
    ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
    //printf("Test LAM at Add 0: ret %i, CamQ %lx \n",ret,(long unsigned int)CamQ);    
    if (ret<0 || CamX==0) printf("Lettura LAM fallita\n");
    //else if (CamQ==1) {
    else {
      CAMAC_I(udev,true);
      // Read four TDC channels
      letture_fallite = 0;
      for (i=0;i<4;i++) {
	      CamA=i;
	      CamF=0;
	      ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
	      if (ret<0 || CamX==0) {
	        printf("Lettura fallita\n");
	        letture_fallite = 1;
	        break;
	      } else {
	        letture[i] = CamD;
	      }
	    }

	    if (!letture_fallite) { // if success
	      // check if all readouts are under threshold
	      letture_4 = 1;
	      for (i = 0; i < 4; ++i) {
	        letture_4 = letture_4 && (letture[i] <= 4);
	      }
	      if (letture_4) { // under threshold
	        Nevt_4++;
	      } else { // over threshold
	        NEvt++;
          // get current time with subsecond precision
          struct timespec spec;
          clock_gettime(CLOCK_REALTIME, &spec);
          double float_time = spec.tv_sec + 1e-9 * spec.tv_nsec;
          // print number of events and human-readble timestamp on terminal
          printf("(%5d) (%5d) ", NEvt, Nevt_4);
          current_time = time(NULL);
          c_time_string = ctime(&current_time);
          c_time_string[strlen(c_time_string) - 1] = '\0';
          printf("[%s]",c_time_string);
          // print and save readouts  
  	      for (i = 0; i < 4; ++i) {
  	        printf("%5d", letture[i]);
  	        fprintf(OutF, "%5d", letture[i]);
  	      }
          printf("\n");
          fprintf(OutF, " %.6lf\n", float_time); // save timestamp
        }
	    }
      CAMAC_C(udev);
      CAMAC_I(udev,false);
      usleep(1000);
    }
  }
  // Close the Device
  xxusb_device_close(udev);

  //Close the file
  fclose(OutF);
  
  return 0;
}

