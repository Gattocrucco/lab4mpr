/* 
 * Continuous ADC read
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

int main (int argc,  char *argv[]) {
  int CamN, CamA, CamF;
  long int CamD;
  int CamQ, CamX;
  char nafin[20];
  char nafinx[20];
  int count;
  int ret, i, nDevices;
  xxusb_device_type devices[100]; 
  struct usb_device *dev;
  usb_dev_handle *udev;       // Device Handle 
  int inhibit;
  time_t current_time;
  char* c_time_string;
  FILE *OutF;
  char OutputFileName[] = "PositronData.txt";
  int NEvt=0;

   // Open the output file
  if (argc>0) 
    OutF = fopen(argv[1],"w");
  else
    OutF = fopen(OutputFileName,"w");

  // Find XX_USB devices and open the first one found
  nDevices = xxusb_devices_find(devices);
  if (nDevices<0) {
    printf ("Privilegi insufficienti - root?\n");
    return 1;
  }
  cout << "Trovati " << nDevices << " controller" << endl;
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
  inhibit=0;

  CamN=0;
  while (CamN==0)  {
    printf("ADC slot? ");
    fflush(stdin);
    scanf("%s",nafinx);
    sscanf(nafinx,"%i",&CamN);
    fflush(stdin);
    if (CamN<1 || CamN>24) CamN=0;
  }
  CamA=1;
  CamF=26;
  ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
  if (ret<0 || CamX==0) {
    printf("LAM enable fallito\n");
    return 1;
  }
  
  while(NEvt < 1000) {
    //
    CamA=0;
    CamF=8;
    ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
    //printf("---->  CamQ: 0x%lx \n",CamQ);
    if (ret<0 || CamX==0) printf("Lettura LAM fallita\n");
    else if (CamQ==1) {
      NEvt++;
      printf("Event: %i \n",NEvt);
      //      fprintf(OutF,"Event: %i \n",NEvt);
      //CAMAC_I(udev,true);
      current_time = time(NULL);
      c_time_string = ctime(&current_time);
      printf("%s\n",c_time_string);
      for (i=0;i<12;i++) {
	CamA=i;
	CamF=2;
	ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
	if (ret<0 || CamX==0) printf("Lettura fallita\n");
	else {
	  	//  printf(", CH %2d: (%s) D=0x%lx",CamA,CamQ?"Q":" ",(unsigned int)CamD);
	  	// fprintf(OutF,", CH %2d: (%s) D=0x%lx",CamA,CamQ?"Q":" ",(unsigned int)CamD);
	  	//  printf(", CH %2d: (%s) D=%i",CamA,CamQ?"Q":" ",(unsigned int)CamD);
	  	//  fprintf(OutF,", CH %2d: (%s) D=%i",CamA,CamQ?"Q":" ",(unsigned int)CamD);
        printf(" %i",(unsigned int)CamD);
	fprintf(OutF," %i",(unsigned int)CamD);

	}
	//
	//CAMAC_I(udev,false);
      }
	printf("\n");
	fprintf(OutF,"\n");
      //      sleep(1);
    }
    else
      {
	printf("No LAM:reset module and wait\n");  
	CamF=9;
	ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
	if (ret<0 || CamX==0) printf("Clear fallito\n");
      sleep(1);
      } 
      sleep(1);//<<<---
    }
  // Close Output File
  fclose(OutF);
  // Close the Device
  xxusb_device_close(udev);
  printf("\n");
  
  return 0;
}

