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
  char nafinx[20]; // deve essere lo slot (6)
  
  int ret, i, nDevices;
  xxusb_device_type devices[100]; 
  struct usb_device *dev;In [150]: run c2 0510_effi_bbn.txt


  usb_dev_handle *udev;       // Device Handle 
  //  int inhibit;
  time_t current_time;
  char* c_time_string;
  FILE *OutF;
  char OutputFileName[] = "TDC-PETData.txt";
  int NEvt=0;

  // Open the output file
  OutF = fopen(OutputFileName,"w");
  
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
In [150]: run c2 0510_effi_bbn.txt


  CamN=0;
  while (CamN==0)  {
    printf("TDC slot? ");
    fflush(stdin);
    scanf("%s",nafinx);// aggiungere il 6
    sscanf(nafinx,"%i",&CamN);
    fflush(stdin);
    if (CamN<1 || CamN>24) CamN=0;
  }
  CamA=1;
  CamF=26;
  ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
  printf("Enabled LAM at Add 1\n");
  if (ret<0 || CamX==0) {
    printf("LAM enable fallito\n");
    return 1;
  }

  //  while(1) {   scommentare
  while(NEvt < 100) {
    CamA=0;
    CamF=8;
    printf("Test LAM at Add 0\n");    
    ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
    printf("Test LAM at Add 0: ret %i, CamQ %lx \n",ret,(long unsigned int)CamQ);    
    if (ret<0 || CamX==0) printf("Lettura LAM fallita\n");
    //else if (CamQ==1) {
    else {
      NEvt++;
      printf("Event: %i \n",NEvt);
      fprintf(OutF,"Event: %i \n",NEvt);
      CAMAC_I(udev,true);
      current_time = time(NULL);
      c_time_string = ctime(&current_time);
      printf("%s\n",c_time_string);
      // Read four TDC channels 
      for (i=0;i<4;i++) {
	CamA=i;
	CamF=0;
	ret=CAMAC_read(udev,CamN,CamA,CamF,&CamD,&CamQ,&CamX);
	if (ret<0 || CamX==0) printf("Lettura fallita\n");
	else { 
	  printf(", CH %2d: (%s) D=%i",CamA,CamQ?"Q":" ",(unsigned int)CamD);   
	  fprintf(OutF,", CH %2d: (%s) D=%i",CamA,CamQ?"Q":" ",(unsigned int)CamD);
	}
	printf("\n");
	fprintf(OutF,"\n");
      }
      CAMAC_C(udev);
      CAMAC_I(udev,false);
      usleep(1000);
      }
  }
  // Close the Device
  xxusb_device_close(udev);
  printf("\n");
  fprintf(OutF,"\n");
  //Close the file
  fclose(OutF);
  
  return 0;
}

