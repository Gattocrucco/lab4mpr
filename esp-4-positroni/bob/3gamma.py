import matplotlib.pyplot as plt
import numpy as np
import lab4
import lab
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#___________________________________________
scala="zoom" #auto o zoom
markersize = 3
zoom_min_ch1 = 0
zoom_max_ch1 = 580
zoom_min_ch2 = 0
zoom_max_ch2 = 580
zoom_min_ch3 = 0
zoom_max_ch3 = 580

#____________________________________________

signal1 = "../DAQ/0522_3gamma.txt"
signal2 = "../DAQ/0523_3gamma_rumore.txt"
noise = "../DAQ/0524_3gamma_rumore_pomeriggio.txt"
calibration = "../DAQ/0522_3gamma_cal.txt"

file = signal1

if (file == noise):
    file_name = "noise"
if (file == signal1):
    file_name ="signal1"
if (file == signal2):
    file_name ="signal2"
    
ch1, ch2, ch3, ts = lab4.loadtxt(file, unpack=True, usecols=(0,5,11,12))

#_____________________________________________________________
#fit calibrazione

ch1_cal, ch2_cal, ch3_cal, ts = lab4.loadtxt(calibration, unpack=True, usecols=(0,5,11,12))

def gauss(x,N1,u,sigma,N2,l):
    return N1 * np.e**(-1*((x-u)**2)/(2*sigma**2)) + N2 * np.e**(l * x)


ch1_beta_cut=[420,460]
ch1_ne_cut=[1000,1200]
ch2_beta_cut=[400,600]
ch2_ne_cut=[800,1000]
ch3_beta_cut=[400,600]
ch3_ne_cut=[800,1000]


hist = hist(ch1_cal, bins = arange(ch1_beta_cut[0],ch1_beta_cut[1]))
X_b1 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch1_beta_cut[0]+ch1_beta_cut[1])/2,(ch1_beta_cut[0]-ch1_beta_cut[1])/2,0,0]
outb1 = lab.fit_curve(gauss,X_b1,Y,dy=dY,p0=val, method="odrpack")

hist = hist(ch1_cal, bins = arange(ch1_ne_cut[0],ch1_ne_cut[1]))
X_n1 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch1_ne_cut[0]+ch1_ne_cut[1])/2,(ch1_ne_cut[0]-ch1_ne_cut[1])/2,0,0]
outn1 = lab.fit_curve(gauss,X_n1,Y,dy=dY,p0=val, method="odrpack")

hist = hist(ch1_cal, bins = arange(ch2_beta_cut[0],ch2_beta_cut[1]))
X_b2 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch2_beta_cut[0]+ch2_beta_cut[1])/2,(ch2_beta_cut[0]-ch2_beta_cut[1])/2,0,0]
outb2 = lab.fit_curve(gauss,X_b2,Y,dy=dY,p0=val, method="odrpack")

hist = hist(ch1_cal, bins = arange(ch2_ne_cut[0],ch2_ne_cut[1]))
X_n2 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch2_ne_cut[0]+ch2_ne_cut[1])/2,(ch2_ne_cut[0]-ch2_ne_cut[1])/2,0,0]
outn2 = lab.fit_curve(gauss,X_b1,Y,dy=dY,p0=val, method="odrpack")

hist = hist(ch3_cal, bins = arange(ch3_beta_cut[0],ch3_beta_cut[1]))
X_b3 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch3_beta_cut[0]+ch3_beta_cut[1])/2,(ch3_beta_cut[0]-ch3_beta_cut[1])/2,0,0]
outb3 = lab.fit_curve(gauss,X_b3,Y,dy=dY,p0=val, method="odrpack")

hist = hist(ch1_cal, bins = arange(ch3_ne_cut[0],ch3_ne_cut[1]))
X_n3 = (hist[1]+0.5)[:-1]
Y = hist[0]
dY = sqrt(Y)
val=[1e5,(ch3_ne_cut[0]+ch3_ne_cut[1])/2,(ch3_ne_cut[0]-ch3_ne_cut[1])/2,0,0]
outn3 = lab.fit_curve(gauss,X_n3,Y,dy=dY,p0=val, method="odrpack")

bins_cal = arange(0,1200)
figure('Calibration').set_tight_layout(True)
clf()
title("Acquisizioni singole signal")
hist(ch1_cal, bins = bins_cal, label="ch1 n=%d"%(len(ch1_cal)),histtype="step")
plot(Xb1,gauss(Xb1,*outb1.par))
plot(Xn1,gauss(Xn1,*outn1.par))
hist(ch2_cal, bins = bins_cal, label="ch2 n=%d"%(len(ch2_cal)),histtype="step")
plot(Xb2,gauss(Xb2,*outb2.par))
plot(Xn2,gauss(Xn2,*outn2.par))
hist(ch3_cal, bins = bins_cal, label="ch3 n=%d"%(len(ch3_cal)),histtype="step")
plot(Xb3,gauss(Xb3,*outb3.par))
plot(Xn3,gauss(Xn3,*outn3.par))

legend(loc=0,fontsize='small')
yscale('log')
show()

#valori noti in keV
ne = 1274
beta = 511

nech1 = 1045
betach1 = 520
nech2 = 870
betach2 = 442
nech3 = 990
betach3 = 535

#nech1 = 1090
#betach1 = 530
#nech2 = 900
#betach2 = 455
#nech3 = 995
#betach3 = 540

mch1 = (nech1-betach1)/(ne-beta)
qch1 = (ne*betach1-beta*nach1)/(ne-beta)
mch2 = (nech2-betach2)/(ne-beta)
qch2 = (ne*betach2-beta*nech2)/(ne-beta)
#mch3 = (nech3-betach3)/(na-beta)
#qch3 = (ne*betach3-beta*nech3)/(ne-beta)

#parabola canale 3
cch3 = (nech3 - (ne*betach3)/beta)/(ne**2-beta*ne)
bch3 = betach3/beta - cch3 * beta

ch1 = (ch1-qch1)/mch1
ch2 = (ch2-qch2)/mch2
#ch3 = (ch3-qch3)/mch3
ch3 = (-b + sqrt(b**2 + 4*c*ch3))/(2*c)

bin_dim = 6
bins_ch1=(arange(0.5,600.5)[0::bin_dim]-qch1)/mch1
bins_ch2=(arange(0.5,600.5)[0::bin_dim]-qch2)/mch2
bins_ch3=(-b + sqrt(b**2 + 4*c*(arange(0.5,600.5)[0::bin_dim])))/(2*c)

#_____________________________________________________________
#calcolo box

def Energy (ang1, ang2):
    ch3en = 2*beta / (sin(ang2)/sin(ang1)+cos(ang2)+1+cos(ang1)*sin(ang2)/sin(ang1))
    ch2en =  ch3en * sin(ang2)/sin(ang1)
    ch1en = 2*beta - ch2en -ch3en
    return [ch1en,ch2en,ch3en]

ang1 = 59 /180 * np.pi
ang2 = 43 /180 * np.pi
d1 = 50
d2 = 50
d3 = 50

#ch1_en=438
#ch2_en=313
#ch3_en=393
#ch1_en=378
#ch2_en=253
#ch3_en=343
#ch1_en = (ch1_en-qch1)/mch1
#ch2_en = (ch2_en-qch2)/mch2
##ch3_en = (ch3_en-qch3)/mch3
#ch3_en = (-b + sqrt(b**2+4*c*ch3_en))/(2*c)
#ch1_lim = [ch1_en-d1,ch1_en+d1]
#ch2_lim = [ch2_en-d2,ch2_en+d2]
#ch3_lim = [ch3_en-d3,ch3_en+d3]

ch1_en = Energy(ang1,ang2)[0]
ch2_en = Energy(ang1,ang2)[1]
ch3_en = Energy(ang1,ang2)[2]  
ch1_lim = [ch1_en1-d1,ch1_en1+d1]
ch2_lim = [ch2_en1-d2,ch2_en1+d2]
ch3_lim = [ch3_en1-d3,ch3_en1+d3]  

box_ch1 = [ch1_lim[0],ch1_lim[0],ch1_lim[1],ch1_lim[1],ch1_lim[0]]
box_ch2 = [ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0]]
box_ch2_ = [ch2_lim[0],ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0]]
box_ch3 = [ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[0]]

#_____________________________________________________________
#trigger vari

trs12 = (ch3>ch3_lim[0]) & (ch3 <ch3_lim[1])
trs13 = (ch2>ch2_lim[0]) & (ch2 <ch2_lim[1])
trs23 = (ch1>ch1_lim[0]) & (ch1 <ch1_lim[1])

out1 = ch1
out2 = ch2
out3 = ch3

out1_12 = ch1[trs12]
out2_12 = ch2[trs12]
out1_13 = ch1[trs13]
out3_13 = ch3[trs13]
out2_23 = ch2[trs23]
out3_23 = ch3[trs23]

#___________________________________________
    
trs_s= trs12 & trs13 & trs23

print("ch1: %d, ch2: %d, ch3: %d  # nel box = %d"%(ch1_lim[0]+d,ch2_lim[0]+d,ch3_lim[0]+d,sum(trs_s)))
  
#________________________________________________________
#SCATTER PLOT 2D

#figure('Singoli ch signal').set_tight_layout(True)
#clf()
#
#title("Acquisizioni singole signal")
#hist(out1s,bins=tutti,label="ch1 n=%d"%(len(out1s)),histtype="step")
#hist(out2s,bins=tutti,label="ch2 n=%d"%(len(out2s)),histtype="step")
#hist(out3s,bins=tutti,label="ch3 n=%d"%(len(out3s)),histtype="step")
#legend(loc=0,fontsize='small')
#yscale('log')

figure('sc2 signal ch1/ch2/ch3: %d/%d/%d'%(ch1_lim[0]+d,ch2_lim[0]+d,ch3_lim[0]+d),figsize=(10, 16)).set_tight_layout(True)
clf()
      
subplot(321)
title("Scatter plot ch1/ch2")
_,_,_,im=plt.hist2d(out1,out2,bins=bins_ch1,norm=LogNorm(),cmap='jet')
colorbar(im)
plot(box_ch1,box_ch2,color="red")
xlabel("ch1 [keV]")
ylabel("ch2 [keV]")
if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch2,zoom_max_ch2)    
    
subplot(323)
title("Scatter plot ch1/ch3")
_,_,_,im=plt.hist2d(out1,out3,bins=bins_ch2,norm=LogNorm(),cmap='jet')
colorbar(im)
plot(box_ch1,box_ch3,color="red")
xlabel("ch1 [keV]")
ylabel("ch3 [keV]")
if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch3,zoom_max_ch3)

subplot(325)
title("Scatter plot ch2/ch3")
_,_,_,im=plt.hist2d(out2,out3,bins=bins_ch3,norm=LogNorm(),cmap='jet')
colorbar(im)
plot(box_ch2_,box_ch3,color="red")
xlabel("ch2 [keV]")
ylabel("ch3 [keV]")
if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch2,zoom_max_ch2)
    ylim(zoom_min_ch3,zoom_max_ch3)  


    


subplot(322)
title("Scatter plot ch1/ch2 con ch3 in (%d,%d)[keV]"%(ch3_lim[0],ch3_lim[1]))
_,_,_,im=plt.hist2d(out1_12,out2_12,bins=bins_ch1,norm=LogNorm(),cmap='jet')
colorbar(im)

plot(box_ch1,box_ch2,color="red")

xlabel("ch1 [keV]")
ylabel("ch2 [keV]")

if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch2,zoom_max_ch2)
 
subplot(324)
title("Scatter plot ch1/ch3 con ch2 in (%d,%d)[keV]"%(ch2_lim[0],ch2_lim[1]))
_,_,_,im=plt.hist2d(out1_13,out3_13,bins=bins_ch2,norm=LogNorm(),cmap='jet')
colorbar(im)

plot(box_ch1,box_ch3,color="red")

xlabel("ch1 [keV]")
ylabel("ch3 [keV]")
if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch3,zoom_max_ch3)
    
subplot(326)
title("Scatter plot  ch2/ch3 con ch1 in (%d,%d)[keV]"%(ch1_lim[0],ch1_lim[1]))
_,_,_,im=plt.hist2d(out2_23,out3_23,bins=bins_ch3,norm=LogNorm(),cmap='jet')
colorbar(im)

plot(box_ch2_,box_ch3,color="red")

xlabel("ch2 [keV]")
ylabel("ch3 [keV]")
if(scala=="auto"):
    xlim(-20,1400)
    ylim(-20,1400)
else:
    xlim(zoom_min_ch2,zoom_max_ch2)
    ylim(zoom_min_ch3,zoom_max_ch3)
    
savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.pdf'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)
savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.png'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)

#show()

