import matplotlib.pyplot as plt
import numpy as np
import lab4
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#___________________________________________

#limiti del boc in keV
ch1inf = 250
ch1sup = 290
ch2inf = 250
ch2sup = 290
ch3inf = 430
ch3sup = 500

#digit fittati
nach1 = 890
betach1 = 400
nach2 = 890
betach2 = 385
nach3 = 750
betach3 = 285

#valori noti in keV
na = 1274
beta = 511

scala="zoom" #auto o zoom
markersize = 5
zooom_min_ch1 = 100
zoom_max_ch1 = 500
zoom_min_ch2 = 100
zoom_max_ch2 = 500
zoom_min_ch3 = 100
zoom_max_ch3 = 500

mch1 = (nach1-betach1)/(na-beta)
qch1 = (na*betach1-beta*nach1)/(na-beta)
mch2 = (nach2-betach2)/(na-beta)
qch2 = (na*betach2-beta*nach2)/(na-beta)
mch3 = (nach3-betach3)/(na-beta)
qch3 = (na*betach3-beta*nach3)/(na-beta)

Ech1 = array([ch1inf,ch1sup])
Ech2 = array([ch2inf,ch2sup])
Ech3 = array([ch3inf,ch3sup])
ch1_lim = Ech1*mch1+qch1
ch2_lim = Ech2*mch2+qch2
ch3_lim = Ech3*mch3+qch3

#ch3_lim = [230,280]

box_ch1 = [ch1_lim[0],ch1_lim[0],ch1_lim[1],ch1_lim[1],ch1_lim[0]]
box_ch2 = [ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0]]
box_ch2_ = [ch2_lim[0],ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0]]
box_ch3 = [ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[0]]

box3D_ch1 = [ch1_lim[0],ch1_lim[0],ch1_lim[0],ch1_lim[0],ch1_lim[0],
             ch1_lim[1],ch1_lim[1],ch1_lim[1],ch1_lim[1],ch1_lim[1],ch1_lim[1],
             ch1_lim[0], ch1_lim[1],ch1_lim[1],ch1_lim[0],ch1_lim[1],ch1_lim[1],ch1_lim[0]]

box3D_ch2 = [ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0],
             ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0],ch2_lim[0],
             ch2_lim[0], ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[1],ch2_lim[1],ch2_lim[1]]

box3D_ch3 = [ch3_lim[0],ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],
             ch3_lim[0],ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[1],
             ch3_lim[1], ch3_lim[1],ch3_lim[1],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[0]]

#____________________________________________

signal = "../DAQ/0508_3gamma.txt"
noise = "../DAQ/0511_3gamma_rumore.txt"

ch1s, ch2s, ch3s, tr1s, tr2s, tr3s, c2s, c3s = lab4.loadtxt(signal, unpack=True, usecols=(0,1,2,4,5,6,8,9))
ch1n, ch2n, ch3n, tr1n, tr2n, tr3n, c2n, c3n = lab4.loadtxt(noise, unpack=True, usecols=(0,1,2,4,5,6,8,9))

#____________________________________________
trs = (tr3s>500)
trn = (tr3n>500)
#trn = (tr3n>500) & (tr1n>500) & (tr2n > 500)

#___________________________________________

trs12s = (tr1s>500) & (tr2s>500)
trs13s = (tr1s>500) & (tr3s>500)
trs23s = (tr2s>500) & (tr3s>500)

trn12s = (tr1n>500) & (tr2n>500)
trn13s = (tr1n>500) & (tr3n>500)
trn23s = (tr2n>500) & (tr3n>500)

trs12 = (ch3s>ch3_lim[0]) & (ch3s <ch3_lim[1]) & trs
trs12_ = ~((ch3s>ch3_lim[0]) & (ch3s <ch3_lim[1])) & trs
trs13 = (ch2s>ch2_lim[0]) & (ch2s <ch2_lim[1]) & trs
trs13_ = ~((ch2s>ch2_lim[0]) & (ch2s <ch2_lim[1])) & trs
trs23 = (ch1s>ch1_lim[0]) & (ch1s <ch1_lim[1]) & trs
trs23_ = ~((ch1s>ch1_lim[0]) & (ch1s <ch1_lim[1])) & trs

trn12 = (ch3n>ch3_lim[0]) & (ch3n <ch3_lim[1]) & trn
trn12_ = ~((ch3n>ch3_lim[0]) & (ch3n <ch3_lim[1])) & trn
trn13 = (ch2n>ch2_lim[0]) & (ch2n <ch2_lim[1]) & trn
trn13_ = ~((ch2n>ch2_lim[0]) & (ch2n <ch2_lim[1])) & trn
trn23 = (ch1n>ch1_lim[0]) & (ch1n <ch1_lim[1]) & trn
trn23_ = ~((ch1n>ch1_lim[0]) & (ch1n <ch1_lim[1])) & trn

#___________________________________________
out1s = ch1s
out2s = ch2s
out3s = ch3s

out1s_12 = ch1s[trs12]
out2s_12 = ch2s[trs12]
out1s_12_ = ch1s[trs12_]
out2s_12_ = ch2s[trs12_]

out1s_13 = ch1s[trs13]
out3s_13 = ch3s[trs13]
out1s_13_ = ch1s[trs13_]
out3s_13_ = ch3s[trs13_]

out2s_23 = ch2s[trs23]
out3s_23 = ch3s[trs23]
out2s_23_ = ch2s[trs23_]
out3s_23_ = ch3s[trs23_]

#________________________________________

out1n = ch1n
out2n = ch2n
out3n = ch3n

out1n_12 = ch1n[trn12]
out2n_12 = ch2n[trn12]
out1n_12_ = ch1n[trn12_]
out2n_12_ = ch2n[trn12_]

out1n_13 = ch1n[trn13]
out3n_13 = ch3n[trn13]
out1n_13_ = ch1n[trn13_]
out3n_13_ = ch3n[trn13_]

out2n_23 = ch2n[trn23]
out3n_23 = ch3n[trn23]
out2n_23_ = ch2n[trn23_]
out3n_23_ = ch3n[trn23_]

#___________________________________________

trs_s= trs12 & trs13 & trs23
trn_s= trn12 & trn13 & trn23

#____________________________________________

out1s3D = ch1s[~trs_s & trs]
out2s3D = ch2s[~trs_s & trs]
out3s3D = ch3s[~trs_s & trs]

out1n3D = ch1n[~trn_s & trn]
out2n3D = ch2n[~trn_s & trn]
out3n3D = ch3n[~trn_s & trn]

out1s3Ds = ch1s[trs_s]
out2s3Ds = ch2s[trs_s]
out3s3Ds = ch3s[trs_s]

out1n3Ds = ch1n[trn_s]
out2n3Ds = ch2n[trn_s]
out3n3Ds = ch3n[trn_s]

#______________________________________________________
#SCATTER PLOT 3D

fig = plt.figure('Scatter 3D')
fig.clf()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(out1s3D, out2s3D, out3s3D, color='red', marker='o')
ax.scatter(out1s3Ds, out2s3Ds, out3s3Ds, color='black', marker='o')
ax.scatter(out1n3D, out2n3D, out3n3D, color='blue', marker='x')
ax.scatter(out1s3Ds, out2s3Ds, out3s3Ds, color='gray', marker='x')
ax.plot(box3D_ch1,box3D_ch2,box3D_ch3)

ax.set_xlabel('ch1')
ax.set_ylabel('ch2')
ax.set_zlabel('ch3')

if(scala!="auto"):
    ax.set_xlim(ch1_lim[0]-50, ch1_lim[1]+100)
    ax.set_ylim(ch2_lim[0]-50, ch2_lim[1]+100)
    ax.set_zlim(ch3_lim[0]-50, ch3_lim[1]+100)

fig.show()

#________________________________________________________
#SCATTER PLOT 2D

tutti=arange(0,1200//8)*8

#figure('Singoli ch signal').set_tight_layout(True)
#clf()
#
#title("Acquisizioni singole signal")
#hist(out1s,bins=tutti,label="ch1 n=%d"%(len(out1s)),histtype="step")
#hist(out2s,bins=tutti,label="ch2 n=%d"%(len(out2s)),histtype="step")
#hist(out3s,bins=tutti,label="ch3 n=%d"%(len(out3s)),histtype="step")
#legend(loc=0,fontsize='small')
#yscale('log')


figure('sc2 signal').set_tight_layout(True)
clf()

subplot(131)

title("Scatter plot ch1/ch2 signal")
_,_,_,im=plt.hist2d(out1s[trs12s],out2s[trs12s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
plot(out1s_12_,out2s_12_, linestyle="", marker="." ,color="white", label="c3")
plot(out1s_12,out2s_12, linestyle="", marker="x", color="black", markersize=markersize, label="ch3 signal")
plot(box_ch1,box_ch2,color="black")
xlabel("ch1")
ylabel("ch2")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out2s)-5,max(out2s)+10)
else:
    xlim(zooom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch2,zoom_max_ch2)
legend(loc=0)


#figure('sc2 ch1/ch3 signal').set_tight_layout(True)
#clf()
subplot(132)
title("Scatter plot ch1/ch3 signal")
_,_,_,im=plt.hist2d(out1s[trs13s],out3s[trs13s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
#plot(out1s_13_,out3s_13_, linestyle="", marker="." ,color="white", label="c3")
plot(out1s_13,out3s_13, linestyle="", marker="x" ,color="white", markersize=markersize, label="ch2 signal")

plot(box_ch1,box_ch3,color="black")
xlabel("ch1")
ylabel("ch3")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out3s)-5,max(out3s)+10)
else:
    xlim(zooom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch3,zoom_max_ch3)
legend(loc=0)


#figure('sc2 ch2/ch3 signal').set_tight_layout(True)
#clf()

subplot(133)
title("Scatter plot  ch2/ch3 signal")
_,_,_,im=plt.hist2d(out2s[trs23s],out3s[trs23s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
#plot(out2s_23_,out3s_23_, linestyle="", marker="." ,color="white", label="c3")
plot(out2s_23,out3s_23, linestyle="", marker="x" ,color="white", markersize=markersize, label="ch1 signal")
plot(box_ch2_,box_ch3,color="black")
xlabel("ch2")
ylabel("ch3")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out3s)-5,max(out3s)+10)
else:
    xlim(zooom_min_ch2,zoom_max_ch2)
    ylim(zoom_min_ch3,zoom_max_ch3)

legend(loc=0)
show()

#________________________________________________

#figure('Singoli ch noise').set_tight_layout(True)
#clf()
#
#title("Acquisizioni singole noise")
#hist(out1n,bins=tutti,label="ch1 n=%d"%(len(out1n)),histtype="step")
#hist(out2n,bins=tutti,label="ch2 n=%d"%(len(out2n)),histtype="step")
#hist(out3n,bins=tutti,label="ch3 n=%d"%(len(out3n)),histtype="step")
#legend(loc=0,fontsize='small')
#yscale('log')


figure('sc2 noise').set_tight_layout(True)
clf()
subplot(131)
title("Scatter plot ch1/ch2 noise")
_,_,_,im=plt.hist2d(out1n[trn12s],out2n[trn12s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
#plot(out1n_12_,out2n_12_, linestyle="", marker="." ,color="white", label="c3")
plot(out1n_12,out2n_12, linestyle="", marker="x" ,color="black", markersize=markersize, label="ch3 signal")
plot(box_ch1,box_ch2,color="black")
xlabel("ch1")
ylabel("ch2")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out3s)-5,max(out3s)+10)
else:
    xlim(zooom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch2,zoom_max_ch2)
legend(loc=0)

#figure('sc2 ch1/ch3 noise').set_tight_layout(True)
#clf()

subplot(132)
title("Scatter plot ch1/ch3 noise")
_,_,_,im=plt.hist2d(out1n[trn13s],out3n[trn13s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
#plot(out1n_13_,out3n_13_, linestyle="", marker="." ,color="white", label="c3")
plot(out1n_13,out3n_13, linestyle="", marker="x" ,color="black", markersize=markersize, label="ch2 signal")
plot(box_ch1,box_ch3,color="black")
xlabel("ch1")
ylabel("ch3")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out3s)-5,max(out3s)+10)
else:
    xlim(zooom_min_ch1,zoom_max_ch1)
    ylim(zoom_min_ch3,zoom_max_ch3)
legend(loc=0)

#figure('sc2 ch2/ch3 noise').set_tight_layout(True)
#clf()

subplot(133)
title("Scatter plot  ch2/ch3 noise")
_,_,_,im=plt.hist2d(out2n[trn23s],out3n[trn23s],bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
#plot(out2n_23_,out3n_23_, linestyle="", marker="." ,color="white", label="c3")
plot(out2n_23,out3n_23, linestyle="", marker="x" ,color="black", markersize=markersize, label="ch1 signal")
plot(box_ch2_,box_ch3,color="black")
xlabel("ch2")
ylabel("ch3")
if(scala=="auto"):
    xlim(min(out1s)-5,max(out1s)+5)
    ylim(min(out3s)-5,max(out3s)+10)
else:
    xlim(zooom_min_ch2,zoom_max_ch2)
    ylim(zoom_min_ch3,zoom_max_ch3)

legend(loc=0)
show()