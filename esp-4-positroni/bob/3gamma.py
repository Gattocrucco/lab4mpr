import matplotlib.pyplot as plt
import numpy as np
import lab4
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#___________________________________________

#valori noti in keV
na = 1274
beta = 511

def Energy (ang1, ang2):
    ch3en = 2*beta / (2*sin(ang2)/sin(ang1)+cos(ang2)+1)
    ch2en =  ch3en * sin(ang2)/sin(ang1)
    ch1en = 2*beta - ch2en -ch3en
    return [ch1en,ch2en,ch3en]

#limiti del boc in keV
delta = 10 / 180 * np.pi
ang1 = 85 /180 * np.pi
ang2 = 65 /180 * np.pi

ch1inf = Energy(ang1+delta,ang2+delta)[0]-10
ch1sup = Energy(ang1-delta,ang2-delta)[0]+10
ch2inf = Energy(ang1+delta,ang2-delta)[1]-10
ch2sup = Energy(ang1-delta,ang2+delta)[1]+10
ch3inf = Energy(ang1-delta,ang2+delta)[2]-10
ch3sup = Energy(ang1+delta,ang2-delta)[2]+10


#digit fittati
nach1 = 1045
betach1 = 520
nach2 = 870
betach2 = 442
nach3 = 990
betach3 = 535

scala="zoom" #auto o zoom
markersize = 3
zoom_min_ch1 = 0
zoom_max_ch1 = 560
zoom_min_ch2 = 0
zoom_max_ch2 = 560
zoom_min_ch3 = 0
zoom_max_ch3 = 560

mch1 = (nach1-betach1)/(na-beta)
qch1 = (na*betach1-beta*nach1)/(na-beta)
mch2 = (nach2-betach2)/(na-beta)
qch2 = (na*betach2-beta*nach2)/(na-beta)
mch3 = (nach3-betach3)/(na-beta)
qch3 = (na*betach3-beta*nach3)/(na-beta)

Ech1 = array([ch1inf,ch1sup])
Ech2 = array([ch2inf,ch2sup])
Ech3 = array([ch3inf,ch3sup])
ch1_lim1 = Ech1*mch1+qch1
ch2_lim1 = Ech2*mch2+qch2
ch3_lim1 = Ech3*mch3+qch3
#ch1_lim = Ech1*mch1+qch1
#ch2_lim = Ech2*mch2+qch2
#ch3_lim = Ech3*mch3+qch3
ch1_lim = [420,460]
ch2_lim = [285,330]
ch3_lim = [380,435]


box_ch1 = [ch1_lim[0],ch1_lim[0],ch1_lim[1],ch1_lim[1],ch1_lim[0]]
box_ch2 = [ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0]]
box_ch2_ = [ch2_lim[0],ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0]]
box_ch3 = [ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[0]]

box1_ch1 = [ch1_lim1[0],ch1_lim1[0],ch1_lim1[1],ch1_lim1[1],ch1_lim1[0]]
box1_ch2 = [ch2_lim1[0],ch2_lim1[1],ch2_lim1[1],ch2_lim1[0],ch2_lim1[0]]
box1_ch2_ = [ch2_lim1[0],ch2_lim1[0],ch2_lim1[1],ch2_lim1[1],ch2_lim1[0]]
box1_ch3 = [ch3_lim1[0],ch3_lim1[1],ch3_lim1[1],ch3_lim1[0],ch3_lim1[0]]

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

signal = "../DAQ/0522_3gamma.txt"
noise = "../DAQ/0523_3gamma_rumore.txt"

ch1s, ch2s, ch3s = lab4.loadtxt(signal, unpack=True, usecols=(0,5,11))
ch1n, ch2n, ch3n = lab4.loadtxt(noise, unpack=True, usecols=(0,5,11))

ch1s = (ch1s-qch1)/mch1
ch2s = (ch2s-qch2)/mch2
ch3s = (ch3s-qch3)/mch3
ch1n = (ch1n-qch1)/mch1
ch2n = (ch2n-qch2)/mch2
ch3n = (ch3n-qch3)/mch3

#____________________________________________
trs = (ch3s>0)
trn = (ch3n>0)
#trn = (tr3n>500) & (tr1n>500) & (tr2n > 500)

#___________________________________________

trs12s = (ch1s>0) & (ch2s>0)
trs13s = (ch1s>0) & (ch3s>0)
trs23s = (ch2s>0) & (ch3s>0)

trn12s = (ch1n>0) & (ch2n>0)
trn13s = (ch1n>0) & (ch3n>0)
trn23s = (ch2n>0) & (ch3n>0)

ch1_lim = [420,460]
ch2_lim = [285,330]
ch3_lim = [380,435]

#for n1 in arange(438,439,1):
#    for n2 in arange(313,314,1):
#        for n3 in arange(393,394,1):
for n1 in arange(378,439,200):
    for n2 in arange(253,254,200):
        for n3 in arange(343,394,200):
            d1 = 25
            d2 = 25
            d3 = 35
            ch3_lim[0]=n3 -d3
            ch3_lim[1]=n3 +d3
            ch2_lim[0]=n2 -d2
            ch2_lim[1]=n2 +d2
            ch1_lim[0]=n1 -d1
            ch1_lim[1]=n1 +d1
            ch1_lim = (array(ch1_lim)-qch1)/mch1
            ch2_lim = (array(ch2_lim)-qch2)/mch2
            ch3_lim = (array(ch3_lim)-qch3)/mch3
            
            
            box_ch1 = [ch1_lim[0],ch1_lim[0],ch1_lim[1],ch1_lim[1],ch1_lim[0]]
            box_ch2 = [ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0],ch2_lim[0]]
            box_ch2_ = [ch2_lim[0],ch2_lim[0],ch2_lim[1],ch2_lim[1],ch2_lim[0]]
            box_ch3 = [ch3_lim[0],ch3_lim[1],ch3_lim[1],ch3_lim[0],ch3_lim[0]]
            
            trs12 = (ch3s>ch3_lim[0]) & (ch3s <ch3_lim[1]) & trs
            trs12_ = ~((ch3s>ch3_lim[0]) & (ch3s <ch3_lim[1])) & trs
            trs13 = (ch2s>ch2_lim[0]) & (ch2s <ch2_lim[1]) & trs
            trs13_ = ~((ch2s>ch2_lim[0]) & (ch2s <ch2_lim[1])) & trs
            trs23 = (ch1s>ch1_lim[0]) & (ch1s <ch1_lim[1]) & trs
            trs23_ = ~((ch1s>ch1_lim[0]) & (ch1s <ch1_lim[1])) & trs
        
            '''
            trn12 = (ch3n>ch3_lim[0]) & (ch3n <ch3_lim[1]) & trn
            trn12_ = ~((ch3n>ch3_lim[0]) & (ch3n <ch3_lim[1])) & trn
            trn13 = (ch2n>ch2_lim[0]) & (ch2n <ch2_lim[1]) & trn
            trn13_ = ~((ch2n>ch2_lim[0]) & (ch2n <ch2_lim[1])) & trn
            trn23 = (ch1n>ch1_lim[0]) & (ch1n <ch1_lim[1]) & trn
            trn23_ = ~((ch1n>ch1_lim[0]) & (ch1n <ch1_lim[1])) & trn
            '''
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
            '''
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
            '''
            #___________________________________________
                
            trs_s= trs12 & trs13 & trs23
            #trn_s= trn12 & trn13 & trn23
            
            print("ch1: %d, ch2: %d, ch3: %d  # nel box = %d"%(ch1_lim[0]+d,ch2_lim[0]+d,ch3_lim[0]+d,sum(trs_s)))
              
            #____________________________________________
            '''
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
            '''
            #______________________________________________________
            #SCATTER PLOT 3D
            
            #fig = plt.figure('Scatter 3D')
            #fig.clf()
            #ax = fig.add_subplot(111, projection='3d')
            #
            #ax.scatter(out1s3D, out2s3D, out3s3D, color='red', marker='o')
            #ax.scatter(out1s3Ds, out2s3Ds, out3s3Ds, color='black', marker='o')
            #ax.scatter(out1n3D, out2n3D, out3n3D, color='blue', marker='x')
            #ax.scatter(out1s3Ds, out2s3Ds, out3s3Ds, color='gray', marker='x')
            #ax.plot(box3D_ch1,box3D_ch2,box3D_ch3)
            #
            #ax.set_xlabel('ch1')
            #ax.set_ylabel('ch2')
            #ax.set_zlabel('ch3')
            #
            #if(scala!="auto"):
            #    ax.set_xlim(ch1_lim[0]-50, ch1_lim[1]+100)
            #    ax.set_ylim(ch2_lim[0]-50, ch2_lim[1]+100)
            #    ax.set_zlim(ch3_lim[0]-50, ch3_lim[1]+100)
            #
            #fig.show()
            
            #________________________________________________________
            #SCATTER PLOT 2D
            
            tutti=arange(0,1400//6)*6
            
            #figure('Singoli ch signal').set_tight_layout(True)
            #clf()
            #
            #title("Acquisizioni singole signal")
            #hist(out1s,bins=tutti,label="ch1 n=%d"%(len(out1s)),histtype="step")
            #hist(out2s,bins=tutti,label="ch2 n=%d"%(len(out2s)),histtype="step")
            #hist(out3s,bins=tutti,label="ch3 n=%d"%(len(out3s)),histtype="step")
            #legend(loc=0,fontsize='small')
            #yscale('log')
            
            
            figure('sc2 ch1/ch2 signal ch1/ch2/ch3: %d/%d/%d'%(ch1_lim[0]+d,ch2_lim[0]+d,ch3_lim[0]+d),figsize=(18, 6)).set_tight_layout(True)
            clf()
            
        #    subplot(121)
        #    
        #    title("Scatter plot ch1/ch2")
        #    _,_,_,im=plt.hist2d(out1s[trs12s],out2s[trs12s],bins=tutti,norm=LogNorm(),cmap='jet')
        #    colorbar(im)
        #    plot(box_ch1,box_ch2,color="black")
        #    #plot(out1s_12_,out2s_12_, linestyle="", marker="." ,color="white", label="c3")
        #    if(scala=="auto"):
        #        xlim(min(out1s)-5,max(out1s)+5)
        #        ylim(min(out2s)-5,max(out2s)+10)
        #    else:
        #        xlim(zoom_min_ch1,zoom_max_ch1)
        #        ylim(zoom_min_ch2,zoom_max_ch2)
        #    
            subplot(131)
            
            title("Scatter plot ch1/ch2 con ch3 in (%d,%d)[keV]"%(ch3_lim[0],ch3_lim[1]))
            _,_,_,im=plt.hist2d(out1s_12,out2s_12,bins=tutti,norm=LogNorm(),cmap='jet')
            colorbar(im)
            
            #plot(out1s_12,out2s_12, linestyle="", marker="x", color="black", markersize=markersize, label="ch3 signal")
            plot(box_ch1,box_ch2,color="red")
            #plot(box1_ch1,box1_ch2,color="red")
            xlabel("ch1 [keV]")
            ylabel("ch2 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch1,zoom_max_ch1)
                ylim(zoom_min_ch2,zoom_max_ch2)
            #legend(loc=0)
            #savefig("sc2 ch1-ch2 signal %d - %d" %(ch3_lim[0],ch3_lim[1]))
        
        
         #   figure('sc2 ch1/ch3 signal %d'%ch2_lim[0]).set_tight_layout(True)
         #   clf()
            #subplot(121)
            #title("Scatter plot ch1/ch3")
            #_,_,_,im=plt.hist2d(out1s[trs13s],out3s[trs13s],bins=tutti,norm=LogNorm(),cmap='jet')
            #colorbar(im)
            #plot(box_ch1,box_ch3,color="black")
            #if(scala=="auto"):
            #    xlim(min(out1s)-5,max(out1s)+5)
            #    ylim(min(out3s)-5,max(out3s)+10)
            #else:
            #    xlim(zoom_min_ch1,zoom_max_ch1)
            #    ylim(zoom_min_ch3,zoom_max_ch3)
            #plot(out1s_13_,out3s_13_, linestyle="", marker="." ,color="white", label="c3")
            subplot(132)
            title("Scatter plot ch1/ch3 con ch2 in (%d,%d)[keV]"%(ch2_lim[0],ch2_lim[1]))
            _,_,_,im=plt.hist2d(out1s_13,out3s_13,bins=tutti,norm=LogNorm(),cmap='jet')
            colorbar(im)
            #plot(out1s_13,out3s_13, linestyle="", marker="x" ,color="black", markersize=markersize, label="ch2 signal")
            plot(box_ch1,box_ch3,color="red")
            #plot(box1_ch1,box1_ch3,color="red")
            xlabel("ch1 [keV]")
            ylabel("ch3 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch1,zoom_max_ch1)
                ylim(zoom_min_ch3,zoom_max_ch3)
            #savefig('sc2 ch1-ch3 signal %d - %d'%(ch2_lim[0],ch2_lim[1]))
            #legend(loc=0)
            
            
            #figure('sc2 ch2/ch3 signal %d'%ch1_lim[0]).set_tight_layout(True)
            #clf()
            
            #subplot(121)
            #title("Scatter plot  ch2/ch3")
            #_,_,_,im=plt.hist2d(out2s[trs23s],out3s[trs23s],bins=tutti,norm=LogNorm(),cmap='jet')
            #colorbar(im)
            #plot(box_ch2_,box_ch3,color="black")
            ##plot(out2s_23_,out3s_23_, linestyle="", marker="." ,color="white", label="c3")
            #if(scala=="auto"):
            #    xlim(min(out1s)-5,max(out1s)+5)
            #    ylim(min(out3s)-5,max(out3s)+10)
            #else:
            #    xlim(zoom_min_ch2,zoom_max_ch2)
            #    ylim(zoom_min_ch3,zoom_max_ch3)
            #    
            subplot(133)
            title("Scatter plot  ch2/ch3 con ch1 in (%d,%d)[keV]"%(ch1_lim[0],ch1_lim[1]))
            _,_,_,im=plt.hist2d(out2s_23,out3s_23,bins=tutti,norm=LogNorm(),cmap='jet')
            colorbar(im)
            #plot(out2s_23,out3s_23, linestyle="", marker="x" ,color="black", markersize=markersize, label="ch1 signal")
            plot(box_ch2_,box_ch3,color="red")
            #plot(box1_ch2_,box1_ch3,color="red")
            xlabel("ch2 [keV]")
            ylabel("ch3 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch2,zoom_max_ch2)
                ylim(zoom_min_ch3,zoom_max_ch3)
            savefig('sc2 noise ch1 %d-%d ch2 %d-%d ch3 %d-%d'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=100)
            
            #legend(loc=0)
            #show()

#________________________________________________
'''
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
    xlim(zoom_min_ch1,zoom_max_ch1)
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
    xlim(zoom_min_ch1,zoom_max_ch1)
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
    xlim(zoom_min_ch2,zoom_max_ch2)
    ylim(zoom_min_ch3,zoom_max_ch3)

legend(loc=0)
show()
'''

