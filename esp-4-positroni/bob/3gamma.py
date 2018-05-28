import matplotlib.pyplot as plt
import numpy as np
import lab4
import lab
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from uncertainties import *
from uncertainties import unumpy as unp
from likefit import *


    
#___________________________________________
scala="zoom" #auto o zoom
doplot = True
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
files = [signal1, signal2, noise]
#files = [signal1]

for d in arange(70,300,300):
    events = [0,0,0]
    events_fit = [0,0,0]
    normalization = [0,0,0]
    rate_corr = [0,0,0]
    
    for file in files:
        if (file == signal1):
            file_name ="signal1"
            i = 0
            t_cut = [10000,45000]
            conteggio = 513087
        elif (file == signal2):
            file_name ="signal2"
            t_cut = [10000,65000]
            i= 1
            conteggio = 707589
        else:
            file_name = "noise"
            i = 2
            t_cut = [10000,50000]
            conteggio = 329981
            
        ch1, ch2, ch3, ts = lab4.loadtxt(file, unpack=True, usecols=(0,5,11,12))
        rate_corr[i] = conteggio / len(ch1)
        ts -= ts[0]
        ch1 = ch1[(ts>t_cut[0]) & (ts<t_cut[1])]
        ch2 = ch2[(ts>t_cut[0]) & (ts<t_cut[1])]
        ch3 = ch3[(ts>t_cut[0]) & (ts<t_cut[1])]
        
        #_____________________________________________________________
        #valori noti in keV
        ne = 1274
        beta = 511
        
        if (file==signal1):
            nech1, betach1, nech2, betach2, nech3, betach3 = 1046.68, 519.96, 868.54, 439.09, 991.28, 537.05
    
        elif (file==signal2):
            nech1, betach1, nech2, betach2, nech3, betach3 = 1091.84,549.45,878.81,454.78,987.84,538.27
            
        else:
            nech1, betach1, nech2, betach2, nech3, betach3 = 1071.16,527.71,845.84,428.40,965.41,518.73
            
        mch1 = (nech1-betach1)/(ne-beta)
        qch1 = (ne*betach1-beta*nech1)/(ne-beta)
        mch2 = (nech2-betach2)/(ne-beta)
        qch2 = (ne*betach2-beta*nech2)/(ne-beta)
        #mch3 = (nech3-betach3)/(na-beta)
        #qch3 = (ne*betach3-beta*nech3)/(ne-beta)
        
        #parabola canale 3
        cch3 = (nech3 - (ne*betach3)/beta)/(ne**2-beta*ne)
        bch3 = betach3/beta - cch3 * beta
        
        ch1_ = (ch1-qch1)/mch1
        ch2_ = (ch2-qch2)/mch2
        #ch3_ = (ch3-qch3)/mch3
        ch3_ = (-bch3 + sqrt(np.abs(bch3**2 + 4*cch3*ch3)))/(2*cch3)
        
        ch1 = ch1_[~np.isnan(ch1_)]
        ch2 = ch2_[~np.isnan(ch2_)]
        ch3 = ch3_[~np.isnan(ch3_)]
        
        bin_dim = 6
        bins_ch1=(arange(0.5,600.5)[0::bin_dim]-qch1)/mch1
        bins_ch2=(arange(0.5,600.5)[0::bin_dim]-qch2)/mch2
        bins_ch3=(-bch3 + sqrt(np.abs(bch3**2 + 4*cch3*(arange(0.5,600.5)[0::bin_dim]))))/(2*cch3)
        
        
        #_____________________________________________________________
        #calcolo box
        
        def Energy (ang1, ang2):
            ch3en = 2*beta / (sin(ang2)/sin(ang1)+cos(ang2)+1+cos(ang1)*sin(ang2)/sin(ang1))
            ch2en =  ch3en * sin(ang2)/sin(ang1)
            ch1en = 2*beta - ch2en -ch3en
            return [ch1en,ch2en,ch3en]
        
        ang1 = 59 /180 * np.pi
        ang2 = 43 /180 * np.pi
        #d = 280
        
        d1sup = d
        d2sup = d
        d3sup = d
        d1inf = d
        d2inf = d
        d3inf = d
        
        ch1_en = Energy(ang1,ang2)[0]
        ch2_en = Energy(ang1,ang2)[1]
        ch3_en = Energy(ang1,ang2)[2]  
#        ch1_lim = [ch1_en-d1inf,ch1_en+d1sup]
#        ch2_lim = [ch2_en-d2inf,ch2_en+d2sup]
#        ch3_lim = [ch3_en-d3inf,ch3_en+d3sup]  
        
        ch1_lim = [301,477]
        ch2_lim = [206,383]
        ch3_lim = [297,434] 
        
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
        
        
#        out1 = np.concatenate([out1, ch1])
#        out2 = np.concatenate([out2, ch2])
#        out3 = np.concatenate([out2, ch2])
#        
#        out1_12 = np.concatenate([out1_12, ch1[trs12]])
#        out2_12 = np.concatenate([out2_12, ch1[trs12]])
#        out1_13 = np.concatenate([out1_13, ch1[trs13]])
#        out3_13 = np.concatenate([out3_13, ch1[trs13]])
#        out2_23 = np.concatenate([out2_23, ch1[trs23]])
#        out3_23 = np.concatenate([out3_23, ch1[trs23]])

        #___________________________________________
            
        #trs_s= np.concatenate([trs_s, trs12 & trs13 & trs23])
        trs_s = trs12 & trs13 & trs23

        normalization[i] = len(ch1)
        events[i] = sum(trs_s)
        rate = sum(trs_s) /(t_cut[1]-t_cut[0])
        rate_err = sqrt(sum(trs_s))/(t_cut[1]-t_cut[0])
        #print("file: "+file_name+" ang1: %d, ang2: %d d=%d  \nrate = %.4f +/- %.4f"%(ang1/np.pi*180,ang2/np.pi*180,d,rate, rate_err))
         
#_______________________________________________________________________________________
#_______________________________________________________________________________________
        
        #FIT LIKELIHOOD
        samples = array([ch1[trs_s],ch2[trs_s],ch3[trs_s]])
        volume = (ch1_lim[1]-ch1_lim[0])*(ch2_lim[1]-ch2_lim[0])*(ch3_lim[1]-ch3_lim[0])
        p0 = np.array([ch1_en,ch2_en,ch3_en,10,10,10,10,10,10,-0.1])
        output = likelihood_fit(minus_log_likelihood, p0, args=(samples, volume))
        #print(lab.format_par_cov(output.par, output.cov))
        # compute meaningful parameters
        upar = uncertainties.correlated_values(output.par, output.cov)
        
        mu = upar[:3]
        L_par = upar[3:9]
        L = np.zeros((3, 3), dtype=object)
        L[np.triu_indices(3)] = L_par
        Sigma = np.dot(L, L.T)
        sigma = unumpy.sqrt(np.diag(Sigma))
        Corr = Sigma / np.outer(sigma, sigma)
        fraction = ufun_01(upar[-1])
        N_sig = uncertainties.ufloat(samples.shape[1], np.sqrt(samples.shape[1])) * fraction
        
        uf_str = np.vectorize(lambda uf: '{:P}'.format(uf) if abs(uf.s / uf.n) > 1e-6 else '{:.3g}'.format(uf.n))
        mu_pretty = lab.TextMatrix([uf_str(mu)])
        corr_pretty = lab.TextMatrix(uf_str(Corr))
        print('Mean:')
        print(mu_pretty)
        print('\nCorrelation matrix:')
        print(corr_pretty)
        print('\nNumber of signal (fraction):')
        print('{:P} ({:P})'.format(N_sig, fraction))
        events_fit[i]=N_sig
        

#_______________________________________________________________________________________
#_______________________________________________________________________________________ 
        
        
        #________________________________________________________
        #Spettro 2D 2D
        
        if (doplot==True):
            figure('sc2 '+file_name+' ch1/ch2/ch3: %d/%d/%d'%(ch1_lim[0]+d,ch2_lim[0]+d,ch3_lim[0]+d),figsize=(10, 16)).set_tight_layout(True)
            clf()
                  
            subplot(321)
            title("Spettro 2D PMT1/PMT2")
            _,_,_,im=plt.hist2d(out1,out2,bins=(bins_ch1,bins_ch2),norm=LogNorm(),cmap='jet')
            colorbar(im)
            plot(box_ch1,box_ch2,color="red")
            xlabel("PMT1 [keV]")
            ylabel("PMT2 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch1,zoom_max_ch1)
                ylim(zoom_min_ch2,zoom_max_ch2)    
                
            subplot(323)
            title("Spettro 2D PMT1/PMT3")
            _,_,_,im=plt.hist2d(out1,out3,bins=(bins_ch1,bins_ch3),norm=LogNorm(),cmap='jet')
            colorbar(im)
            plot(box_ch1,box_ch3,color="red")
            xlabel("PMT1 [keV]")
            ylabel("PMT3 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch1,zoom_max_ch1)
                ylim(zoom_min_ch3,zoom_max_ch3)
            
            subplot(325)
            title("Spettro 2D PMT2/PMT3")
            _,_,_,im=plt.hist2d(out2,out3,bins=(bins_ch2,bins_ch3),norm=LogNorm(),cmap='jet')
            colorbar(im)
            plot(box_ch2_,box_ch3,color="red")
            xlabel("PMT2 [keV]")
            ylabel("PMT3 [keV]")
            if(scala=="auto"):
                xlim(-20,1400)
                ylim(-20,1400)
            else:
                xlim(zoom_min_ch1,zoom_max_ch1)
                ylim(zoom_min_ch3,zoom_max_ch3)
            
            
            subplot(322)
            title("Spettro 2D PMT1/PMT2 \n PMT3 in (%d,%d)[keV]"%(ch3_lim[0],ch3_lim[1]))
            _,_,_,im=plt.hist2d(out1_12,out2_12,bins=(bins_ch1,bins_ch2),norm=LogNorm(),cmap='jet')
            colorbar(im)
            
            plot(box_ch1,box_ch2,color="red")
            
            xlabel("PMT1 [keV]")
            ylabel("PMT2 [keV]")
        
            xlim(zoom_min_ch1,zoom_max_ch1)
            ylim(zoom_min_ch2,zoom_max_ch2)
             
            subplot(324)
            title("Spettro 2D PMT1/PMT3 \n PMT2 in (%d,%d)[keV]"%(ch2_lim[0],ch2_lim[1]))
            _,_,_,im=plt.hist2d(out1_13,out3_13,bins=(bins_ch1,bins_ch3),norm=LogNorm(),cmap='jet')
            colorbar(im)
            
            plot(box_ch1,box_ch3,color="red")
            
            xlabel("PMT1 [keV]")
            ylabel("PMT3 [keV]")
        
            xlim(zoom_min_ch1,zoom_max_ch1)
            ylim(zoom_min_ch3,zoom_max_ch3)
                
            subplot(326)
            title("Spettro 2D  PMT2/PMT3 \n PMT1 in (%d,%d)[keV]"%(ch1_lim[0],ch1_lim[1]))
            _,_,_,im=plt.hist2d(out2_23,out3_23,bins=(bins_ch2,bins_ch3),norm=LogNorm(),cmap='jet')
            colorbar(im)
            
            plot(box_ch2_,box_ch3,color="red")
            
            xlabel("PMT2 [keV]")
            ylabel("PMT3 [keV]")
        
            xlim(zoom_min_ch2,zoom_max_ch2)
            ylim(zoom_min_ch3,zoom_max_ch3)
                
            #savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.pdf'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)
            savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.png'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)
            
            show()
    
    
    signal_plus_noise0 = ufloat(events[0],sqrt(events[0]))*rate_corr[0]
    signal_plus_noise1 = ufloat(events[1],sqrt(events[1]))*rate_corr[1]
    signal_plus_noise = signal_plus_noise0 + signal_plus_noise1
    t_s = ufloat(90000,1)
    noise = ufloat(events[2], sqrt(events[2]))*rate_corr[2]
    t_n = ufloat(40000,1)
    
    total_rate_s0 = ufloat(normalization[0],sqrt(normalization[0])) * rate_corr[0]
    total_rate_s1 = ufloat(normalization[1],sqrt(normalization[1])) * rate_corr[1]
    total_rate_s = total_rate_s0 + total_rate_s1
    total_rate_s = total_rate_s/t_s
    total_rate_n = ufloat(normalization[2], sqrt(normalization[2])) * rate_corr[2]
    total_rate_n = total_rate_n/t_n
    ratio = total_rate_s/(total_rate_n)
    
    rate_signal_plus_noise =  signal_plus_noise/t_s
    rate_noise = noise/t_n*ratio
    rate_signal = rate_signal_plus_noise - rate_noise
    
    rate_signal_fit = (events_fit[0]*rate_corr[0]+events_fit[1]*rate_corr[1])/t_s
    print(rate_signal, rate_signal_fit)

