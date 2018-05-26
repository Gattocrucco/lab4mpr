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
from scipy.optimize import minimize
import numdifftools
from scipy import linalg

def likelihood_fit(log_likelihood,p0, args=()):
    result = minimize(log_likelihood,p0,options=dict(disp=True), args=args)
    print(result.message)
    hessian = numdifftools.Hessian(log_likelihood)(result.x,*args)
    cov = linalg.inv(hessian)
    return result.x, cov
    
def signal(x,norm,mean,inv_cov, const):
    return norm/np.sqrt(2*np.pi*linalg.det(inv_cov)) * np.exp (-1/2 * np.reshape(x-mean, (1,-1))@inv_cov@ np.reshape(x-mean,(-1,1))) + const

def log_likelihood(p, xs):
    norm = p[0]
    mean = p[1:4]
    angles = p[4:7]
    sigma = p[7:10]
    const = p[10]
    rot1 = np.array([[cos(angles[0]),-sin(angles[0]),0],
            [sin(angles[0]),cos(angles[0]),0],
            [0,0,1]  ])
    rot2 = np.array([[cos(angles[1]),0,sin(angles[1])],
        [0,1,0],
        [-sin(angles[1]),0,cos(angles[1])]])
    rot3 = np.array([[1,0,0],
        [0,cos(angles[2]),-sin(angles[2])],
        [0,sin(angles[2]),cos(angles[2])]  ])
    diag = np.array([ [sigma[0],0,0],
            [0,sigma[1],0],
            [0,0,sigma[2]] ])
    inv_cov = linalg.inv(diag @ rot1 @ rot2 @ rot3)
    somma = 0 
    for x in xs:
        somma += np.log(signal(x, norm, mean, inv_cov, const))
    return somma

    
#___________________________________________
scala="auto" #auto o zoom
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
#files = [signal1, signal2, noise]
files = [signal1]
    
for d in arange(70,300,300):
    events = [0,0,0]
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
        #fit calibrazione
        
    #    def gauss(x,N1,u,sigma,N2,l):
    #        return N1 * np.e**(-1*((x-u)**2)/(2*sigma**2)) + N2 * np.e**(l * (x-u))
    #    def gauss_only(x,N1,u,sigma,N2,l):
    #        return N1 * np.e**(-1*((x-u)**2)/(2*sigma**2))
    #    def fondo_only(x,N1,u,sigma,N2,l):
    #        return N2 * np.e**(l * (x-u))
    #    
    #    if (file==signal1):
    #        ch1_beta_cut=[440,550]
    #        ch1_ne_cut=[950,1100]
    #        ch2_beta_cut=[380,490]
    #        ch2_ne_cut=[820,930]
    #        ch3_beta_cut=[480,580]
    #        ch3_ne_cut=[940,1040]
    #        
    #        
    #    if (file==signal2):
    #        ch1_beta_cut=[500,600]
    #        ch1_ne_cut=[980,1115]
    #        ch2_beta_cut=[370,480]
    #        ch2_ne_cut=[800,920]
    #        ch3_beta_cut=[450,570]
    #        ch3_ne_cut=[950,1030]
    #        
    #    if (file==noise):
    #        ch1_beta_cut=[500,600]
    #        ch1_ne_cut=[980,1115]
    #        ch2_beta_cut=[370,480]
    #        ch2_ne_cut=[780,900]
    #        ch3_beta_cut=[450,600]
    #        ch3_ne_cut=[880,1000]
    #    
    ##    figure(1)
    #    
    #    hist1 = hist(ch1, bins = arange(ch1_beta_cut[0],ch1_beta_cut[1]))
    #    X_b1 = (hist1[1]+0.5)[:-1]
    #    Y = hist1[0]
    #    dY = sqrt(Y)
    #    val=[1e3,(ch1_beta_cut[0]+ch1_beta_cut[1])/2,(ch1_beta_cut[0]-ch1_beta_cut[1])/2,2e2,-1e-2]
    #    outb1 = lab.fit_curve(gauss,X_b1,Y,dy=dY,p0=val, method="odrpack")
    #    
    #    hist2 = hist(ch1, bins = arange(ch1_ne_cut[0],ch1_ne_cut[1]))
    #    X_n1 = (hist2[1]+0.5)[:-1]
    #    Y = hist2[0]
    #    dY = sqrt(Y)
    #    val=[1e2,(ch1_ne_cut[0]+ch1_ne_cut[1])/2,(ch1_ne_cut[0]-ch1_ne_cut[1])/2,1e1,-1e-2]
    #    outn1 = lab.fit_curve(gauss,X_n1,Y,dy=dY,p0=val, method="odrpack")
    #   
    #    hist3 = hist(ch2, bins = arange(ch2_beta_cut[0],ch2_beta_cut[1]))
    #    X_b2 = (hist3[1]+0.5)[:-1]
    #    Y = hist3[0]
    #    dY = sqrt(Y)
    #    val=[1e3,(ch2_beta_cut[0]+ch2_beta_cut[1])/2,(ch2_beta_cut[0]-ch2_beta_cut[1])/2,1e2,-1e-2]
    #    outb2 = lab.fit_curve(gauss,X_b2,Y,dy=dY,p0=val, method="odrpack")
    #    
    #    hist4 = hist(ch2, bins = arange(ch2_ne_cut[0],ch2_ne_cut[1]))
    #    X_n2 = (hist4[1]+0.5)[:-1]
    #    Y = hist4[0]
    #    dY = sqrt(Y)
    #    val=[1e2,(ch2_ne_cut[0]+ch2_ne_cut[1])/2,(ch2_ne_cut[0]-ch2_ne_cut[1])/2,1e1,-1e-2]
    #    outn2 = lab.fit_curve(gauss,X_n2,Y,dy=dY,p0=val, method="odrpack")
    #    
    #    hist5 = hist(ch3, bins = arange(ch3_beta_cut[0],ch3_beta_cut[1]))
    #    X_b3 = (hist5[1]+0.5)[:-1]
    #    Y = hist5[0]
    #    dY = sqrt(Y)
    #    val=[1e3,(ch3_beta_cut[0]+ch3_beta_cut[1])/2,(ch3_beta_cut[0]-ch3_beta_cut[1])/2,1e2,-1e-2]
    #    outb3 = lab.fit_curve(gauss,X_b3,Y,dy=dY,p0=val, method="odrpack")
    #    
    #    hist6 = hist(ch3, bins = arange(ch3_ne_cut[0],ch3_ne_cut[1]))
    #    X_n3 = (hist6[1]+0.5)[:-1]
    #    Y = hist6[0]
    #    dY = sqrt(Y)
    #    val=[1e2,(ch3_ne_cut[0]+ch3_ne_cut[1])/2,(ch3_ne_cut[0]-ch3_ne_cut[1])/2,1e1,-1e-2]
    #    outn3 = lab.fit_curve(gauss,X_n3,Y,dy=dY,p0=val, method="odrpack")
    #    
    #    bins_cal = arange(0,1200)
    #    figure('Calibration ch1').set_tight_layout(True)
    #    clf()
    #    title("Acquisizioni singole signal")
    #    hist(ch1, bins = bins_cal, label="ch1 n=%d"%(len(ch1_cal)),histtype="step")
    #    plot(X_b1,gauss(X_b1,*outb1.par))
    #    plot(X_n1,gauss(X_n1,*outn1.par))
    #    yscale('log')
    #    show()
    #    
    #    figure('Calibration ch2').set_tight_layout(True)
    #    clf()
    #    hist(ch2, bins = bins_cal, label="ch2 n=%d"%(len(ch2_cal)),histtype="step")
    #    plot(X_b2,gauss(X_b2,*outb2.par))
    #    plot(X_n2,gauss(X_n2,*outn2.par))
    #    yscale('log')
    #    
    #    figure('Calibration ch3').set_tight_layout(True)
    #    clf()
    #    hist(ch3, bins = bins_cal, label="ch3 n=%d"%(len(ch3_cal)),histtype="step")
    #    plot(X_b3,gauss(X_b3,*outb3.par))
    #    plot(X_n3,gauss(X_n3,*outn3.par))
    #    yscale('log')
        #show()
        
        #valori noti in keV
        ne = 1274
        beta = 511
        
        if (file==signal1):
            nech1, betach1, nech2, betach2, nech3, betach3 = 1046.68, 519.96, 868.54, 439.09, 991.28, 537.05
    
        elif (file==signal2):
            nech1, betach1, nech2, betach2, nech3, betach3 = 1091.84,549.45,878.81,454.78,987.84,538.27
            
        else:
            nech1, betach1, nech2, betach2, nech3, betach3 = 1071.16,527.71,845.84,428.40,965.41,518.73
            
    #    nech1 = outn1.par[1]
    #    betach1 = outb1.par[1]
    #    nech2 = outn2.par[1]
    #    betach2 = outb2.par[1]
    #    nech3 = outn3.par[1]
    #    betach3 = outb3.par[1]
    #    
    #    print(nech1,betach1,nech2,betach2,nech3,betach3)
        
        
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
        ch3_ = (-bch3 + sqrt(bch3**2 + 4*cch3*ch3))/(2*cch3)
        
        ch1 = ch1_[(ch1_>0) & (ch2_>0) & (ch3_>0)]
        ch2 = ch2_[(ch1_>0) & (ch2_>0) & (ch3_>0)]
        ch3 = ch3_[(ch1_>0) & (ch2_>0) & (ch3_>0)]
        
        bin_dim = 6
        bins_ch1=(arange(0.5,1100.5)[0::bin_dim]-qch1)/mch1
        bins_ch2=(arange(0.5,1100.5)[0::bin_dim]-qch2)/mch2
        bins_ch3=(-bch3 + sqrt(bch3**2 + 4*cch3*(arange(0.5,1050.5)[0::bin_dim])))/(2*cch3)
        
        
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
        ch1_lim = [ch1_en-d1inf,ch1_en+d1sup]
        ch2_lim = [ch2_en-d2inf,ch2_en+d2sup]
        ch3_lim = [ch3_en-d3inf,ch3_en+d3sup]  
        
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
        _1ch1_norm = [650,1100]
        _1ch2_norm = [100,450]
        _1ch3_norm = [100,450]
        
        _1box_ch1_norm = [_1ch1_norm[0],_1ch1_norm[0],_1ch1_norm[1],_1ch1_norm[1],_1ch1_norm[0]]
        _1box_ch2_norm = [_1ch2_norm[0],_1ch2_norm[1],_1ch2_norm[1],_1ch2_norm[0],_1ch2_norm[0]]
        _1box_ch2_norm_ = [_1ch2_norm[0],_1ch2_norm[0],_1ch2_norm[1],_1ch2_norm[1],_1ch2_norm[0]]
        _1box_ch3_norm = [_1ch3_norm[0],_1ch3_norm[1],_1ch3_norm[1],_1ch3_norm[0],_1ch3_norm[0]]
        
        
        _1tr_norm12 = (ch3>_1ch3_norm[0]) & (ch3 <_1ch3_norm[1])
        _1tr_norm13 = (ch2>_1ch2_norm[0]) & (ch2 <_1ch2_norm[1])
        _1tr_norm23 = (ch1>_1ch1_norm[0]) & (ch1 <_1ch1_norm[1])
        _1tr_norm= _1tr_norm12 & _1tr_norm13 & _1tr_norm23
        
        _2ch1_norm = [100,450]
        _2ch2_norm = [650,1100]
        _2ch3_norm = [100,450]
        
        _2box_ch1_norm = [_2ch1_norm[0],_2ch1_norm[0],_2ch1_norm[1],_2ch1_norm[1],_2ch1_norm[0]]
        _2box_ch2_norm = [_2ch2_norm[0],_2ch2_norm[1],_2ch2_norm[1],_2ch2_norm[0],_2ch2_norm[0]]
        _2box_ch2_norm_ = [_2ch2_norm[0],_2ch2_norm[0],_2ch2_norm[1],_2ch2_norm[1],_2ch2_norm[0]]
        _2box_ch3_norm = [_2ch3_norm[0],_2ch3_norm[1],_2ch3_norm[1],_2ch3_norm[0],_2ch3_norm[0]]
        
        
        _2tr_norm12 = (ch3>_2ch3_norm[0]) & (ch3 <_2ch3_norm[1])
        _2tr_norm13 = (ch2>_2ch2_norm[0]) & (ch2 <_2ch2_norm[1])
        _2tr_norm23 = (ch1>_2ch1_norm[0]) & (ch1 <_2ch1_norm[1])
        _2tr_norm= _2tr_norm12 & _2tr_norm13 & _2tr_norm23
        
        _3ch1_norm = [100,450]
        _3ch2_norm = [100,450]
        _3ch3_norm = [650,1100]
        
        _3box_ch1_norm = [_3ch1_norm[0],_3ch1_norm[0],_3ch1_norm[1],_3ch1_norm[1],_3ch1_norm[0]]
        _3box_ch2_norm = [_3ch2_norm[0],_3ch2_norm[1],_3ch2_norm[1],_3ch2_norm[0],_3ch2_norm[0]]
        _3box_ch2_norm_ = [_3ch2_norm[0],_3ch2_norm[0],_3ch2_norm[1],_3ch2_norm[1],_3ch2_norm[0]]
        _3box_ch3_norm = [_3ch3_norm[0],_3ch3_norm[1],_3ch3_norm[1],_3ch3_norm[0],_3ch3_norm[0]]
        
        _3tr_norm12 = (ch3>_3ch3_norm[0]) & (ch3 <_3ch3_norm[1])
        _3tr_norm13 = (ch2>_3ch2_norm[0]) & (ch2 <_3ch2_norm[1])
        _3tr_norm23 = (ch1>_3ch1_norm[0]) & (ch1 <_3ch1_norm[1])
        _3tr_norm= _3tr_norm12 & _3tr_norm13 & _3tr_norm23
        
        #normalization[i] = sum(_1tr_norm) +sum(_2tr_norm) +sum(_3tr_norm)
        normalization[i] = len(ch1)
        events[i] = sum(trs_s)
        rate = sum(trs_s) /(t_cut[1]-t_cut[0])
        rate_err = sqrt(sum(trs_s))/(t_cut[1]-t_cut[0])
        #print("file: "+file_name+" ang1: %d, ang2: %d d=%d  \nrate = %.4f +/- %.4f"%(ang1/np.pi*180,ang2/np.pi*180,d,rate, rate_err))
         
        
        #FIT LIKELIHOOD
        p0 = [1000, ch1_en,ch2_en,ch3_en,0.1,0.1,0.1,100,100,100,100]
        xs = np.array([ch1[trs_s],ch2[trs_s],ch3[trs_s]]).T
        #par, cov = likelihood_fit(log_likelihood,p0,args=(xs,))
        #print (lab.format_par_cov(par,cov))
        
        
        
        #________________________________________________________
        #SCATTER PLOT 2D
        
        
#        figure('sc2 '+file_name+' ch1/ch2/ch3: %d/%d/%d'%(ch1_lim[0]+d1,ch2_lim[0]+d2,ch3_lim[0]+d3),figsize=(10, 16)).set_tight_layout(True)
#        clf()
#              
#        subplot(321)
#        title("Scatter plot ch1/ch2")
#        _,_,_,im=plt.hist2d(out1,out2,bins=bins_ch1,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        plot(box_ch1,box_ch2,color="red")
#        plot(_1box_ch1_norm,_1box_ch2_norm,color="white")
#        plot(_2box_ch1_norm,_2box_ch2_norm,color="grey")
#        plot(_3box_ch1_norm,_3box_ch2_norm,color="black")
#        xlabel("ch1 [keV]")
#        ylabel("ch2 [keV]")
#        if(scala=="auto"):
#            xlim(-20,1400)
#            ylim(-20,1400)
#        else:
#            xlim(zoom_min_ch1,zoom_max_ch1)
#            ylim(zoom_min_ch2,zoom_max_ch2)    
#            
#        subplot(323)
#        title("Scatter plot ch1/ch3")
#        _,_,_,im=plt.hist2d(out1,out3,bins=bins_ch2,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        plot(box_ch1,box_ch3,color="red")
#        plot(_1box_ch1_norm,_1box_ch3_norm,color="white")
#        plot(_2box_ch1_norm,_2box_ch3_norm,color="grey")
#        plot(_3box_ch1_norm,_3box_ch3_norm,color="black")
#        xlabel("ch1 [keV]")
#        ylabel("ch3 [keV]")
#        if(scala=="auto"):
#            xlim(-20,1400)
#            ylim(-20,1400)
#        else:
#            xlim(zoom_min_ch1,zoom_max_ch1)
#            ylim(zoom_min_ch3,zoom_max_ch3)
#        
#        subplot(325)
#        title("Scatter plot ch2/ch3")
#        _,_,_,im=plt.hist2d(out2,out3,bins=bins_ch3,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        plot(box_ch2_,box_ch3,color="red")
#        plot(_1box_ch2_norm_,_1box_ch3_norm,color="white")
#        plot(_2box_ch2_norm_,_2box_ch3_norm,color="grey")
#        plot(_3box_ch2_norm_,_3box_ch3_norm,color="black")
#        xlabel("ch2 [keV]")
#        ylabel("ch3 [keV]")
#        if(scala=="auto"):
#            xlim(-20,1400)
#            ylim(-20,1400)
#        else:
#            xlim(zoom_min_ch1,zoom_max_ch1)
#            ylim(zoom_min_ch3,zoom_max_ch3)
#        
#        
#            
#        
#        
#        subplot(322)
#        title("Scatter plot ch1/ch2 con ch3 in (%d,%d)[keV]"%(ch3_lim[0],ch3_lim[1]))
#        _,_,_,im=plt.hist2d(out1_12,out2_12,bins=bins_ch1,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        
#        plot(box_ch1,box_ch2,color="red")
#        
#        xlabel("ch1 [keV]")
#        ylabel("ch2 [keV]")
#    
#        xlim(zoom_min_ch1,zoom_max_ch1)
#        ylim(zoom_min_ch2,zoom_max_ch2)
#         
#        subplot(324)
#        title("Scatter plot ch1/ch3 con ch2 in (%d,%d)[keV]"%(ch2_lim[0],ch2_lim[1]))
#        _,_,_,im=plt.hist2d(out1_13,out3_13,bins=bins_ch2,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        
#        plot(box_ch1,box_ch3,color="red")
#        
#        xlabel("ch1 [keV]")
#        ylabel("ch3 [keV]")
#    
#        xlim(zoom_min_ch1,zoom_max_ch1)
#        ylim(zoom_min_ch3,zoom_max_ch3)
#            
#        subplot(326)
#        title("Scatter plot  ch2/ch3 con ch1 in (%d,%d)[keV]"%(ch1_lim[0],ch1_lim[1]))
#        _,_,_,im=plt.hist2d(out2_23,out3_23,bins=bins_ch3,norm=LogNorm(),cmap='jet')
#        colorbar(im)
#        
#        plot(box_ch2_,box_ch3,color="red")
#        
#        xlabel("ch2 [keV]")
#        ylabel("ch3 [keV]")
#    
#        xlim(zoom_min_ch2,zoom_max_ch2)
#        ylim(zoom_min_ch3,zoom_max_ch3)
#            
#        #savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.pdf'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)
#        savefig('sc2 '+file_name +' ch1 %d-%d ch2 %d-%d ch3 %d-%d.png'%(ch1_lim[0],ch1_lim[1],ch2_lim[0],ch2_lim[1],ch3_lim[0],ch3_lim[1]),dpi=200)
#        
        #show()
    
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
    print(rate_signal_plus_noise, rate_noise, rate_signal)
    #print("rate s:",rate_signal)

