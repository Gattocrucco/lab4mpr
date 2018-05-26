# copiato da marasciulli/differenza.py e poi modificato
## DIFFERENZA TRA GLI ISTOGRAMMI
import lab,lab4,pylab as py,matplotlib.pyplot as plt
from matplotlib import colors
from fit_peak import fit_peak
import gvar
import numpy as np

print('loading data...')

cartella_dati = '../DAQ/'

file1=cartella_dati + "0518_rimbalzi.txt"
rate1=19817608/1894989*1000

file2=cartella_dati + "0518_rimbalzi_piombo.txt"
rate2=20262459/1896859*1000

ch1,ch2=lab4.loadtxt(file1,unpack=True,usecols=(0,1))
ch3,ch4=lab4.loadtxt(file2,unpack=True,usecols=(0,1))

binni=py.arange(0, 1150, 4)

# fit e conversione

h1,e1=py.histogram(ch1,bins=binni)
h2,e2=py.histogram(ch2[ch2>200],bins=binni)
h3,e3=py.histogram(ch3,bins=binni)
h4,e4=py.histogram(ch4[ch4>200],bins=binni)

# da mettere in una funzione

print('calibration...')

def p2(count,edges):

    X=edges[1:]-edges[:1]/2
    
    for j in range(2):
        if j==0:
            dom=X[X<504]
            cont=count[X<500]
        else:
            dom=X[X>496]
            cont=count[X>500]
             
        argmax=py.argmax(cont)
        cut = (dom[argmax]-40,dom[argmax]+40)
        ordinata=gvar.gvar(cont,py.sqrt(cont))
            
        if j==0:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut)
            beta=outdict['peak1_mean']
            sbeta=outdict['peak1_sigma']
        else:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut)
            neon=outdict['peak1_mean']
            sneon=outdict['peak1_sigma']
    
    return beta,sbeta,neon,sneon


b1,sb1,n1,sn1=p2(h1,e1)
b2,sb2,n2,sn2=p2(h2,e2)
b3,sb3,n3,sn3=p2(h3,e3)
b4,sb4,n4,sn4=p2(h4,e4)

bn=510.9989 # per la gioia di Bob
nn=1274.5

# conversione

q1=(b1*nn-n1*bn)/(b1-n1)
m1=(bn-q1)/b1

q2=(b2*nn-n2*bn)/(b2-n2)
m2=(bn-q2)/b2

q3=(b3*nn-n3*bn)/(b3-n3)
m3=(bn-q3)/b3

q4=(b4*nn-n4*bn)/(b4-n4)
m4=(bn-q4)/b4

ch1=gvar.mean(ch1*m1+q1)
ch2=gvar.mean(ch2*m2+q2)
ch3=gvar.mean(ch3*m3+q3)
ch4=gvar.mean(ch4*m4+q4)

bins1 = gvar.mean(binni * m1 + q1)
bins2 = gvar.mean(binni * m2 + q2)
bins3 = gvar.mean(binni * m3 + q3)
bins4 = gvar.mean(binni * m4 + q4)

print('plot...')

# figure 2d

fig=py.figure('diff')
fig.clf()
fig.set_tight_layout(True)

norm = colors.LogNorm()
rimb, pb, diff = fig.subplots(1, 3, sharex=True, sharey=True)

pb.set_title('Con piombo')
pb.minorticks_on()
pb.set_xlabel('energia PMT 1 [keV]')

ist2,x2,y2,im2=pb.hist2d(ch3,ch4,bins=(bins3, bins4),norm=norm,cmap='jet')
fig.colorbar(im2, ax=pb)

rimb.set_title('Senza piombo')
rimb.minorticks_on()
rimb.set_xlabel('energia PMT 1 [keV]')
rimb.set_ylabel('energia PMT 2 [keV]')

ist1,x1,y1,im1=rimb.hist2d(ch1,ch2,bins=(bins1, bins2),norm=norm,cmap='jet')

diff.set_title('Residui normalizzati')
diff.minorticks_on()
diff.set_xlabel('energia PMT 1 [keV]')

diff_bins_x = np.arange(min(np.min(bins1), np.min(bins3)), max(np.max(bins1), np.max(bins3)), 8)
diff_bins_y = np.arange(min(np.min(bins2), np.min(bins4)), max(np.max(bins2), np.max(bins4)), 8)

ist1c, _, _ = np.histogram2d(ch1, ch2, bins=(diff_bins_x, diff_bins_y))
ist2c, _, _ = np.histogram2d(ch3, ch4, bins=(diff_bins_x, diff_bins_y))

ist1c_unc = np.where(ist1c > 0, np.sqrt(ist1c), 1)
sum_ist1c = np.sum(ist1c)
ist1c *= rate1 / sum_ist1c
ist1c_unc *= rate1 / sum_ist1c

ist2c_unc = np.where(ist2c > 0, np.sqrt(ist2c), 1)
sum_ist2c = np.sum(ist2c)
ist2c *= rate2 / sum_ist2c
ist2c_unc *= rate2 / sum_ist2c

ist_diff = (ist1c - ist2c) / np.sqrt(ist1c_unc ** 2 + ist2c_unc ** 2)

cdict = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0)
    ],
    'blue': [
        (0.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0)
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0)
    ]
}
cmap = colors.LinearSegmentedColormap('cippa', cdict)
vminmax = np.max(np.abs(ist_diff))
norm = colors.Normalize(vmin=-vminmax, vmax=vminmax)

im3=diff.imshow(ist_diff.T, origin='lower', cmap=cmap, aspect='auto', extent=(diff_bins_x[0], diff_bins_x[-1], diff_bins_y[0], diff_bins_y[-1]), norm=norm)
fig.colorbar(im3, ax=diff)

fig.show()

'''
fig2=py.figure('1d')
fig2.set_tight_layout(True)

uno=fig2.add_subplot(111)
alt,bor,art=plt.hist(ch1,bins='auto',histtype='step')

altp,borp,artp=plt.hist(ch2,bins='auto',histtype='step')
py.yscale('log')
py.show()
'''