## DIFFERENZA TRA GLI ISTOGRAMMI
import lab,lab4,pylab as py,matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fit_peak import fit_peak
import gvar

file1="0518_rimbalzi.txt"
rate1=19817608/1894989*1000

file2="0518_rimbalzi_piombo.txt"
rate2=20262459/1896859*1000

ch1,ch2=lab4.loadtxt(file1,unpack=True,usecols=(0,1))
ch3,ch4=lab4.loadtxt(file2,unpack=True,usecols=(0,1))

binni=py.arange(0,1200//4)*4
#binni=py.arange(0,max(ch1)+1)

# fit e conversione

h1,e1=py.histogram(ch1,bins=binni)
h2,e2=py.histogram(ch2,bins=binni)
h3,e3=py.histogram(ch3,bins=binni)
h4,e4=py.histogram(ch4,bins=binni)

# da mettere in una funzione

def p2(count,edges):

    X=edges[1:]-edges[:1]/2
    
    for j in range(2):
        if j==0:
            dom=X[X<504]
            cont=count[X<500]
        else:
            dom=X[X>496]
            cont=count[X>500]
            
        print(j)   
        argmax=py.argmax(cont)
        cut = (dom[argmax]-50,dom[argmax]+50)
        ordinata=gvar.gvar(cont,py.sqrt(cont))
            
        if j==0:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut)
            beta=outdict['peak1_mean']
            sbeta=outdict['peak1_sigma']
        else:
            outdict,indict = fit_peak(dom,ordinata,bkg=None,npeaks=1,cut=cut)
            neon=outdict['peak1_mean']
            sneon=outdict['peak1_sigma']
    
    return beta,sbeta,neon,sneon

# figure 2d

fig=py.figure('diff')
fig.set_tight_layout(True)

rimb=fig.add_subplot(131)
rimb.set_title('Senza piombo')
rimb.minorticks_on()

ist1,x1,y1,im1=py.hist2d(ch1,ch2,bins=binni,norm=LogNorm(),cmap='jet')

#fig.colorbar(im1)

pb=fig.add_subplot(132)
pb.set_title('Con piombo')
pb.minorticks_on()

ist2,x2,y2,im2=py.hist2d(ch3,ch4,bins=binni,norm=LogNorm(),cmap='jet')
#fig.colorbar(im2)

diff=fig.add_subplot(133)
diff.set_title('Senza piombo')
diff.minorticks_on()

ist1c=ist1/rate1*(len(ch1)+len(ch2))
ist2c=ist2/rate2*(len(ch3)+len(ch4))

im3=plt.imshow(abs(ist1c-ist2c),origin='lower',norm=LogNorm(),cmap='jet')
#fig.colorbar(im3)

py.show()

'''
fig2=py.figure('1d')
fig2.set_tight_layout(True)

uno=fig2.add_subplot(111)
alt,bor,art=plt.hist(ch1,bins=binni,histtype='step')

altp,borp,artp=plt.hist(ch2,bins=binni,histtype='step')

py.show()
'''