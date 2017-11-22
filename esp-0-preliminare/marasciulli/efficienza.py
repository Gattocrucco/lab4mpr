## EFFICIENZA DEL PMT3
try:
    os.chdir("esperienza preliminare")
except FileNotFoundError:
    pass
    
v=array([i for i in range(1700,2100,100)])
file=["eff_1700V.txt","eff_1800V.txt","eff_1900V.txt","eff_2000V.txt"]
risu=open("efficienza.txt","w")


py.figure(1)
py.rc("font",size=16)
py.grid(linestyle="--",color="black")
py.minorticks_on()

for j in range(len(file)):
    x=py.empty(4)
    y=uarray([],[])

    s1,s2,s3,s4=py.loadtxt(file[j],unpack=True)
    S=[s1,s2,s3,s4]
    
    for k in range(len(S)):
        x[k]=S[k][0]
    
    for t in range(len(S)):
        due=uf(S[t][1],sqrt(S[t][1]))
        tre=uf(S[t][2],sqrt(S[t][2]))
        rapp=tre/due
        y=py.append(y,rapp)
    
    py.errorbar(x,med(y),err(y),linestyle="--",marker="o",label="%i V"%v[j])
    
    print("alimentazione pmt3=%i V"%v[j],file=risu)
    for j in range(len(x)):
        print("{:.1f} mV".format(x[j]), "\t efficienza non correlata= {:.1u} ".format(y[j]),file=risu)
    print("\n",file=risu)
    
py.title("Efficienza PMT3",size=18)
py.xlabel("soglia (mV)")
py.ylabel("efficienza")

py.legend(loc="lower right",fontsize=10)
py.tight_layout()
py.show()
risu.close()