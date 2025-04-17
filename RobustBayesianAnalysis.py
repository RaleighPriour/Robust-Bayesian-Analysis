if __name__=="__main__":
    print("Libaries Loading")
import matplotlib.pyplot as plt
import mpmath.calculus
import numpy as np
from mpmath import harmonic, mp, exp, loggamma,log,polygamma,sqrt,re,cbrt
from collections import deque
import numbers
mp.dps=25
if __name__=="__main__":
    print("Libaries Loaded")
class Beta():
    def __init__(self,a,b):
        self.a=(a)
        self.b=(b)
        self.n=a+b
        self.logNormalazationConstant=loggamma(a+b+2)-loggamma(a+1)-loggamma(b+1)
        self.mode,self.mean,self.variance,self.stdev,self.invstdev=None,None,None,None,None
        self._stats_computed=False
    def computeStats(self):
        if not self._stats_computed:
            if self.n==0:
                self.mode=None
            else:
                self.mode=self.a/self.n
            self.mean=(self.a+1)/(self.n+2)
            self.variance=((self.a+1)*(self.b+1))/((self.n+2)*(self.n+2)*(self.n+3))
            self.stdev=(sqrt(self.variance))
            self.invstdev=(sqrt((self.n+2)*(self.n+2)*(self.n+3)/((self.a+1)*(self.b+1))))
            self._stats_computed=True
    def pdf(self,x):
        if x==1:
            if self.b==0:
                return (1+self.a)
            return 0
        elif x==0:
            if self.a==0:
                return (1+self.b)
            return 0
        lnx=log(x)
        lnxc=log(1-x)
        return (exp(self.logNormalazationConstant+self.a*lnx+self.b*lnxc))  
def highestProbDistBeliefForBetaDist(a,b):
    t=exp(harmonic(a)-harmonic(b))
    return (1-1/(1+t))
def dydxForOptimalBeta(a,n):
    "computes dy/dx for the plot of the optimal max beta beliefs very complicated formula"
    b=n-a
    lnx=harmonic(a)-harmonic(b)
    xplus=1+exp(lnx)
    return (exp(loggamma(n+2)-loggamma(a+1)-loggamma(b+1)+(a-1)*lnx-log(xplus)*(n-2))*(a-n+n/xplus))
def dydaForOptimalBetaYandX(a,n):
    "computes dy/da for the plot of the optimal max beta beliefs very complicated formula"
    b=n-a
    lnx=harmonic(a)-harmonic(b)
    xplus=1+exp(lnx)
    y=(exp(loggamma(n+2)-loggamma(a+1)-loggamma(b+1)+a*lnx-n*log(xplus)))   
    return (y*(a-n+n/xplus)*(polygamma(1,a+1)-polygamma(1,b+1))),(y),(1-1/(1+exp(lnx)))
def dadxForOptimalBetaYandX(a,n):
    b=n-a
    lnx=harmonic(a)-harmonic(b)
    xplus=1+exp(lnx)
    y=(exp(loggamma(n+2)-loggamma(a+1)-loggamma(b+1)+a*lnx-n*log(xplus))) 
def ddydxdxAndOtherInfo(a,n):
    """Computes ddy/dxdx for the plot of optimal max bet beliefs among other things very long formulas
    returns ddy/dxdx,da/dx,y,x"""
    b=n-a
    ΔH=harmonic(a)-harmonic(b)
    expΔH=exp(ΔH)
    expΔHplus1=1+expΔH
    x=1-1/expΔHplus1
    di,tri=polygamma(1,a+1)+polygamma(1,b+1),polygamma(2,a+1)-polygamma(2,b+1)    
    dlnyda=(a-n*x)*di
    dadx=expΔHplus1/(x*di)
    ddlnydada=di+(a-n*x)*tri-n*x*di*di/expΔHplus1
    lny=loggamma(n+2)-loggamma(a+1)-loggamma(b+1)+a*log(x)+b*log(1-x)
    dxdxdada=x*((1-2*x)*di*di+tri)/expΔHplus1
    lnddydxdx=lny+2*log(dadx)+log(ddlnydada+dlnyda*(dlnyda-dadx*dxdxdada))
    return (re(exp(lnddydxdx))),(dadx),(exp(lny)),(x)
def maxProb(a,n):
    b=n-a
    lnx=harmonic(a)-harmonic(b)
    xplus=1+exp(lnx)
    return (exp(loggamma(n+2)-loggamma(a+1)-loggamma(b+1)+a*lnx-n*log(xplus)))
def derivatives(y:numbers.Real,x:numbers.Real,a:numbers.Real,b:numbers.Real,multi:numbers.Real=1,divi:numbers.Real=1):
    "returns dydx,ddydxdx for beta distrubtion ∝x^a(1-x)^b with prob density y at point x"
    match x:
        case 0:
            dydx=-b*y
            ddydxdx=b*(b-1)*y
            #dddydxdxdx=b*(b*(3-b)-2)*y
        case 1:
            dydx=a*y
            ddydxdx=a*(a-1)*y
            #dddydxdxdx=a*(a*(a-3)+2)*y
        case _:
            xc=1-x
            tmp1=a/x-b/xc
            dydx=y*tmp1
            xsq,xsqc=x*x,xc*xc
            #xcb,xcbc=xsq*x,xsqc*x
            tmp2=a/xsq+b/xsqc
            #tmp3=a/xcb-b/xcbc
            ddydxdx=y*(tmp1*tmp1-tmp2)
            #dddydxdxdx=y*(tmp1**3-3*tmp1*tmp2+2*tmp3)
    return multi*dydx/divi,multi*ddydxdx/divi#,multi*dddydxdxdx/divi
class dda:
    def __init__(self,successes,failures,d,maxY,stdev:None):
        self.successes=successes
        self.failures=failures
        self.d=d
        self.maxY=maxY
        if stdev==None:
            tmp=RobustBayesianAnalysis(self.d,150,self.successes,self.failures)
            stdev=tmp.maxStdev()
        self.stdev=stdev
        self.X=[]
    def f(self,a):
        self.slope,self.dadx,y,x=ddydxdxAndOtherInfo(a+self.successes,self.d+self.successes+self.failures)
        self.X.append(x)
        return y
    def ddydxdx(self,x,y):
        return self.stdev*self.slope/(self.maxY*self.dadx**3)
class RobustBayesianAnalysis():
    MAXBINOMINALN=2<<31
    MAXPROB=2<<30
    def __init__(self,d,an:int=150,successes:int=0,failures:int=0):
        """"   
           d : Degree of Prior Certitude of Beliefs
           an : Asymptotic number of points calculated (can be as low as 75 and still give good results)
           successes : Observed successes defaults to 0
           failures  : Observed failures defaults to 0
        """
        self.d=d
        self.an=an
        self.h=6.61889/(an**3)
        self.HPDB=highestProbDistBeliefForBetaDist
        self.V=np.arange(0,self.d,self.h)
        self.successes=successes
        self.failures=failures
        self.sqrt2=sqrt(2)
        self.cbrt6=cbrt(6)
        self.multi=1
    def maxVar(self):
        a,b=self.successes,self.failures
        if self.successes>self.failures:
            a,b=b,a
        aoptimal=float(((.75*b+7/6)**2-1/9)**.5-(.25*b+1.5))#in case of mpmath
        if aoptimal>a:
            a=min(a+self.d,aoptimal)
        beta=Beta(a,b)
        beta.computeStats()
        return beta.variance
    def maxStdev(self):
        return self.maxVar()**.5
    def sampler(self,f,ddfdxdx,h,upperbound,lowerbound,multi:numbers.Real=1,x=None,mini_h=None,minimum=0):
        #not needed
        # toClose=True
        # if mini_h==None or mini_h==h:
        #     mini_h=h
        #     toClose=False
        if x==None:
            x=lowerbound
        toClose=False
        "f is functions with one parameter x"
        "ddydxdx is a function with 2 parameters x,y"
        "Returns X,Y"
        X,Y=[],[]      
        while lowerbound<=x and x<=upperbound:
            y=f(x)
            if y<=minimum:
                return X,Y
            Y.append(y)
            X.append(x)
            ddydxdx=abs(ddfdxdx(x,y))
            if toClose:
                xtmp=x+multi*mini_h
                ytmp=f(xtmp)
                ddydxdxtmp=abs(ddfdxdx(xtmp,ytmp))
                toClose=ddydxdxtmp>ddydxdx
                if toClose:
                    ddydxdx=ddydxdxtmp
            if ddydxdx==0:
                return X,Y                    
            x+=multi*self.cbrt6*cbrt(h/ddydxdx)             
        return X,Y
    def computeGraph(self,ends:bool=None):
        """
        ends: If False cuts outs the regions with less than .1% relative likelyhood from the graph recommened when dealing with 10**6 or more datapoints
        """        
        if ends==None:
            ends=(self.successes+self.failures+self.d)<(2<<20)
        self.lowerBeta,self.upperBeta=Beta(self.successes,self.d+self.failures),Beta(self.d+self.successes,self.failures)
        self.upperBeta.computeStats()
        self.lowerBeta.computeStats()
        self.maxstdev=self.maxStdev()
        a,b=self.successes,self.failures
        if self.successes<self.failures:
            a,b=b,a
        self.minvarBeta=Beta(a+self.d,b)
        self.minvarBeta.computeStats()
        mp.dps=max(25,7+int(-1*log(self.minvarBeta.stdev/self.an)))        
        lowerBound,upperBound=self.HPDB(self.lowerBeta.a,self.lowerBeta.b),self.HPDB(self.upperBeta.a,self.upperBeta.b)
        swappoint=1/(1+exp(sum([log(self.failures+1+k)-log(self.successes+1+k) for k in range(self.d)])/self.d))
        X,upperProb,XLow,maxY=[],[],[],1
        if ends:
            X.append(0)
            upperProb.append(self.lowerBeta.pdf(0))
        if self.successes>self.failures:
            maxY=self.upperBeta.pdf(self.upperBeta.mode)
        else:
            maxY=self.lowerBeta.pdf(self.lowerBeta.mode)
        minimum=0
        if not ends:
            minimum=maxY/1000
        ddydxdx=lambda x,y:derivatives(y,x,self.successes,self.d+self.failures,1,1)[1] #maxY*self.lowerBeta.stdev
        mini_h=min(1,self.lowerBeta.stdev)*self.h
        Xt,Yt=self.sampler(self.lowerBeta.pdf,ddydxdx,self.h,lowerBound,0,-1,lowerBound,mini_h,minimum)
        X.extend(reversed(Xt))
        upperProb.extend(reversed(Yt))
        lowerProb=[self.upperBeta.pdf(x) for x in X]
        XLow.extend(X)
        asamp=dda(self.successes,self.failures,self.d,maxY,self.maxstdev)
        Yt=self.sampler(asamp.f,asamp.ddydxdx,self.h,self.d,0)[1]
        X.extend(asamp.X)
        upperProb.extend(Yt)                 
        ddydxdx=lambda x,y:derivatives(y,x,self.successes+self.d,self.failures,1,1)[1] #maxY*self.upperBeta.stdev)
        mini_h=min(1,self.upperBeta.stdev)*self.h
        Xt,Yt=self.sampler(self.upperBeta.pdf,ddydxdx,self.h,swappoint,lowerBound,-1,swappoint,mini_h)
        XLow.extend(reversed(Xt))
        lowerProb.extend(reversed(Yt))                     
        ddydxdx=lambda x,y:derivatives(y,x,self.successes,self.d+self.failures,1,1)[1] #maxY*self.lowerBeta.stdev
        Xt,Yt=self.sampler(self.lowerBeta.pdf,ddydxdx,self.h,upperBound,swappoint,1,swappoint,mini_h)
        XLow.extend(Xt)
        lowerProb.extend(Yt)
        ddydxdx=lambda x,y:derivatives(y,x,self.successes+self.d,self.failures,1,1)[1] #maxY*self.upperBeta.stdev
        Xt,Yt=self.sampler(self.upperBeta.pdf,ddydxdx,self.h,1,upperBound,1,upperBound,mini_h,minimum)
        if ends:
            Xt.append(1)
            Yt.append(self.upperBeta.pdf(1))
        X.extend(Xt)
        upperProb.extend(Yt)
        lowerProb.extend([self.lowerBeta.pdf(x) for x in Xt])
        XLow.extend(Xt)
        if maxY>self.MAXPROB:
            upperProb=[y/maxY for y in upperProb]
            lowerProb=[y/maxY for y in lowerProb]
            print("MaxY="+str(maxY))         
        return X,upperProb,XLow,lowerProb        
    def plotBeta(self,a,b):
        beta=Beta(a,b)
        beta.computeStats()
        delta=self.h*beta.stdev
        mp.dps=max(25,7+int(-1*log(delta))) 
        maxY=beta.pdf(beta.mode)
        mini_h=min(1,beta.stdev)*self.h
        X=deque([])
        Prob=deque([])
        x=beta.mode
        toClose=True
        ddydxdx=lambda x,y:((a/x -b/(1-x))**2 -(a/(x*x) +b/((1-x)*(1-x))))*y/maxY
        while x>0:
            y=beta.pdf(x)
            X.appendleft(x)
            Prob.appendleft(y)
            slope=abs(ddydxdx(x,y))
            if toClose:
                xtmp=x-mini_h
                ytmp=beta.pdf(xtmp)
                slopetmp=abs(ddydxdx(xtmp,ytmp))
                if slopetmp>slope:
                    step=self.sqrt2*sqrt(self.h/slopetmp)

                    x-=max(mini_h,step)
                else:
                    toClose=False
                    step=self.sqrt2*sqrt(self.h/slope)
                    x-=max(mini_h,step)
            else:
                step=self.sqrt2*sqrt(self.h/slope)
                x-=step  
        X.appendleft(0)
        Prob.appendleft(beta.pdf(0))
        X,Prob=list(X)[:-1],list(Prob)[:-1] #so it doesnt have a duplicate points at x=beta.mode
        toClose=True
        x=beta.mode
        while x<1:
            y=beta.pdf(x)
            X.append(x)
            Prob.append(y)
            slope=abs(ddydxdx(x,y))
            if toClose:
                xtmp=x+mini_h
                ytmp=beta.pdf(xtmp)
                slopetmp=abs(ddydxdx(xtmp,ytmp))
                if slopetmp>slope:
                    step=self.sqrt2*sqrt(self.h/slopetmp)

                    x+=max(mini_h,step)
                else:
                    toClose=False
                    step=self.sqrt2*sqrt(self.h/slope)
                    x+=max(mini_h,step)
            else:
                step=self.sqrt2*sqrt(self.h/slope)
                x+=step                            
        X.append(1)
        Prob.append(beta.pdf(1)) 
        plt.plot(X,Prob,"k")
        plt.xlabel("Population Success Chance Probabilty")
        plt.ylabel("Probabilty")  
        plt.title("B(x;{0},{1})".format(a,b))
        plt.show()            
    def reasonablevaluesprint(self,region,parameter_name:str="",sig_digits:int=3,format:str="pm"):
        """
        region is a list or tuple in the following format [low1,upp1,low2,upp2,...,low_n,upp_n] which contains the reasonable values.
        parameter_name is the name of the parameter if not specifed is ommitted.
        sig_digits is the amount of digits after lower and upper disagree that is displayed.
        format is a string and can be "pm" "[]" or "wordy".
        "pm" reasonable values printed as x±y
        "[]" reasonable values printed as [low,upp]
        "wordy" reasonable values printed as "{low} to {upp}"
        """
        region=[float(r) for r in region]
        if format not in {"pm","[]","wordy"}:
            format="pm"
        print("Reasonable Beliefs about the {0} are".format(parameter_name.capitalize()),end="")
        output=""
        for low,upp in zip(region[::2],region[1::2]):
            agree_digits=0
            for l,u in zip(str(low),str(upp)):
                if l==u:
                    if not l==".":
                        agree_digits+=1
                else:
                    break
            if format=="pm":
                middle=(low+upp)/2
                diff=abs(low-upp)/2
                if agree_digits==0:
                    output+=" {0:#.{s}g} ± {0:#.{s}g} and".format(middle,diff,s=sig_digits)
                else:
                    output+=" {0:#.{n}g} ± {1:#.{s}g} and".format(middle,diff,n=agree_digits+sig_digits,s=sig_digits)
            elif format== "[]":
                low,upp=min(low),max(upp)
                output+=" [{0:#.{s}g},{1:#.{s}g}] and".format(low,upp,s=sig_digits+agree_digits)
            elif format== "wordy":
                output+=" from {0:#.{s}g} to {1:#.{s}g} and".format(low,upp,s=sig_digits+agree_digits)
        print(output[:-4])
    def stats(self,ends:bool=None,posteior:bool=True):
        """
        ends: If False cuts outs the regions with less than .1% relative likelyhood from the graph recommened when dealing with 10**6 or more datapoints
        """
        print("------------------------------------")
        print("Successes:{0}\tFailures{1}\tTotal:{2}".format(self.successes,self.failures,self.successes+self.failures))
        X,upperProb,XLow,lowerProb=self.computeGraph(ends)
        a,b=self.successes,self.failures
        if self.successes<self.failures:
            a,b=b,a
        minvarBeta=Beta(a+self.d,b)
        minvarBeta.computeStats()
        self.reasonablevaluesprint((self.lowerBeta.mean,self.upperBeta.mean),"mean") 
        self.reasonablevaluesprint((self.lowerBeta.mode,self.upperBeta.mode),"mode") 
        self.reasonablevaluesprint((minvarBeta.stdev,self.maxstdev),"standard deviation")
        # import pandas as pd
        # ends_=(ends in [True,None])
        # data={"x":[round(float(x),6) for x in X],"y":[round(float(p),6) for p in upperProb]}
        # df=pd.DataFrame(data)
        # df.to_csv("RBA {d} upper ends={ends} S&P500.csv".format(d=self.d,ends=ends_),sep=",",index=False)    
        # data={"x":[round(float(x),6) for x in XLow],"y":[round(float(p),6) for p in lowerProb]}    
        # df=pd.DataFrame(data)
        # df.to_csv("RBA {d} lower ends={ends} S&P500.csv".format(d=self.d,ends=ends_),sep=",",index=False)      
        print(len(set(X)))        
        print("{0} Calculated".format(["Priors","Posteiors"][posteior]))
        plt.plot(X,upperProb)
        plt.plot(XLow,lowerProb)
        plt.xlabel("Population Success Probabilty")
        plt.ylabel("Probabilty")
        plt.title("Reasonable Postetiors")
        plt.legend(["Upper Probabilty","Lower Probabilty"])  
        plt.show()   
        plt.scatter(X,upperProb)
        plt.plot(X,upperProb)     
        plt.show()   


class Demo(RobustBayesianAnalysis):
    def __init__(self,p,n,an):
        """p : Population Success Probabilty
           n : Degree of Prior Certitude of Beliefs
           an : Asymptotic number of points calculated (can be as low as 50 and still give good results)
        """              
        super().__init__(n,an)
        self.p=p
        self.rng=np.random.Generator(np.random.SFC64())    
    def getData(self,n):
        "perform n samples of the population"
        if n<self.MAXBINOMINALN:
            successes=self.rng.binomial(n,self.p)
        else:
            stdev=(sqrt(n*self.p*(1-self.p)))
            successes=round(np.random.normal(self.p*n,stdev))
        failures=n-successes
        self.successes+=successes
        self.failures+=failures
        print("------------------------------------")        
        print("Observed {0} Successes and {1} Failures".format(successes,failures))
        print("Total Successes {0} out of {1}".format(self.successes,self.successes+self.failures))        

if __name__=="__main__":                 
    Analysis=Demo(.5,2,125)  
    Analysis.getData(10**22)
    Analysis.stats(ends=False)         
    Analysis=Demo(.70,10,125)
    Analysis.stats(None,False)
    Analysis.getData(20)
    Analysis.stats(ends=True)
    Analysis.getData(980)    
    Analysis.stats(ends=True)
    Analysis.getData(10**6)
    Analysis.stats(ends=True)
    Analysis.getData(10**12)
    Analysis.stats(ends=False)   
    Analysis=Demo(.0,10,125)
    #Analysis.stats(None,False)
    Analysis.getData(20)
    Analysis.stats(ends=True)
    Analysis.getData(980)    
    Analysis.stats(ends=False)
    Analysis.getData(10**6)
    Analysis.stats(ends=False)
    Analysis.getData(10**20)
    Analysis.stats(ends=False)      
    Analysis=Demo(.50,1000,125)
    #Analysis.stats(None,False)
    Analysis.getData(20)
    Analysis.stats(ends=True)
    Analysis.getData(980)    
    Analysis.stats(ends=True)
    Analysis.getData(10**6)
    Analysis.stats(ends=True)
    Analysis.getData(10**14)
    Analysis.stats(ends=False)  
    del Analysis
    Analysis=Demo(.25,1000,1000)
    #Analysis.stats(None,False)
    Analysis.getData(20)
    Analysis.stats(ends=True)
    Analysis.getData(980)    
    Analysis.stats(ends=True)
    Analysis.getData(10**6)
    Analysis.stats(ends=True)
    Analysis.getData(10**12)
    Analysis.stats(ends=False)          
    #(1000,113) (100,35) (10000,338)