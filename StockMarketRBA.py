from RobustBayesianAnalysis import *
import pandas as pd
data=pd.read_csv("SPX.csv",parse_dates=["Date"])
data['Date']=pd.to_datetime(data["Date"],format='%Y-%m-%d')
data30yr=data.loc[data["Date"]>="1990-11-03"]#&(data["Date"]<="1940-11-04")]
if __name__=="__main__":
    print("dependecies loaded")
n=len(data30yr)
Closings=data30yr["Close"]
successes=sum([1 for o,c in zip(Closings[:-1],Closings[1:]) if c>=o])
failures=n-successes
def save(RBAVars:dict,addlow=None,addhigh=None):
    if addlow!=None:
        RBAVars["X"].insert(0,addlow)
        RBAVars["XLow"].insert(0,addlow)
        RBAVars['upperProb'].insert(0,RBAVars["self"].lowerBeta.pdf(addlow))
        RBAVars['lowerProb'].insert(0,RBAVars["self"].upperBeta.pdf(addlow))
    if addhigh!=None:
        RBAVars["X"].append(addhigh)
        RBAVars["XLow"].append(addhigh)
        RBAVars['upperProb'].append(RBAVars["self"].upperBeta.pdf(addhigh))
        RBAVars['lowerProb'].append(RBAVars["self"].lowerBeta.pdf(addhigh))      
    ends_=(RBAVars['ends'] in [True,None])
    d=RBAVars['self'].d
    data={"x":[round(float(x),6) for x in RBAVars['X']],"y":[round(float(p),6) for p in RBAVars['upperProb']]}
    xlow,xhigh=data["x"][0],data["x"][-1]
    df=pd.DataFrame(data)
    df.to_csv("RBA {d} upper ends={ends} S&P500.csv".format(d=d,ends=ends_),sep=",",index=False)    
    data={"x":[round(float(x),6) for x in RBAVars['XLow']],"y":[round(float(p),6) for p in RBAVars['lowerProb']]}    
    xlow,xhigh=min(xlow,data["x"][0]),max(xhigh,data["x"][-1])
    df=pd.DataFrame(data)
    df.to_csv("RBA {d} lower ends={ends} S&P500.csv".format(d=d,ends=ends_),sep=",",index=False)          
    return xlow,xhigh
RBA=RobustBayesianAnalysis(100,80,successes,failures)
RBA.stats(True,True,save)
RBA=RobustBayesianAnalysis(100,80,successes,failures)
xlow,xhigh=RBA.stats(False,True,save)
RBA=RobustBayesianAnalysis(30,80,successes,failures)
RBA.stats(True,True,save)
RBA=RobustBayesianAnalysis(30,80,successes,failures)
RBA.stats(False,True,lambda v:save(v,xlow,xhigh))


