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
RBA=RobustBayesianAnalysis(100,80,successes,failures)
RBA.stats()
RBA=RobustBayesianAnalysis(100,80,successes,failures)
RBA.stats(False)
RBA=RobustBayesianAnalysis(30,80,successes,failures)
RBA.stats()
RBA=RobustBayesianAnalysis(30,80,successes,failures)
RBA.stats(False)

