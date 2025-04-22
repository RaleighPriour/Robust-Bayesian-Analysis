from RobustBayesianAnalysis import *
RBA=RobustBayesianAnalysis(4,50,0,0)
RBA.failures=10**6
RBA.stats()
RBA.failures=10**12
RBA.stats()