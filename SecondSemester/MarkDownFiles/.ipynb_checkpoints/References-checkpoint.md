# References

Comprehensif overview of the different references that have been used throughout the master thesis. These are sorted by day and give a description on what information was gathered from it. 

# Januari 2020

## 22/1/20

### Discussions on when to centre and normalise the data.
https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia/111997
In this post:

First answer argues on why it should be done. It gives intuitif explanation but doesn't really discuss performance

Seconds answer talks more about performance. But I think what this person is more trying to say is to not trust people saying that variance corresponds to importance -> Not true. However it also says that centring and scaling as no influance this however I find sceptical since many others point to the opposite! Maybe she is just talking about the normal linear regression.

Third post: This one I like to most here He really goes over on how to ask the influance on the infered data. Important here is to look at how transformation effect the ESTIMATER and from this one can see how this effects results!

### Another post discussing this
https://stats.stackexchange.com/questions/201909/when-to-normalize-data-in-regression
Actually here it is very interesting the person that answer argues that when the estimator is invariant under scaling (like simple linear)
then it doesn't matter and so one could also just compute the estimator on the unscaled data since it would be simular. However this is not the case for estimators that or not invariant! (like ridge regression) -> then scaling has effect! The person then argues to use scaling and centering on those types!
But doesn't really provide any reference or something.

### Collinearity problem for non centered data:
https://stats.stackexchange.com/questions/60476/collinearity-diagnostics-problematic-only-when-the-interaction-term-is-included#61022
(this is the case for independed X Y) but the argument can also be made for polynomial terms..
Also the second post in this stack exchange has references to papers!!


## 23/1/20

### Scaling and centring invariant
Scaling and centering does not effect normal multivariate However it does not apply to regression:
https://roamanalytics.com/2016/11/17/translation-and-scaling-invariance-in-regression-models/

### Very interesting approach to polynomial regression:
http://home.iitk.ac.in/~shalab/regression/Chapter12-Regression-PolynomialRegression.pdf
Here they define how to do polynomial regression in with polynomials that form an orthogonal basis! Then X'X becomes diagonal matrix!! very clever.
In here they actually consider on how to very effectively do polynomial regressions!


