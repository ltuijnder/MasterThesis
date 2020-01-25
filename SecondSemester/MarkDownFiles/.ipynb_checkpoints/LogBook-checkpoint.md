# LogBook Masterthesis

The purpose of this logbook is as follows:

* Have a history of what is done in with your master thesis such that you can remember it later
* Have a way of looking back at different references. Such that you know what you used and when. *Actually there is now a seperate markdown file for this!*
* Have proof that your work is original with this logbook. Good for protecting Intelectual Property (IP)
* Easier way of navigating through your old stuff. Since I will also be writting what has been written where.

I only started this logbook in the beginning of the second semester. Results of the first are a bit more difficult to get by. :/ 

Important results of what has been done before was mainly the oriantation and getting familiar with the stuff. Some redementary linear regression was first done. This is mostely it.

# Januari 2020

## 22/1/20

This day I mainly did some analytical stuff and reoriantation since I had not worked at the thesis for more then 1.5 months!

Main work is done on the notes and these are labeled as follows:

1. Redoing the fundamental analysis of the GLV starting from the exponential expression. Here it was important to see what is the link between the exponential expression and the differential one which is most often taken as the true definiation. Another important results was to see the importance of the $dlog/dt$ limit! And realise that this is the reason why splines are used in the orignal paper.

2. This mostely me discussing on why we should center and scaling the data based on the opinion of some stack exchange post -> see reference.md 

3. Failed attempt to do the least square estimation with a general kernel expansion. The $(X'X)^{-1}$ make is it brutally hard to consider general cases! This day ended in giving up the calculation.

### General conlusion

* The reason on why spline is done is to have a good estimation on the derivative!. If your data oscillates quicker then the timeresolution your estimate of $Y$ would be bad and thus your Least Square would be totally of.

* Scaling and centering is a hot debated topic

* Considering the scaling and centering for a general basis expansion turned out to be more challenging then first thought.

## 23/1/20

On this day I continued analytical results with now the determination to calculate some scaling and centering effects for simple cases and not the most general basis expansion. 

Again the results are mainly on paper and what has been done is labeld as follows.

1. In this first section I argue on why we should first **center** our data! The reason being is that when fitting a model behaviour of coefficient in a polynomial expression is what is the derivative/contribution of that order **AT $X$=0**. I argued then for our intiuative feeling of the problem we should centre our data. Else we cannot view the higher orders as correction. This can also be easiliy seen with taylor expansions.

2. I explicitly did the scaling computation for the model $Y= \beta_0 + \beta_1x +\beta_2 x^2$. The results was as expected and the calculation went quickly because I first centered the data.

3. I now explicitly calcualte the scaling computation for the model  $Y= \beta_0 + \beta_1x_1 +\beta_2 x_2$. The results was also as expected and I also first scaled the data. 

4. After learning how the scaling worked in the two models above I generalised even further to: $Y= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta-4 x_2^2$ Here I dropped even the centering condition, but I only did the calculation in orders of scaling factor since that is what I was interested in. For this I introduced costum notation that made the calculations go quicker.

5. After what I have learned from the more general model and seeing how there was symetry in the scaling factors I step it up and I proved the scaling invariance on the test statistic for the most general scaling case. 

6. Scaling on the effect of scaling $x$ does not effect the log difference and hence in our calculation it does not matter when it is done. When $Y$ would scale this is would change the coefficients in a trivial way.

### General conclusion

* We first need to centre our data such that the interpretation of the interaction terms are that what we expected! 

* I proved the scale invariance of the test statictic in the most general case! 

We should however becarefull in the case of ridgeregression since there the interpretation of the terms again changes ! And thus is not scale inviariant!.

## 24/1/20

On this I did nothing really directly towards the master thesis itself but I learned the platform JupyterLab and started playing around with it. This is the IDE I'm planning to use for my master thesis in the future. 

I also am again reading in the bayesian inference book since I want to maybe start doing bayesian inference on the data. 

## 25/1/20



