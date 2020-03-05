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

Total hours: 8 (ish)

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

Total hours: 8 (ish)


## 24/1/20

On this I did nothing really directly towards the master thesis itself but I learned the platform JupyterLab and started playing around with it. This is the IDE I'm planning to use for my master thesis in the future. 

I also am again reading in the bayesian inference book since I want to maybe start doing bayesian inference on the data. 

Total hours: a couple

## 25/1/20

The creation of these logbooks and the References and a thinking about the GoalLeastSquare.

Pushed then also these changes onto github

Total hours: 3 

## 27/1/20

In the last hour of the day I thinked about how to implement the generation and fitting of the data.

Total hours: 1

## 28/1/20

Implemented DataGen.ipynb. In this notebook I started on working the class structure "Data" from which I thought I would derive everything. 

I succeeded in making the code, and made something very general that could generate data based on previous data. 

Total hours: 6 (ish)

## 29/1/20

On this day I realised on what I created in "DataGen.ipynb" was to general and actually not suitable for time generations as I first thought. This gave me a mild discouragment on to work any further that day.

Later that evening I worked a small hour or so on a new scheme on how I wanted to construct the timeseries generation.

Total hours: 2 

## 30/1/20

On this day with now a more fresh mind I started working further out on how I would want to tackle the timegeneration. I have thought out on how to do the time generation and I also have thought about how to do the Least Square fit. 

In general I also realised that I should not just start coding but first have a solid idea of what I want to code such that I code effectifly and do not waste a few days like just happend. 

This day I didn't work that much (couple of hours) more like really thinking that actually doing stuff

Total hours: 2

## 31/1/20

I started coding the timeseries code. I just wanted to do this in the morning because in the afternoon I wanted to work on my assistent job. So I endedup working 3 hours on it.

I managed to code the parent class _Timeseries_, which is capable of generating a timeseries for any general set of differential equations. This was tested with an example of the harmonic oscillator which worked fine.

I then ran into a problem of for the General Lotka Volterra (GLV) on how to code this. Since generating the data centered or uncenterd results in different model parameters. I didn't think through on how I wanted to manage the different model parameters for just GLV. 

Do I want to generate a seperate class for this? Or do I want to manage it with flags, which is ok if it just stays to these two extensions but if it is going to be more then it becomes cumbersome. On the other hand putting it different classes results in copys of the same code. -> So maybe again doing an inherentance is the solution, but I should think about that again. And I should first more think about it analytically on what is really the difference between the two on paper. And also immediatly just think about the higher orders.

Total hours: 3 

# February 2020

## 5/2/20

I first updated the logbook, since I forgot a few days before.

At 10am I had a discussion with prof. Sophie on the standing of the thesis and discussion on PHD. This is what I got out of it:

In the meeting it was first discussed on what I have done. Here I presented the main result that scaling for a general way does not effect the z-component. And I presented the idea I had with centering. 

After a discussion (I should become better at presenting my result), we came indeed to the conclusion that we should be carefull on which model we want to interpret and think about which p_value. 

Further in the discussion it was clear that I should first just focus on GLV and for the moment forget higher order terms and just write down the result in a latex document (not polished) everything about just the GLV. Such that we have a foundation to work from. 

Things I need to explain and explore:
* Explore the p-value of the GLV for different settings:
    * First just use every time point
    * Then do pertubations in the timeseries to try and kick it of.
    * Explore different time resolutions, (with perfect dlog/dt and without)
    * Explore it for different kind of noise levels. 
    
From this we explore in which regions of noise it is possible to infer. How many time points we need and how good the time resolutions needs to be. -> We want to ask what is the condition inorder to have good inference!

Next we need to just apply the this GLV inference to Davids data and Stijns data. -> See what we can say about the confidence (think about results. Is overfitting here an issue?)

These are the short term results. From there on we build and go to different methods.

Total hours: 3 

## 12/2/20

Creation GLV.ipynb and just and started at the timeseries generation for GLV but imperially. This will probably not be continued.

Total hours: 0.5

## 13/2/20

On this day I prepared the conversation for the meeting in the after noon. A general feeling I had from last meeting was that we loosed a bit track of what has been done and that we needed to organize it and start to write down. 

So as prepartion for this meeting I went back through everything I have done to write down on a few pages of what concretely I have done. I organised this with main bullet points and also wrote of what further has to be done. 

At 2pm the meeting happend and I reviewed of what I had to say. The main part was of me going over the bullet points. And the general conclusion of the meeting was that I first need to focuse of finisching GLV and HGLV. 

But one of the first things that I need to figure out is of how we we are going to generate the data and which model we are going to use. And of what model we want to interpret. What the meaning of the different parameters. For this I think again my class generation of what I was doing is the best option, in order to consider of what we want to do -> giving the most flexibility. 

We should also ask our self what can we controll in the generation of the data what might influence the results and what can we now for sure does not. 


