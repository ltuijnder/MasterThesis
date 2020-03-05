# Goals Least Square 1
* written on 25/1/20

In this document I describe the short term goals for the master thesis regarding Least Square stuff. There are several things that we still need to figure out/check out. For this reason I will here describe different goals I set up for myself of what I want. 

This document may change over time for adding new information or updating the results. But large descriptions and results of the different goals will not be written here. They will get there own document. However I will also not add new big goals of least square these will be moved down into a new document.

The meaning of this document is to have a sort of milestone goal document, kind of like a check list. It is to serve as a nice overvieuw sucht that I can see what is left to do and why I wanted to do stuff. So now that this disclaimer is said lets start describing the goals that I want.

## Generation data
The script that is now being used was cobbeld togheter from that of lana (which is ofcourse not bad) but now we are a bit wiser and further into the analysis and I think it time to have again a fresh start and so also develop a new function for generating the data.

I want to have the following bullet points checked of:

* Have a notebook that illustrates how the data is generated with comments. But let the eventual function be defined in a .py script such that it can be used by other notebooks.
* I want to have two modes. First is the generation mode before scaling and centering and seconds is the generation after scaling and centering.
* Have in the notebook a few rudementary scripts of viewing the data. -> See what is being linear fitted aka plot the log diff. 
* Be able to generate reference data for simple regression testing

This should be able to be done in a day. Unless we maybe go for something fancy In where I generate the data using classes. Important is that the code should be scalable, easy to read and adaptable!. Since we probabily want to change it later on!

## Implement basic regression

Here I just want again a basic Linear regression model. I also want this to be scalable and easily adjustable. So maybe I will do this in class way. Such that code is later clean later on when doing the data analyis. 

We could also incorporate already existing libraries this way with ease. We can easily then also incorporate gradient descent. Basically have a kind of optimize class where you just in the beginning specify how you want to optimize.

I then also want some methods that nicely reproduce the results we want -> Have fitted results graph (look at the generation data). Try and think on how to implement the p-value visualistation. And think of other ways that we can visualise the fit result.

## Experiment: centering and scaling.

In the last few days I did some analytical calculations on centering and scaling. Now I also want to proof my results with real data! For this I want to have the following two experiments

### Experiment on centering

The main result was here that centering altered the meaning of the coefficients and the meaning that we are searching for is that of centered data. To prove this that this is indeed the case we will first need to do it some test generation data and look at fit results implemented by the methods above. 

Things to consider during this experiment:
* how does the generation of the data influance my result. -> Try and work with different data sets and if stochatic effects are pressents do it multiple times.
* Different fit techniques are possible (Analytical, gradient decent, k-fold,...) Pick out different and compare the perfomance on this result. -> repeat the experiment multiple times if stochatic effects are pressent
* Compare methods on the same data set! -> The comparison can ofcourse be done on multiple instance of the data set. 
* Try to identify the goal of what we want to prove! Can we formulate this as statistical test?
* Try and test results on the real data set! And see its influance

### Experiment on scaling

The result that we got here was that the scaling of the data DID not matter. Lets just have a simple experiment that verifies this.

Things I want to have experimented:
* Try and verify this result on dummy/reference data set we controll. And species data set!.
* Have a p-value test between the p-values fit results before and after scaling!. 
* Repeat the experiment multiple times -> Multiple instances of the data set.
* Just pick one regression method. (analytical solution should be okay)


## Experiment: Noise controll!

## Experiment: do higher orders matter?

## Experiment: Want on the real life data set? 

## Ideas for future Least Square regression

Here I describe some idea goals that do not belong here in Goals Least Square 1 but in some other future goals consideration (Maybe Least Square 2)

* Can we still perform analytical solution when we add saturation terms? 
* Can we let saturation terms incorparte centering? Or does centering fundemntally alter the equations then? 
* Can we find an basis expansion that simply incorporate the saturations? If not can we maybe find an approximation? 