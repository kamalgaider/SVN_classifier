# SVN_classifier
Classification based on Support Vector Machine

we require two functions for training- Loss function & Objective function


The LOSS FUNCTION we use here is 'Hinge loss'. It is used for training classifiers 
for maximum margin classification(max seperation of classes).

c(x,y,f(x)) = (1 - y * f(x))_

where 	c is loss function
		x is input point
		y is actual label
		f(x) is predicted label
		_ means that loss is 0, if  y * f(x) >= 1
		
		(FYI : for our example y and f(x) can only be 1 or -1
		& loss is 0, if y == f(x))



The OBJECTIVE FUNCTION of SVM  has 2 terms : regularizer(λ - heart of SVM)
& Summation of loss. Regularizer balances between margin maximization 
& loss. 
λ too high - overfit. 
λ too low - underfit.

minimize:
				  λ ||w||²   +   Σ (1-y(x,w))_

To optimize our Objective function we need to minimize loss.
To do that we have to calculate the gradient descent.
Gradient descent is the derivative of function based on w. 

so the regularizer term becomes : 2λw
& loss becomes : 0  if y(x,w) >= 1      else:  -yx

And using this gradient, we update the weight after each prediction

For misclassification ( y(x,w) < 1) ,  update weight as:
w = w + η(yx-2λw)
	where η is learning rate
	η too high - algo may overshoot optimal point
	η too low - algo may take too long(or never) to converge
For correct classification:
w = w + η(-2λw)
