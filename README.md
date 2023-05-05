Download Link: https://assignmentchef.com/product/solved-cse512-hw3-nearest-neighbor-classifiers
<br>
1-NN with asymmetric loss

Suppose we want to build a binary classifier, but the cost of false positive (predicting positive for a negative case) is much higher than the cost of false negative (predicting negative for a positive case). One can consider an asymmetric loss function, where the cost of false negative is 1, while the cost of false positive is <em>α &gt; </em>1. For a point <em>x</em>, let <em>η</em>(<em>x</em>) be the probability that <em>x </em>is positive.

Show that the optimal Bayes risk for data point <em>x </em>is <em>r</em><sup>∗</sup>(<em>x</em>) = min{<em>η</em>(<em>x</em>)<em>,α</em>(1 − <em>η</em>(<em>x</em>))}.

Let <em>r</em>(<em>x</em>) be the asymptotic risk for data point <em>x</em>, express <em>r</em>(<em>x</em>) in terms of <em>α </em>and <em>η</em>(<em>x</em>).

Prove that <em>r</em>(<em>x</em>) ≤ (1 + <em>α</em>)<em>r</em><sup>∗</sup>(<em>x</em>)(1 − <em>r</em><sup>∗</sup>(<em>x</em>)). Hint: use <em>α &gt; </em>1

Let <em>R </em>be the asymptotic risk of the 1-NN classifier and <em>R</em><sup>∗ </sup>be Bayes risk. Prove that: <em>R </em>≤ (1+<em>α</em>)<em>R</em><sup>∗</sup>(1−<em>R</em><sup>∗</sup>)

1.2 <em>k</em>-NN classifier (20 points)

Consider a <em>k</em>-NN classifier: classify a point as positive if at least (k+1)/2 nearest neighbors are positive.

1.2.1

Consider drawing <em>k </em>points randomly from a Bernoulli distribution with two outcomes: positive or negative, and the probability of the point being positive is <em>η</em>. Let <em>g</em>(<em>η,k</em>) be the probability that at least (<em>k </em>+ 1)<em>/</em>2 out of <em>k </em>points are positive. Express the asymptotic risk <em>r</em>(<em>x</em>) for a point <em>x </em>in terms of <em>η</em>(<em>x</em>) and the function <em>g</em>(·<em>,</em>·).

1.2.2

Prove that <em>r</em>(<em>x</em>) = <em>r</em><sup>∗</sup>(<em>x</em>) + (1 − 2<em>r</em><sup>∗</sup>(<em>x</em>))<em>g</em>(<em>r</em><sup>∗</sup>(<em>x</em>)<em>,k</em>)

1.2.3

Using Hoeffding’s Inequality (https://en.wikipedia.org/wiki/Hoeffding_inequality),

<table width="0">

 <tbody>

  <tr>

   <td width="610">prove that:</td>

   <td width="17"> </td>

  </tr>

  <tr>

   <td width="610"><em>g</em>(<em>r</em><sup>∗</sup>(<em>x</em>)<em>,k</em>) ≤ exp(−2(0<em>.</em>5 − <em>r</em><sup>∗</sup>(<em>x</em>))<sup>2</sup><em>k</em>)</td>

   <td width="17">(1)</td>

  </tr>

 </tbody>

</table>

1.2.4     (4 points)

Prove that:. Hint: you should use the above inequality Eq. (1). Note that: from this result, you can see that the Asymptotic risk of <em>k</em>-NN classifier is the Bayes Risk if <em>k </em>goes to infinity.

2 Question 2 – Implementation of Logistic Regression Classifier for <em>k </em>classes (60 points + 10 bonus)

In this Question, you will implement Logistic Regression using Stochastic Gradient Descent (SGD) optimization. Suppose the training data is {(<em>X</em><sup>1</sup><em>,Y </em><sup>1</sup>)<em>,</em>··· <em>,</em>(<em>X<sup>n</sup>,Y <sup>n</sup></em>)}, where <em>X<sup>i </sup></em>is a column vector of <em>d </em>dimensions and <em>Y <sup>i </sup></em>is the target label. For a column vector <em>X</em>, let <em><sup>X</sup></em><sup>¯ </sup>denotes [<em>X</em>;1], the vector obtained by appending 1 to the end of <em>X</em>. <em>θ </em>is the set of parameters <em>θ</em><sub>1</sub><em>,θ</em><sub>2</sub><em>,…,θ</em><em><sub>k</sub></em><sub>−1</sub>. Logistic regression for <em>k </em>classes assumes the following probability function:

<em>,                       </em>

Logistic regression minimizes the average conditional log likelihood:

<em>.                                                                 </em>(4)

To minimize this loss function, we can use gradient descent:

(5)

where(6)

This gradient is computed by enumerating over all training data. It turns out that this gradient can be approximated using a batch of training data. Suppose B is a subset of {1<em>,</em>2<em>,</em>··· <em>,n</em>}

(7)

This leads to the following stochastic gradient descent algorithm:

Algorithm 1 Stochastic gradient descent for Logistic Regression

1: Inputs:  (for data), <em>m </em>(for batch size), <em>η</em><sub>0</sub><em>,η</em><sub>1 </sub>(for step size), <em>max epoch</em>, <em>δ </em>(stopping criteria)

2: for <em>epoch </em>= 1<em>,</em>2<em>,…,max </em><em>epoch </em>do

3:             <em>η </em>← <em>η</em><sub>0</sub><em>/</em>(<em>η</em><sub>1 </sub>+ <em>epoch</em>)

4:                  (<em>i</em><sub>1</sub><em>,…,i<sub>n</sub></em>) = <em>permute</em>(1<em>,…,n</em>)

5:          Divide (<em>i</em><sub>1</sub><em>,…,i<sub>n</sub></em>) into batches of size <em>m </em>or <em>m </em>+ 1 6: for each batch B do

7:                                  Update <em>θ </em>using Eqs. (5) and (7)

8: Break if <em>L</em>(<em>θ</em><em><sup>new</sup></em>) <em>&gt; </em>(1 − <em>δ</em>)<em>L</em>(<em>θ</em><em><sup>old</sup></em>) // not much progress, terminate 9: Outputs: <em>θ</em>.

2.1     Derivation (10 points)

Prove that:

<em>.                                               </em>(8)

where <em>δ</em>(<em>c </em>= <em>Y <sup>i</sup></em>) is the indicator function which takes a value of 1 if the class <em>c </em>equals the ground truth label <em>Y <sup>i</sup></em>, and 0 otherwise. Use Equation (8) to derive the gradient of the loss with respect to the parameters <em>θ</em><sub>1</sub><em>,θ</em><sub>2</sub><em>,…,θ</em><em><sub>k</sub></em>−<sub>1</sub>.

2.2      Crowd Image Classification

In this question of the homework, you will work with image data. We are providing you with the features extracted from the crowd images, so you do not need to extract features from the raw images. We are also providing you with the raw images for error analysis and visualization purposes. Your task is to classify an image into 4 different categories. The data has already been split into train, validation, and test sets.

Dataset details (obtain data from Kaggle competition page)

—- train set (4000 images)

—- val set (2000 images)

—- test set (2000 images)

2.3        Implement Logistic Regression with SGD (50 points + 10 bonus)

Your task is to implement Logistic Regression with <em>k </em>= 4 classes using SGD. You should use Python or Matlab for your implementation.

<ol>

 <li>(15 points) Run your implementation on the provided training data with <em>max epoch </em>= 1000<em>,m </em>= 16<em>,η</em><sub>0 </sub>= 0<em>.</em>1<em>,η</em><sub>1 </sub>= 1<em>,δ </em>= 0<em>.</em>00001.

  <ul>

   <li>Report the number of epochs that your algorithm takes before exiting.</li>

   <li>Plot the curve showing <em>L</em>(<em>θ</em>) as a function of <em>epoch</em>.</li>

   <li>What is the final value of <em>L</em>(<em>θ</em>) after the optimization?</li>

  </ul></li>

 <li>(10 points) Keep <em>m </em>= 16<em>,δ </em>= 0<em>.</em>00001, experiment with different values of <em>η</em><sub>0 </sub>and <em>η</em><sub>1</sub>. Can you find a pair of parameters (<em>η</em><sub>0</sub><em>,η</em><sub>1</sub>) that leads to faster convergence?

  <ul>

   <li>Report the values of (<em>η</em><sub>0</sub><em>,η</em><sub>1</sub>). How many epochs does it take? What is the final value of <em>L</em>(<em>θ</em>)?</li>

   <li>Plot the curve showing <em>L</em>(<em>θ</em>) as a function of <em>epoch</em>.</li>

  </ul></li>

 <li>(10 points) Evaluate the performance on validation data

  <ul>

   <li>Plot <em>L</em>(<em>θ</em>) as a function of <em>epoch</em>. On the same plot, show two curves, one for training and one for validation data.</li>

   <li>Plot the accuracy as a function of <em>epoch</em>. On the same plot, show two curves, one for training and one for validation data.</li>

  </ul></li>

 <li>(5 points) With the learned classifier:</li>

</ol>

(a) Report the confusion matrices on the validation and the training data