---
**<img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> CS109A Introduction to Data Science**
**<img style="float: right; padding-right: 10px; width: 50px" src="https://i.imgur.com/2ptDvXd.png">**
---


## Contents
{:.no_toc}
*  
{: toc}


**Harvard University**<br/>
**CSCI E-109A**<br/>
**Fall 2018**<br/>

**Group #47 Team Members**:<br/> 
Victor Chen<br/>
Danielle Crumley<br/>
Mohamed Elsalmi<br/>
Hoon Kang<br/>


<hr style="height:1.5pt">

## <font color='maroon'>Background Information</font>

The Lending Club is a peer-to-peer lending network for loans ranging from \$1,000 to \$40,000. The Lending Club provides a large amount of loan data online so that its investors can make their own informed investment decisions. This online data dates from 2007 to Q2 2018 and provides certain borrower demographic and credit history information, as well as loan-specific information (including the purpose of the loan, the amount requested and amount funded, the interest rate, and the loan grade assigned by the Lending Club).

## <font color='maroon'>Project Aim</font>

The main aim of this project is to develop a model that will predict whether a Lending Club-approved loan will end up being fully paid or charged off. These predictions, of course, would be very useful to investors in determining whether or not to invest. We aim to develop a model that is both 1) accurate/effective in producing a useful prediction for investors and 2) non-discriminatory with regard to demographic features such as race and home address.

We have chosen to include only completed loans in our model (i.e., none that are still in progress, whether current or in default, since we don’t know the final outcome of these loans).


## <font color='maroon'>Data</font>


We used the aforementioned data that is available for download from the Lending Club website.

Unfortunately, the Lending Club data for rejected loans is much more limited in scope: it includes only a few feature columns such as risk score, debt-to-income ratio, zip code, and length of employment. Since there is too much missing data, our final model will be built using only the data from the loans that were approved by the Lending Club.

First, we downloaded all of the available data on funded loans, which dates from 2007 to Q3 of 2018. It includes 145 columns and over 2 million rows.  We did not use the lending club’s data on rejected loans, since these datasets have much less information for each borrower/loan. 

Note that we were not able to incorporate borrower FICO scores into our model, since the Lending club restricts access to this information to its approved investors. However, there are many other credit risk-related features in the data, including many variables that come from the credit report such as the number of delinquencies in the borrower’s credit file in the past 2 years, the average current balance on all accounts, and the total credit revolving balance. Additionally, the lending club assigns its own loan grade (and loan subgrade) based on the FICO score and other variables, and we do have access to the Lending Club’s loan grades.

## <font color='maroon'>Methodology</font>



## <font color='maroon'>Results</font>


## <font color='maroon'>Sources</font>
Data Source: The Lending Club, https://www.lendingclub.com/


## <font color='maroon'>Discussion</font>

As Lending Club is a peer-to-peer lending platform, the investment approaches between investors are likely to vary with the intent and level of risk aversion of each investor. The only common thread between investors will most likely be the desire to maximize profits. However, bad loans that are never fully paid off equating to financial loss are unavoidable in this business.
As such, for us to provide the best model for a given investor, the investor must first decide for themselves what an acceptable level of loss is. That means out of the total number of loans the investor funds, what percentage of those loans is the investor comfortable with being a charged off loan. A new investor may only willing to accept a 1% charged off rate while a veteran investor may know that it is still profitable even at a 10% charged off rate. 
This maximum threshold of loss will help us select the best model for the investor by first selecting only models that are within the acceptable level of loss and then selecting the model that maximizes the number of fully paid off funded loans (which acts as a proxy metric to overall profit). 
To mimic this process in our modeling, we will set the positive case in our target variable to be a fully paid off loan while the negative case will be set as a charged off loan. In this scenario, precision now represents the number of funded loans that are fully paid out of the total number of loans funded (funded loans that are fully paid + funded loans that are charged off). The investor’s accepted percentage of charged off loans is represented by 1 – precision. In our tests we will assume the investor is willing to take a 10% charged off loan rate. This means we will look for models that have a minimum precision of 0.90 and then maximize the true positive count (funded loans that are fully paid) as an attempt to maximize profit. 



We will use model probability scores to order and rank the list of recommended loans. The higher the probability of a loan being a fully paid loan, the higher on the list it will rank. It is unrealistic to expect that investors will have the capital to invest in all recommend loans. This means we care more about better model performance at higher probability values than we do about model performance near the default 0.50 threshold simply because the loans at higher probability values will have higher priority to be fulfilled first. In deciding our final model, we will move the threshold to higher values and compare performance. 


Further Work
	Investors will be willing to accept differing levels of risk and loss. There may be an optimal percentage maximizing profit by combining in factors such as profit from fully paid loans, loss from charged off loans, differing loan amounts and interest rates, and opportunity cost.


Probability calibrations?

