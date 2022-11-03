import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'VerificationType':2,'Amount':9,'Interest':6,'LoanDuration':2,'MonthlyPayment':3,'UseOfLoan':5,'EmploymentStatus':6,'IncomeTotal':2,'PlannedInterestTillDate':5,'PrincipalPaymentsMade':3,'InterestAndPenaltyBalance':2,'AmountOfPreviousLoansBeforeLoan':4,'NrOfScheduledPayments':2})

print(r.json())