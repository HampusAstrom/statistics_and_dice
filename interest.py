import matplotlib.pyplot as plt

iterations = 26

year = [0] * iterations
interest = [0] * iterations
loan_size = [0] * iterations
interest_paid = [0] * iterations


installment_fraction = 0.04
interest_inc_per_year = 0.005

year[0] = 0
interest[0] = 0.014
loan_size[0] = 1910000 - 550000
interest_paid[0] = 0 # intererest is paid for previous year

installment_per_year = [loan_size[0] * installment_fraction] * iterations

for i in range(1, iterations):
    year[i] = year[i - 1] + 1
    interest_paid[i] = loan_size[i - 1] * interest[i - 1]
    loan_size[i] = loan_size[i - 1] - installment_per_year[i]
    interest[i] = min(interest[i - 1] + interest_inc_per_year, 0.07)

plt.plot(year, interest_paid)
plt.plot(year, installment_per_year)
plt.show()

plt.plot(year, loan_size)
plt.show()

plt.plot(year, interest)
plt.show()
