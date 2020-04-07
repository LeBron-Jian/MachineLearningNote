from math import log

# 初始熵值
Entropy_base = -(9 / 14) * log(9 / 14, 2) - (5 / 14) * log(5 / 14, 2)
# outlook=sunny
Entropy_Outlook_sunny = -(2 / 5) * log(2 / 5, 2) - (3 / 5) * log(3 / 5, 2)
# outlook=overcost
Entropy_Outlook_overcost = -(4 / 4) * log(4 / 4, 2)
# outlook=rainy
Entropy_Outlook_rainy = -(3 / 5) * log(3 / 5, 2) - (2 / 5) * log(2 / 5, 2)
print(Entropy_Outlook_sunny, Entropy_Outlook_overcost, Entropy_Outlook_rainy)
# 0.9709505944546686 -0.0 0.9709505944546686
Entropy_Outlook = (5 / 14) * Entropy_Outlook_sunny + (4 / 14) * Entropy_Outlook_overcost + (
        5 / 14) * Entropy_Outlook_rainy
print('Entropy_Outlook:', Entropy_Outlook)
Entropy_test = -(1 / 2) * log(1 / 2, 2) - (1 / 2) * log(1 / 2, 2)
Entropy_test1 = -(1 / 3) * log(1 / 3, 2) - (1 / 3) * log(1 / 3, 2) - (1 / 3) * log(1 / 3, 2)
Entropy_test2 = -(1 / 4) * log(1 / 4, 2) - (1 / 4) * log(1 / 4, 2) - (1 / 4) * log(1 / 4, 2) - (1 / 4) * log(1 / 4, 2)
print(Entropy_test, Entropy_test1, Entropy_test2)
# 1.0     1.584962500721156     2.0



print((1/3)*log(Entropy_test1, 2))
print((1/4)*log(Entropy_test2, 2))
