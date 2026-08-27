even = 0
odd = 0
for i in range(10000):
    if i % 2 == 0:
        even = even + 1
    else:
        odd = odd + 1
print(even, odd)
