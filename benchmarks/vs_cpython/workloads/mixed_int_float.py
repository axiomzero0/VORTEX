isum = 0
fprod = 1.0
i = 1
while i <= 1000:
    isum = isum + i
    fprod = fprod * 1.001
    i = i + 1
print(isum)
print(fprod > 2.0 and fprod < 3.0)
