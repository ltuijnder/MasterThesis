x = ts[:,0:2]
print(InterM.dot(x)*x)
for k in range(2):
    print(" ")
    for i in range(5):
        sum=0
        for j in range(5):
            sum+=InterM[i,j]*x[j,k]
        print(sum)