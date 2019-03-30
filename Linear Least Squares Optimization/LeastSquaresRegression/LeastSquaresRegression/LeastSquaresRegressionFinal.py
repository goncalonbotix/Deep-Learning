import sys
import numpy as np
import matplotlib.pyplot as plt
def main():
    #d = 2*5.2 + 4 * 6.7 + 6 * 9.1 + 8 * 10.9
    #print(d)
    #e = 2*5.2 + 2*6.7 +2*9.1 + 2*10.9
    #print(e)
    x = np.ndarray((6,1))
    y = np.ndarray((6,1))
    x[0,0] = 1
    x[1,0] = 2
    x[2,0] = 3
    x[3,0] = 4
    x[4,0] = 5
    x[5,0] = 6
    y[0,0] = 3.2
    y[1,0] = 6.4
    y[2,0] = 10.5
    y[3,0] = 17.7
    y[4,0] = 28.1
    y[5,0] = 38.5
    A = np.ndarray((3,3))

    a1=a2=a3=b1=b2=b3=c1=c2=c3=d1=d2=d3=0
    i=j=0

    for i in range(len(x)):
            
            a1+=x[i,0]*x[i,0]*x[i,0]*x[i,0]
            a2+=x[i,0]*x[i,0]*x[i,0]
            a3+=x[i,0]*x[i,0]

            b1+=x[i,0]*x[i,0]*x[i,0]
            b2+=x[i,0]*x[i,0]
            b3+=x[i,0]

            c1+= x[i,0]*x[i,0]
            c2+= x[i,0]
            c3+= 1

            d1+=y[i,0]*x[i,0]*x[i,0]
            d2+=y[i,0]*x[i,0]
            d3+=y[i,0]


    A[0,0] = a1
    A[0,1] = b1
    A[0,2] = c1
    A[1,0] = a2
    A[1,1] = b2
    A[1,2] = c2
    A[2,0] = a3
    A[2,1] = b3
    A[2,2] = c3

    z = np.ndarray((3,1))

    z[0,0] = d1
    z[1,0] = d2
    z[2,0] = d3

    
    #A[0,0] = 60
    #A[0,1] = 20
    #A[1,0] = 20
    #A[1,1] = 8

    ainv = np.linalg.inv(A)
    #z[0,0] = 179
    #z[1,0] = 63.8
    res = np.dot(ainv,z) # a = res[0,0] and b=[1,0]
    print(res)
    # do a scatter plot of the data
    area = 10
    colors =['black']
    plt.scatter(x, y, s=area, c=colors, alpha=0.5, linewidths=8)
    plt.title('Linear Least Squares Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    #plot the fitted line
    yfitted = x*x*res[0,0] + res[1,0]*x + res[2,0]
    line,=plt.plot(x, yfitted, '--', linewidth=2) #line plot
    line.set_color('red')
    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
