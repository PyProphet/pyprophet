import numpy
import sklearn.lda

cov = numpy.array(((1,0,-1), (0,1,1), (1, 0, 1.0)))

N=100

m1 = numpy.random.multivariate_normal(mean=(0,0,0), cov=cov, size=(N,))
m2 = numpy.random.multivariate_normal(mean=(2,2,0), cov=0.2*cov, size=(N,))

cov2 = numpy.array(((1,0,-1.2), (0,1,1.2), (1, 0, 1.2)))
m3 = numpy.random.multivariate_normal(mean=(0,0,0), cov=cov2, size=(5*N,))
m4 = numpy.random.multivariate_normal(mean=(2,2,2), cov=0.2*cov2, size=(5*N,))

Xfull = numpy.vstack((m1, m2, m3, m4))
yfull = numpy.zeros((2*N+10*N,))
yfull[N:2*N] = 1.0
yfull[-5*N:] = 1.0
lda = sklearn.lda.LDA()
lda.fit(Xfull, yfull)
print "optimal scaling"
print lda.scalings_.T
print

X0 = numpy.vstack((m1, m2))
y0 = numpy.zeros((2*N,))
y0[N:] = 1.0


lda = sklearn.lda.LDA()
lda.fit(X0, y0)
print lda.scalings_.T

import pylab

def hist(m1, m2, m3, m4, w):
    v1 = [ numpy.dot(row, w) for row in m1]
    v2 = [ numpy.dot(row, w) for row in m2]
    v3 = [ numpy.dot(row, w) for row in m3]
    v4 = [ numpy.dot(row, w) for row in m4]
    pylab.figure()
    pylab.hist(v3, label="m3", bins=20)
    pylab.hist(v4, label="m4", bins=20)
    pylab.hist(v1, label="m1", bins=20)
    pylab.hist(v2, label="m2", bins=20)
    pylab.legend()

hist(m1, m2, m3, m4, lda.scalings_.flatten())

X1 = numpy.vstack((m1, m2, m3, m4))

for _ in range(2):
    
    hist(m1, m2, m3, m4, lda.scalings_.flatten())

    print "%2d " % lda.predict(m1).sum(),0
    print "%2d " % lda.predict(m2).sum(),N
    print "%2d " % lda.predict(m3).sum(),0
    print "%2d " % lda.predict(m4).sum(),5*N

    y_pred = lda.predict(X1)
    y_pred[:N] = 0.0
    y_pred[N:2*N] = 1.0
    lda = sklearn.lda.LDA()
    lda.fit(X1, y_pred)
    print lda.scalings_.T

hist(m1, m2, m3, m4, lda.scalings_.flatten())


print "%2d " % lda.predict(m1).sum(),0
print "%2d " % lda.predict(m2).sum(),N
print "%2d " % lda.predict(m3).sum(),0
print "%2d " % lda.predict(m4).sum(),5*N
print

pylab.show()
