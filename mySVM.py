import numpy as np
import math
import random

__author__ = 'banban'
#import sklearn.svm
#a=sklearn.svm.SVC()
#a.fit
#a.predict()

class SVMError(Exception):
    pass


class MySVM(object):
    def __init__(self, x, y, kf_type='none', c=1.0, eps=0.001, tol=0.001):
        self.x = np.array(x)  # x is a point in R2 in the format of tuple.
        self.y = np.array(y)  # y is the category of x.
        self.sample_size = self.x.shape[0]
        self.feature_size = self.x.shape[1]
        self.c = c
        self.eps = eps
        self.tol = tol
        self.alpha = np.zeros(self.sample_size)
        self.b = 0.0
        self.kf_type = kf_type

    def train(self):
        #while not (self._kkt() and self._update_param(i1,i2):
        while not self._kkt():
            i1, i2 = self._working_set_selection()
            self._update_param(i1,i2)

    def predict(self, new_x):
        new_x = np.array(new_x)
        if self._u(new_x) > 0:
            label = 1
        else:
            label = -1
        return label

    def _kernel(self, x, z): # kernel function, assign the specific kernel, varible x and z, return the K(x,z).
        kf = {}
        p = 2
        sigma = 1.0
        kf['none'] = np.dot(x, z)
        kf['polynomial'] = (np.dot(x, z) + 1) ** p
        kf['gaussian'] = math.exp(-1.0 * np.dot((x - z), (x - z)) / 2 * (sigma ** 2))
        # kf['gaussian'], kf_p = 1.0, 0.0
        # for i in xrange(self.feature_size):
        #    kf_p += x[i] * z[i]
        #    kf['gaussian'] *= math.exp(-1.0 * ((x[i] - z[i]) ** 2) / 2 * (sigma ** 2))
        # kf['polynomial'] = (kf_p + 1)**p
        if self.kf_type not in kf.keys():
            print 'unsupported kernel!\n ' \
                  'supported kernel function are: %s \n' % kf.keys()
            raise SVMError("unsupported kernel")
        return kf[self.kf_type]

    def _u(self, x):
        u = 0.0
        for j in xrange(self.sample_size):
            u += self.y[j] * self.alpha[j] * self._kernel(self.x[j], x)
        return u + self.b

    def _update_param(self, i1, i2):
        # calculating L and H, according to whether y1 equals y2.
        x1, x2 = self.x[i1], self.x[i2]
        y1, y2 = self.y[i1], self.y[i2]
        alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
        k = self._kernel
        e1 = self._u(x1) - y1
        e2 = self._u(x2) - y2
        eta = k(x1, x1) + k(x2, x2) - 2 * k(x1, x2)
        if y1 != y2:
            l = max(0.0, alpha2 - alpha1)
            h = min(self.c, self.c - alpha1 + alpha2)
        else:
            l = max(0.0, alpha1 + alpha2 - self.c)
            h = min(self.c, alpha1 + alpha2)

        # alpha2 new unclipped
        if eta > 0:
            alpha2_new_unc = alpha2 + y2 * (e1 - e2) / eta
        else:
            w_alpha = lambda t: y2*(e2-e1)*t
            if w_alpha(l) < w_alpha(h) - self.eps:
                alpha2_new_unc = l
            elif w_alpha(l) > w_alpha(h) + self.eps:
                alpha2_new_unc = h
            else:
                alpha2_new_unc = alpha2

        # clipping alpha2
        if alpha2_new_unc > h:
            alpha2_new = h
        elif alpha2_new_unc < l:
            alpha2_new = l
        else:
            alpha2_new = alpha2_new_unc
        alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

        # calculating b
        b1_new = -1 * e1 - y1 * k(x1, x1) * (alpha1_new - alpha1) \
                 - y2 * k(x2, x1) * (alpha2_new - alpha2) + self.b
        b2_new = -1 * e2 - y1 * k(x1, x2) * (alpha1_new - alpha1) \
                 - y2 * k(x2, x2) * (alpha2_new - alpha2) + self.b
        if (0.0 < alpha1_new < self.c) and (0.0 < alpha2_new < self.c):
            b_new = b1_new
        elif 0.0 < alpha1_new < self.c:
            b_new = b1_new
        elif 0.0 < alpha2_new < self.c:
            b_new = b2_new
        else:  # alpha1_new and alpha2_new are 0 or C.
            b_new = (b1_new + b2_new) / 2.0

        #calculating the fluctuation of alpha2
        #if abs(alpha2_new - alpha2) < self.eps*(alpha2_new+alpha2+self.eps):
            #delta = False
        #else:
            #delta = True
        # updating alpha1.
        self.alpha[i1] = alpha1_new
        # updating alpha2.
        self.alpha[i2] = alpha2_new
        # updating b.
        self.b = b_new
        #return delta

    def _working_set_selection(self, selection_method='heuristic'):
        if selection_method == 'heuristic':
            # select the i1 th sample which violate kkt conditions.
            for i in xrange(self.sample_size):
                if (0 < self.alpha[i] < self.c) and (self.y[i] * self._u(self.x[i]) - 1 > self.tol):   # is this conditon right?
                    i1 = i
                    break
                elif self.alpha[i] == 0 and (self.y[i] * self._u(self.x[i]) - 1 < -1*self.tol):
                    i1 = i
                    break
                elif self.alpha[i] == self.c and (self.y[i] * self._u(self.x[i]) - 1 > self.tol):
                    i1 = i
                    break
            # select i2 to maximize |E1 - E2| and then to maximize the change of alpha2.
            x1 = self.x[i1]
            y1 = self.y[i1]
            e1 = self._u(x1) - y1
            i2 = 0
            tmax = 0
            for j in xrange(self.sample_size):
                #if 0 < self.alpha[j] < self.c:    # why only non-bound samples instead of all traing set?
                e = self._u(self.x[j]) - self.y[j]
                tmp = abs(e1 - e)
                if tmp > tmax:
                    tmax = tmp
                    i2 = j
        elif selection_method == 'random':
            i1 = random.randint(0, self.sample_size - 1)
            while True:
                i2 = random.randint(0, self.sample_size - 1)
                if i1 != i2:
                    break
        else:
            print 'unsupported selection method!\n supported selection methods are : "random" \n'
            raise SVMError("Unsupported method")
        return i1, i2

    def _kkt(self):
        for i in xrange(self.sample_size):
            violated_1 = (self.alpha[i] < self.c) and (self.y[i] * self._u(self.x[i]) - 1 < -1*self.tol)
            violated_2 = (self.alpha[i] > 0.0) and (self.y[i] * self._u(self.x[i]) - 1 > self.tol)
            if violated_1 or violated_2:
                return False
            else:
                continue
        return True

    #def _kkt(self):
        #for i in xrange(self.sample_size):
            #kkt1 = self.alpha[i] == 0 and self.y[i]*self._u(self.x[i]) -1 >=0
            #kkt2 = 0<self.alpha[i]<self.c and self.y[i]*self._u(self.x[i]) -1 ==0
            #kkt3 = self.alpha[i] == self.c and self.y[i]*self._u(self.x[i]) -1 <= 0
            #if not (kkt1 or kkt2 or kkt3):
                #return False
            #else:
                #continue
        #return True


# test code
if __name__ == '__main__':
    #o = MySVM([[3,3], [4,3], [1,1]], [1,1,-1])
    o = MySVM([[0.3858,0.4687], [0.4871,0.611], [0.9218,0.4103], [0.7382, 0.8936],[0.1763,0.0579],
               [0.4057,0.3529], [0.9355, 0.8132], [0.2146, 0.0099]], [1,-1,-1,-1,1,1,-1,1])
    o.train()
    #print o.predict([6,6])
    print o.predict([0.3,0.5])
