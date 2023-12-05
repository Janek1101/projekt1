import numpy as np

class Interplacja_sklejana_3st:
    def __init__(self, dane, start, koniec, krok):
        self.dane = dane
        self.n = dane.shape[0] - 1
        self.hi = self.liczenie_h()
        self.bi = self.liczenie_b()
        self.ui = self.liczenie_u()
        self.vi = self.liczenie_v()
        self.zi = self.liczenie_z()
        self.Ai = self.liczenie_A()
        self.Bi = self.liczenie_B()
        self.Ci = self.liczenie_C()
        self.poczatek = start
        self.koniec = koniec
        self.krok = krok
        self.x = np.linspace(start, koniec, krok)
        self.Si = self.S()




    def liczenie_h(self):
        h = np.zeros(self.n)
        for i in range(self.n):
            h[i] = self.dane[i+1, 0] - self.dane[i, 0]
        return h

    def liczenie_b(self):
        b = np.zeros(self.n)
        for i in range(self.n):
            b[i] = (6 / self.hi[i]) * (self.dane[i+1, 1] - self.dane[i, 1])
        return b

    def liczenie_u(self):
        u = np.zeros(self.n - 1)
        u[0] = 2 * (self.hi[0] + self.hi[1])
        for i in range(1, u.shape[0]):
            u[i] = 2 * (self.hi[i] + self.hi[i + 1]) - ((self.hi[i] ** 2) / (u[i - 1]))
        return u

    def liczenie_v(self):
        v = np.zeros(self.n - 1)
        v[0] = self.bi[1] - self.bi[0]
        for i in range(1, v.shape[0]):
            v[i] = (self.bi[i + 1] - self.bi[i] - (self.hi[i] * v[i - 1]) / self.ui[i - 1])
        return v

    def liczenie_z(self):
        z = np.zeros(self.n + 1)
        for i in range(self.n - 1, 0, -1):
            z[i] = (1/self.ui[i - 1]) * (self.vi[i - 1] - (self.hi[i] * z[i + 1]))
        return z

    def liczenie_A(self):
        A = np.zeros(self.n)
        for i in range(self.n):
            A[i] = ((1 / (6 * self.hi[i])) * (self.zi[i + 1] - self.zi[i]))
        return A

    def liczenie_B(self):
        B = np.zeros(self.n)
        for i in range(self.n):
            B[i] = self.zi[i] / 2
        return B

    def liczenie_C(self):
        C = np.zeros(self.n)
        for i in range(self.n):
            C[i] = -1 * (self.hi[i] / 6) * ( self.zi[i + 1] + 2 * self.zi[i]) + (1 / self.hi[i]) * (self.dane[i+1, 1] - self.dane[i, 1])
        return C

    def S(self):
        Si = np.zeros(self.krok)
        for i in range(self.n):
            Si += (self.dane[i, 1] + (self.x - self.dane[i, 0]) * (self.Ci[i] + (self.x - self.dane[i, 0]) * (self.Bi[i] + (self.x - self.dane[i, 0]) * self.Ai[i]))) * (self.x >= self.dane[i, 0]) * (self.x < self.dane[i + 1, 0])
        Si += (self.dane[0, 1] + (self.x - self.dane[0, 0]) * (self.Ci[0] + (self.x - self.dane[0, 0]) * (self.Bi[0] + (self.x - self.dane[0, 0]) * self.Ai[0]))) * (self.x < self.dane[0, 0]) * (self.x >= self.poczatek)
        Si += (self.dane[-1, 1] + (self.x - self.dane[-1, 0]) * (self.Ci[-1] + (self.x - self.dane[-1, 0]) * (self.Bi[-1] + (self.x - self.dane[-1, 0]) * self.Ai[-1]))) * (self.x >= self.dane[-1, 0]) * (self.x <= self.koniec)
        return Si

class Lagrange:
    def __init__(self, data, poczatek, koniec, krok):
        self.data = data
        self.x = np.linspace(poczatek, koniec, krok)
        self.n = data.shape[0]
        self.poly = self.lagrange_poly()
        self.dzialaXDDDD = self.lgr()

    def lagrange_poly(self):
        lagrangepoly = np.ones((self.n, self.x.shape[0]))
        for i in range(self.n):
            for j in range(self.n):
                if j != i:
                    lagrangepoly[i, :] *= (self.x - self.data[j, 0]) / (self.data[i, 0] - self.data[j, 0])
        return lagrangepoly

    def lgr(self):
        p = np.zeros(self.x.shape[0])
        for i in range(self.n):
            p += self.poly[i,:] * self.data[i, 1]
        return p

class Sklejana_stopnia_pierwszego:
    def __init__(self, data, poczatek, koniec, krok):
        self.data = data
        self.x = np.linspace(poczatek, koniec, krok)
        self.poczatek = poczatek
        self.koniec = koniec
        self.sklej = self.sklejana()

    def sklejana(self):
        P_linear = np.zeros(self.x.shape)
        for n in range(self.data.shape[0] - 1):
            if n == 0:
                P_linear += ((self.data[n + 1, 1] - self.data[n, 1]) / (self.data[n + 1, 0] - self.data[n, 0]) * (self.x - self.data[n, 0])
                             + self.data[n, 1]) * (self.x <= self.data[n + 1, 0])
            elif n == self.data.shape[0] - 2:
                P_linear += ((self.data[n + 1, 1] - self.data[n, 1]) / (self.data[n + 1, 0] - self.data[n, 0]) * (self.x - self.data[n, 0])
                             + self.data[n, 1]) * (self.x > self.data[n, 0])
            else:
                P_linear += ((self.data[n + 1, 1] - self.data[n, 1]) / (self.data[n + 1, 0] - self.data[n, 0]) * (self.x - self.data[n, 0])
                             + self.data[n, 1]) * (self.x > self.data[n, 0]) * (self.x <= self.data[n + 1, 0])
        return P_linear













