import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import dolfin as df
from scipy.special import erfcinv, erfinv, erf, erfc
import random
import bisect

df.set_log_active(False)

class ARS(object):
    def __init__(self, pdf, bounds, nsp=50):
        self.pdf = pdf

        if isinstance(bounds[0], (np.int,np.float)):
            bounds = [bounds]

        self.dim = len(bounds)
        nsp = [nsp]*self.dim
        df = np.diff(bounds)
        for i in range(1, self.dim):
            nsp[i] = nsp[i-1]*df[i][0]/df[i-1][0]

        points = np.array([np.linspace(*bounds[i], nsp[i], retstep=True)
                                                      for i in range(self.dim)])
        self.dv = [points[i][1] for i in range(self.dim)]
        self.volume = np.prod(self.dv)

        sp = [points[i][0] for i in range(self.dim)]
        self.sp = np.meshgrid(*sp, indexing='ij')

        self.xs = [self.sp[i].flatten() for i in range(self.dim)]

        self.build_pdf()

    def build_pdf(self):
        f_sp = self.pdf(self.sp)
        f_sp[np.where(f_sp<0)] = 0

        u_slice = [slice(0,None), slice(0,None), slice(1,None)]
        l_slice = [slice(0,None), slice(0,None), slice(0,-1)]
        u_sl = [None]*self.dim
        l_sl = [None]*self.dim
        for i in range(self.dim):
            u_sl[i] = u_slice[-(i+1):]
            l_sl[i] = l_slice[-(i+1):]

        pdf_max = np.maximum(f_sp[u_sl[0]], f_sp[l_sl[0]])
        for i in range(self.dim-1):
            pdf_max = np.maximum(pdf_max[u_sl[i]], pdf_max[l_sl[i]])

        integral = self.volume*pdf_max
        w = integral/np.sum(integral)

        self.pdf_max = pdf_max.flatten()
        self.w_s = w.shape
        w = w.flatten()
        self.weights = np.cumsum(w)

    def sample_pdf(self, n):
        r = np.random.rand(n)
        inds = np.searchsorted(self.weights, r, side='right')
        index = np.unravel_index(inds, self.w_s)
        vs = np.array([self.sp[i][index]+self.dv[i]*np.random.rand(n) for i in range(self.dim)]).T
        pdf_vs = self.pdf_max[inds]*np.random.rand(len(inds))
        return vs, pdf_vs

    def sample(self, N):
        vs = np.array([]).reshape(-1,self.dim)
        count = 0
        while len(vs) < N:
            n = N - len(vs)
            vs_new, p_vs_new = self.sample_pdf(n)
            pdf_vs_new = self.pdf(vs_new.T)
            vs_new = vs_new[np.where(p_vs_new<pdf_vs_new)]
            vs = np.concatenate([vs, vs_new])
            count +=1
        print("ARS iterations: ", count)
        return vs

class ORS(object):
    def __init__(self, pdf, bounds, nsp=50):
        self.pdf = pdf
        if isinstance(bounds[0], (np.int,np.float)):
            bounds = [bounds]
        self.dim = len(bounds)
        nsp = [nsp]*self.dim
        df = np.diff(bounds)
        for i in range(1, self.dim):
            nsp[i] = nsp[i-1]*df[i][0]/df[i-1][0]

        points = np.array([np.linspace(*bounds[i], nsp[i], retstep=True)
                                                      for i in range(self.dim)])
        self.dv = [points[i][1] for i in range(self.dim)]
        self.volume = np.prod(self.dv)

        sp = [points[i][0] for i in range(self.dim)]
        self.sp = np.meshgrid(*sp, indexing='ij')

        self.xs = [self.sp[i].flatten() for i in range(self.dim)]
        self.build_pdf()

    def build_pdf(self):
        f_sp = self.pdf(self.sp)
        f_sp[np.where(f_sp<0)] = 0

        u_slice = [slice(0,None), slice(0,None), slice(1,None)]
        l_slice = [slice(0,None), slice(0,None), slice(0,-1)]
        u_sl = [None]*self.dim
        l_sl = [None]*self.dim
        for i in range(self.dim):
            u_sl[i] = u_slice[-(i+1):]
            l_sl[i] = l_slice[-(i+1):]

        pdf_max = np.maximum(f_sp[u_sl[0]], f_sp[l_sl[0]])
        for i in range(self.dim-1):
            pdf_max = np.maximum(pdf_max[u_sl[i]], pdf_max[l_sl[i]])

        integral = self.volume*pdf_max
        w = integral/np.sum(integral)

        self.pdf_max = pdf_max.flatten()
        self.w_s = w.shape
        w = w.flatten()
        self.weights = np.cumsum(w)

    def sample_pdf(self, n):
        inds = [bisect.bisect_right(self.weights,
                                    np.random.random()*self.weights[-1])
                                                              for i in range(n)]
        index = np.array(np.unravel_index(inds, self.w_s))

        vs = np.array([[self.sp[i][tuple(index[:,j])]+self.dv[i]*np.random.rand()
                        for i in range(self.dim)]
                            for j in range(n)])

        pdf_vs = self.pdf_max[inds]
        return vs, pdf_vs

    def sample(self, N):
        vs = np.array([]).reshape(-1, self.dim)
        count = 0
        while len(vs) < N:
            n = N - len(vs)
            vs_new, p_vs_new = self.sample_pdf(n)
            pdf_vs_new = self.pdf(vs_new.T)
            alpha = np.divide(pdf_vs_new,p_vs_new)
            r = np.random.rand(n)
            vs_new = np.array([vs_new[i,:] for i in range(n) if r[i] <= alpha[i]]).reshape((-1,self.dim))
            vs = np.concatenate([vs, vs_new])
            count +=1
        print("ORS iterations: ", count)
        return vs

def rejection_sampling(N, dim, pdf, pdf_max=1.0, lb=0.0, ub=1.0, transform=lambda t: t):
    vs = np.array([]).reshape((-1,dim))
    count = 0
    while len(vs) < N:
        n = N - len(vs)
        u = np.random.uniform(lb, ub, (n,dim))
        r = pdf_max * np.random.random(n)
        new_vs = [transform(u[i]) for i in range(n) if r[i] < pdf(u[i])]
        new_vs = np.array(new_vs).reshape((-1,dim))
        vs = np.concatenate([vs, new_vs])
        count +=1
    print("RS iterations: ", count)
    return vs

def maxwellian(v_thermal, v_drift, N):
    dim = N[1]
    if v_thermal == 0.0:
        v_thermal = np.finfo(float).eps

    if isinstance(v_drift, (float, int)):
        v_drift = np.array([v_drift] * dim)

    cdf_inv = lambda x, vd=v_drift, vth=v_thermal: vd - \
        np.sqrt(2.) * vth * erfcinv(2 * x)
    w_r = np.random.random((N[0], dim))
    return cdf_inv(w_r)

if __name__ == "__main__":

    def test_flux():
        bounds = [[-7,7],[-7,7]]
        nsp = 50
        N = 10000000
        dim = 2#len(bounds)
        vd = np.array([0.0,0.0])
        vth = 1.0
        # pdf = lambda x: x[0]*np.exp(-0.5*((x[0]-3.)/2.)**2)/(np.sqrt(2*np.pi)*2)
        # pdf = lambda x: np.exp(-0.5*x[0]**2)*(1+(np.sin(3*x[0]))**2)*(1+(np.cos(5*x[0]))**2)
        # g = lambda x: np.exp(-0.5*x**2)*(1+(np.sin(3*x))**2)*(1+(np.cos(5*x))**2)
        # g = lambda x: x*np.exp(-0.5*((x-3.)/2.)**2)/(np.sqrt(2*np.pi)*2)
        pdf = lambda x, vth=vth: np.exp(-0.5*((x[0])**2 + (x[1])**2)/vth**2)/((np.sqrt(2*np.pi)*vth)**2)
        # pdf_flux = lambda x, vn=vn, pdf=pdf: np.add(*[x[i]*vn[i] for i in range(dim)])*pdf(x)
        # bounds = [[0,15],[-8,8]]

        # Dependendent pdf:
        # rho = 0.9
        # sx, sy = 1, 1
        # mx, my = 0, 0
        # def pdf(x):
        #     return (1./(2*np.pi*np.sqrt((1-rho**2))))*np.exp(-0.5*(x[0]**2 + x[1]**2-2*rho*x[0]*x[1]) / (1-rho**2))

        # gen = ORS(pdf, bounds, nsp)
        # t0 = time.time()
        # vs1 = gen.sample(N)
        # print("Time ORS: ", time.time()-t0)

        gen = ARS(pdf, bounds, nsp)
        t0 = time.time()
        vs2 = gen.sample(N)
        print("Time ARS: ", time.time()-t0)

        t0 = time.time()
        vs3 = maxwellian(vth, vd, [N,2])
        print("Time maxwell: ", time.time()-t0)

        plt.hist2d(vs2[:,0],vs2[:,1],bins=100,norm=LogNorm())
        plt.axis('equal')
        plt.figure()
        plt.hist2d(vs3[:,0],vs3[:,1],bins=100,norm=LogNorm())
        plt.axis('equal')
        plt.show()
        exit()
        t0 = time.time()
        vs3 = rejection_sampling(N, dim, pdf, pdf_max=1.0, lb=bounds[0][0], ub=bounds[0][1])
        print("Time RS: ", time.time()-t0)
        exit()
        def pdf_maxwellian(i, t):
            return 1.0 / (np.sqrt(2 * np.pi) * vth) *\
                np.exp(-0.5 * ((t - vd[i])**2) / (vth**2))

        def pdf_flux_nondrifting(s, t):

            return s * (t / vth**2) * np.exp(-0.5 * (t / vth)**2)

        def pdf_flux_drifting(i, s, t):
            return (s * t * np.exp(-0.5 * ((t - vd[i])**2) / (vth**2))) /\
                (vth**2 * np.exp(-0.5 * (vd[i] / vth)**2) +
                 s * np.sqrt(0.5 * np.pi) * vd[i] * vth *
                 (1. + s * erf(vd[i] / (np.sqrt(2) * vth))))

        xs = np.linspace(0,12,1000)#vd[0] - 5 * vth, vd[0] + 5 * vth, 1000)
        dx = sum((xs[1]-xs[0])*g(xs))
        for vs in [vs1, vs2,vs3]:
            plt.figure(figsize=(6, 5))
            plt.hist(vs[:], bins=300, color = 'blue', normed=1)
            plt.plot(xs, (1/dx)*g(xs), color='red')
            # # plt.hist2d(vs[:,0],vs[:,1],bins=100,norm=LogNorm())
            # plt.figure(figsize=(6, 5))
            # plt.hist(vs[:,0], bins=300, color = 'blue', normed=1)
            # plt.plot(xs, pdf_maxwellian(0,xs), color='red')
            # # plt.plot(xs, pdf_flux_nondrifting(1,xs), color='red')
            # plt.figure(figsize=(6, 5))
            # plt.hist(vs[:,1], bins=300, color = 'blue', normed=1)
            # plt.plot(xs, pdf_maxwellian(1,xs), color='red')
        plt.show()

    test_flux()