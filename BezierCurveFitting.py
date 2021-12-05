import numpy as np

class BezierCurveFittingModel:
    def __init__(self, func, numCurves, x0, x1):
        # Store the parameters:
        self.n = numCurves
        self.f = func
        self.x0 = x0
        self.x1 = x1
        # Data preparation:
        x = np.linspace(x0, x1, 4*numCurves+1)
        # Evaluate the function:
        fArr = np.zeros(len(x))
        for i in range(4*numCurves+1):
            fArr[i] = func(x[i])
        # Define subproblems to be solved
        self.subproblems = []
        for i in range(numCurves):
            P0 = np.array([x[4*i], fArr[4*i]], dtype=np.float64)
            P1_ = np.array([x[4*i+1], fArr[4*i+1]], dtype=np.float64)
            Pmid = np.array([x[4*i+2], fArr[4*i+2]], dtype=np.float64)
            P2_ = np.array([x[4*i+3], fArr[4*i+3]], dtype=np.float64)
            P3 = np.array([x[4*(i+1)], fArr[4*(i+1)]], dtype=np.float64)
            self.subproblems.append((P0, P1_, Pmid, P2_, P3))
    
    # Functions needed for optimization:
    def B(self, P0, P1, P2, P3, t):
        return P0*(1-t)**3 + 3*P1*t*(1-t)**2 + 3*P2*(1-t)*t**2 + P3*t**3
    
    def dB_t(self, P0, P1, P2, P3, t):
        return 3*(P1-P0)*(1-t)**2 + 6*(P2-P1)*t*(1-t) + 3*(P3-P2)*t**2
    
    def dB_P1(self, t):
        return 3*t*(1-t)**2
    
    def dB_P2(self, t):
        return 3*(1-t)*t**2
    
    # Solve function:
    def fit(self, P0, P1_, Pmid, P2_, P3, N=1000, tol=1e-9):
        t1 = 1/4
        tmid = 1/2
        t2 = 3/4
        P1 = np.copy(P1_)
        P2 = np.copy(P2_)
        P1_prev = np.copy(P1_)
        P2_prev = np.copy(P2_)
        t1_prev = t1
        t2_prev = t2
        tmid_prev = tmid
        alpha = 0.01
        L = np.inf
        L_prev = L
        tol1 = 1e-5
        for i in range(N):
            # Optimize P1, P2 for fixed t1, tmid and t2:
            for j in range(N):
                D_P1 = 2*(self.B(P0,P1,P2,P3,t1) - P1_)*self.dB_P1(t1) + 2*(self.B(P0,P1,P2,P3,t2) - P2_)*self.dB_P1(t2) + 2*(self.B(P0,P1,P2,P3,tmid) - Pmid)*self.dB_P1(tmid)
                D_P2 = 2*(self.B(P0,P1,P2,P3,t1) - P1_)*self.dB_P2(t1) + 2*(self.B(P0,P1,P2,P3,t2) - P2_)*self.dB_P2(t2) + 2*(self.B(P0,P1,P2,P3,tmid) - Pmid)*self.dB_P2(tmid)

                P1 -= alpha*D_P1
                P2 -= alpha*D_P2

                if np.all(np.abs(P1 - P1_prev) < tol1) and np.all(np.abs(P2 - P2_prev) < tol1):
                    break
                else:
                    P1_prev = np.copy(P1)
                    P2_prev = np.copy(P2)
            # Optimize t1, t2, tmid for fixed P1, P2:
            for j in range(N):
                D_t1 = np.sum(2*(self.B(P0,P1,P2,P3,t1) - P1_)*self.dB_t(P0,P1,P2,P3,t1))
                D_t2 = np.sum(2*(self.B(P0,P1,P2,P3,t2) - P2_)*self.dB_t(P0,P1,P2,P3,t2))
                D_tmid = np.sum(2*(self.B(P0,P1,P2,P3,tmid) - Pmid)*self.dB_t(P0,P1,P2,P3,tmid))
                
                t1 -= alpha*D_t1
                t2 -= alpha*D_t2
                tmid -= alpha*D_tmid

                if np.abs(t1-t1_prev) < tol1 and np.abs(t2-t2_prev) < tol1 and np.abs(tmid - tmid_prev) < tol1:
                    break
                else:
                    t1_prev = t1
                    t2_prev = t2
                    tmid_prev = tmid
            # Evaluate loss function and check for termination:
            L = np.sum((self.B(P0,P1,P2,P3,t1) - P1_)**2) + np.sum((self.B(P0,P1,P2,P3,t2) - P2_)**2) +  np.sum((self.B(P0,P1,P2,P3,tmid) - Pmid)**2)
            #print(f'Iteration {i+1} Loss: {L}')
            if np.abs(L-L_prev) < tol:
                break
            else:
                L_prev = L
        return (P1,P2)
    
    def solve(self):
        points = []
        for (P0, P1_, Pmid, P2_, P3) in self.subproblems:
            (P1,P2) = self.fit(P0, P1_, Pmid, P2_, P3)
            points.append((P0,P1,P2,P3))
        return points


    
