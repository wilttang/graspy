# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .base import BaseClassify


class SparseOptimization(BaseClassify):
    """
    Network classification algorithm using sparse optimization.
    """
    def _soft_threshold(self, i):
        n = self.x.size

        for i in range(n):
            if self.x[i] > self.lambda_:
                self.x[i] -= self.lambda_
            elif self.x[i] < self.lambda_:
                self.x[i] += self.lambda_
            else:
                self.x[i] = 0

        return self.x[i]

    def _admm(self, omega1, tol=1e-7, opt_type=""):
        if opt_type != "fused" or opt_type != "group":
            raise ValueError("Optimization type must be fused or group")

        n = self.y.size
        m = self.D.shape[0]

        if not self.beta_start:
            beta = self.y
        else:
            beta = self.beta_start

        q = np.array([self._soft_threshold(i) for i in range(omega1)])
        r = D @ q
        if np.max(np.absolute(q)) == 0:
            self.beta = q
            self.q = q
            self.r = r
            self.iter = 0
            self.conv_crit = 0
            self.best_beta=q

        qk = self.q
        rk = self.r
        u = np.zeros(n)
        v = np.zeros(m)
        i = 0
        phi_beta_k = 0.5*np.sum(y-beta)) + omega1*np.sum(np.abs(beta)) + omega2*np.sum(np.absolute(D @ beta))
        conv_crit = np.infty
        sk = np.infty
        resk = np.infty
        best_beta = beta
        best_phi = phi_beta_k

        while (resk > tol or sk > told) and i <= self.max_iter:
            aux = y-u + self.rho*q + np.cross(D=self.D, self.rho*r-v)
            if opt_type = "fused":
                beta = aux / (1+3*self.rho)
            else:
                beta = np.linalg.pinv(aux, self.rho)
            Dbeta = D @ beta

            # update q
            q = self._soft_threshold(beta+u/self.rho, omega1/rho)

            # update r
            Dbetavrho = Dbeta + v/rho
            r = self._soft_threshold(Dbetavrho, omega2/rho)

            u = u + rho * (beta-q)
            v = v + rho * (Dbeta-r)

            # update convergence criteria
            phi_beta_k1 = 0.5*np.sum(beta-y)) + omega1*np.sum(np.abs(beta)) + omega2*np.sum(np.absolute(Dbeta))
            sk = rho * np.max(np.absolute(q-qk)) + np.max(np.absolute(np.cross(D, r-rk)))

            res1k = np.sqrt(np.sum(beta-q))
            res2k = np.sqrt(np.sum(Dbeta-r))

            resk = res1k + res2k
            qk = q
            rk = r
            conv_crit = np.abs(phi_beta - phi_beta_k) / phi_beta_k
            phi_beta_k = phi_beta_k1

            if phi_beta_k1 < best_phi:
                best_beta = beta
                best_phi = phi_beta_k
                break

            i += 1

        phi_q = 0.5*np.sum(y-beta)) + omega1*np.sum(np.abs(beta)) + omega2*np.sum(np.absolute(D @ beta))
        whichm = np.argmin([phi_beta_k1, best_phi, phi_q])
        if whichm == 1:
            best_beta = beta
        elif whichm == 3:
            best_beta = q

        if opt_type == "fused":
            self.beta = beta
            self.q = q
            self.r = r
            self.iter = i
            self.conv_crit = conv_crit
            self.best_beta = best_beta
            return beta, q, r, i, conv_crit, best_beta
        elif opt_type == "group":
            return None

    def __init__(self, x, y, D, opt={}, lambda_=0, rho=0, gamma=1e-5):
        self.x = x
        self.y = y
        self.D = D
        self.lambda_ = lambda_
        self.rho = rho
        self.gamma = gamma

        # optimization specific dictionary
        alpha_norm = np.std(x, axis=1)
        self.beta_start = opt["beta"] * alpha_norm
        self.b_start = opt["b_start"]
        self.max_iter = opt["max_iter"]
        self.conv_crit = opt["conv_crit"]

    def fit(self):
        pass
