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
from collections import Counter

from .base import BaseClassify


def _soft_threshold(x, lambda_, opt_type, nodes=None):
    n = x.shape[0]

    for i in range(n):
        if opt_type == "fused":
            if x[i] > lambda_:
                x[i] -= lambda_
            elif x[i] < lambda_:
                x[i] += lambda_
            else:
                x[i] = 0
        else:
            norm_node = _gl_penalty(x, nodes)
            t = 1 - lambda_/norm_node
            for j in range((nodes-1)*i, (nodes-1)*(i+1)):
                if norm_node <= lambda_:
                    x[j] = 0
                else:
                    x[j] *= t

    return x

def _gl_penalty(b, nodes=None):
    for i in range(nodes):
        norm_node = np.sum([b[j] ** 2 for j in range((nodes-1)*i, (nodes-1)*(i+1))])
        gl = np.sqrt(norm_node)

    return gl


class SparseOpt(BaseClassify):
    """
    Network classification algorithm using sparse optimization.
    """
    def __init__(self, x, y, D, opt={}, lambda_=0, rho=0, gamma=1e-5, nodes=None):
        self.x = x
        self.y = y
        self.D = D
        self.lambda_ = lambda_
        self.rho = rho
        self.gamma = gamma
        self.nodes = nodes

        # optimization specific dictionary
        alpha_norm = np.std(x, axis=1)
        self.beta_start = opt["beta"] * alpha_norm
        self.b_start = opt["b_start"]
        self.max_iter = opt["max_iter"]
        self.conv_crit = opt["conv_crit"]

    def _admm(self, omega1, omega2, G_penalty_factors=None, tol=1e-7, opt_type="", nodes=None):
        if opt_type != "fused" or opt_type != "weights" or opt_type != "group":
            raise ValueError("Optimization type must be fused, group, or weights")

        n = self.y.size
        if opt_type == "group":
            D_list = self.D
            G = len(self.D)
        else:
            m = self.D.shape[0]

        if not self.beta_start:
            beta = self.y
        else:
            beta = self.beta_start

        q = _soft_threshold(self.y, omega1, opt_type="fused")
        if opt_type == "group":
            R = [q[D] for D in D_list]
            counts = Counter(D_list)
        else:
            r = self.D @ q
        if np.max(np.absolute(q)) == 0:
            beta = q
            i = 0
            conv_crit = 0
            best_beta = q

            self.beta = beta
            self.q = q
            self.r = r
            self.iter = i
            self.conv_crit = conv_crit
            self.best_beta=best_beta

            return beta, q, r, i, conv_crit, best_beta

        qk = self.q
        rk = self.r
        u = np.zeros(n)
        if opt_type == "group":
            v = np.zeros((n, G))
        else:
            v = np.zeros(m)
        i = 0
        phi_beta_k = 0.5*np.sum(self.y-beta) + omega1*np.sum(np.abs(beta))
        if opt_type == "fused":
            phi_beta_k += omega2*np.sum(np.absolute(self.D @ beta))
        elif opt_type == "group":
            beta2 = beta ** 2
            phi_beta_k += np.cross(G_penalty_factors, np.array([beta2[D] for D in D_list]))
        else:
            phi_beta_k +=  omega2*_gl_penalty(self.D @ beta)
        conv_crit = np.infty
        sk = np.infty
        resk = np.infty
        best_beta = beta
        best_phi = phi_beta_k

        while (resk > tol or sk > tol) and i <= self.max_iter:
            aux = self.y-u + self.rho*q
            if opt_type == "group":
                aux += np.cross(self.D, self.rho*r-v)
                beta = aux / (1+G)*self.rho
            else:
                aux += self.rho*np.sum(R) - np.sum(V)
                beta = aux / (1+3*self.rho)
            Dbeta = self.D @ beta

            # update q
            q = _soft_threshold(beta+u/self.rho, omega1/self.rho, opt_type="fused")

            # update r
            Dbetavrho = Dbeta + v/self.rho
            if opt_type == "fused":
                r = _soft_threshold(Dbetavrho, omega2/self.rho, opt_type)
            else:
                r = _soft_threshold(Dbetavrho, omega2/self.rho, opt_type)

            u = u + self.rho * (beta-q)
            v = v + self.rho * (Dbeta-r)

            # update convergence criteria
            phi_beta_k1 = 0.5*np.sum(self.y-beta) + omega1*np.sum(np.abs(beta))
            if opt_type == "fused":
                phi_beta_k1 += omega2*np.sum(np.absolute(Dbeta))
            else:
                phi_beta_k1 +=  omega2*_gl_penalty(self.D, nodes)
            sk = self.rho * np.max(np.absolute(q-qk)) + np.max(np.absolute(np.cross(self.D, r-rk)))

            res1k = np.sqrt(np.sum(beta-q))
            res2k = np.sqrt(np.sum(Dbeta-r))

            resk = res1k + res2k
            qk = q
            rk = r
            conv_crit = np.abs(phi_beta_k1 - phi_beta_k) / phi_beta_k
            phi_beta_k = phi_beta_k1

            if phi_beta_k1 < best_phi:
                best_beta = beta
                best_phi = phi_beta_k
                break

            i += 1

        if opt_type == "fused":
            phi_q =  0.5*np.sum(self.y-beta) + omega1*np.sum(np.abs(beta)) + omega2*np.sum(np.absolute(self.D @ beta))
            whichm = np.argmin([phi_beta_k1, best_phi, phi_q])
            if whichm == 1:
                best_beta = beta
            elif whichm == 3:
                best_beta = q
        else:
            beta_q = beta * (q == 0).astype(int)
            0.5*np.sum(self.y-beta_q) + omega1*np.sum(np.abs(beta_q)) + omega2*_gl_penalty(self.D @ beta_q)
        whichm = np.argmin([phi_beta_k1, best_phi, beta_q])
        if whichm == 1:
            best_beta = beta
        elif whichm == 3:
            best_beta = beta_q

        self.beta = beta
        self.q = q
        self.r = r
        self.iter = i
        self.conv_crit = conv_crit
        self.best_beta = best_beta

        return beta, q, r, i, conv_crit, best_beta

    def fit(self):
        pass
