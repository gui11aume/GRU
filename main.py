#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys


def onehot(sz, pos):
   o = np.zeros(sz); o[pos] = 1
   return o


class loss:
   @staticmethod
   def fn(x,y):
      return - np.log(np.dot(softmax.fn(x), y))
   @staticmethod
   def dx(x,y):
      return softmax.fn(x) - y
      
class softmax:
   @staticmethod
   def fn(x):
      e = np.exp(x - np.max(x))
      return e / e.sum()

class sigmoid:
   @staticmethod
   def fn(x):
      return 1 / (1 + np.exp(-x))
   @staticmethod
   def df(x):
      return x * (1-x)

class tanh:
   @staticmethod
   def fn(x):
      return np.tanh(x)
   @staticmethod
   def df(x):
      return 1 - x**2


class GRULayer:

   def __init__(self, xsz, hsz, ysz):
      # Dimensions.
      self.xsz = xsz
      self.hsz = hsz
      self.ysz = ysz

      # Hidden states through time.
      self.x  = list()           # Input vector.
      self.z  = list()           # Update gate.
      self.r  = list()           # Reset gate.
      self.hh = list()           # Hidden intermediate.
      self.h  = [np.zeros(hsz)]  # Hidden state.
      self.y  = list()           # Outputs.

      # Parameters.
      self.Wz = np.random.rand(hsz, xsz) * 0.1 - 0.05
      self.Uz = np.random.rand(hsz, hsz) * 0.1 - 0.05
      self.bz = np.zeros(hsz) 

      self.Wr = np.random.rand(hsz, xsz) * 0.1 - 0.05
      self.Ur = np.random.rand(hsz, hsz) * 0.1 - 0.05
      self.br = np.zeros(hsz) 

      self.Wh = np.random.rand(hsz, xsz) * 0.1 - 0.05
      self.Uh = np.random.rand(hsz, hsz) * 0.1 - 0.05
      self.bh = np.zeros(hsz) 

      self.Wy = np.random.rand(ysz, hsz) * 0.1 - 0.05
      self.by = np.zeros(ysz) 

      # Gradients.
      self.zerograd()
      self.dhn = np.zeros(self.hsz)

      # Adagrad parameters.
      self.mdWz = np.zeros_like(self.Wz)
      self.mdWr = np.zeros_like(self.Wr)
      self.mdWh = np.zeros_like(self.Wh)
      self.mdWy = np.zeros_like(self.Wy)

      self.mdUz = np.zeros_like(self.Uz)
      self.mdUr = np.zeros_like(self.Ur)
      self.mdUh = np.zeros_like(self.Uh)

      self.mdbz = np.zeros_like(self.bz)
      self.mdbr = np.zeros_like(self.br)
      self.mdbh = np.zeros_like(self.bh)
      self.mdby = np.zeros_like(self.by)


   def flush(self):
      self.h  = [np.zeros(self.hsz)]
      self.dhn = np.zeros(self.hsz)


   def zerograd(self):
      self.dWz = np.zeros_like(self.Wz)
      self.dWr = np.zeros_like(self.Wr)
      self.dWh = np.zeros_like(self.Wh)
      self.dWy = np.zeros_like(self.Wy)

      self.dUz = np.zeros_like(self.Uz)
      self.dUr = np.zeros_like(self.Ur)
      self.dUh = np.zeros_like(self.Uh)

      self.dbz = np.zeros_like(self.bz)
      self.dbr = np.zeros_like(self.br)
      self.dbh = np.zeros_like(self.bh)
      self.dby = np.zeros_like(self.by)


   def get_state(self):
      return self.h[:]


   def set_state(self, h):
      self.h = h


   def forward(self, x):
      # Update and reset gates.
      z = sigmoid.fn(np.dot(self.Wz, x) + \
            np.dot(self.Uz, self.h[-1]) + self.bz)
      r = sigmoid.fn(np.dot(self.Wr, x) + \
            np.dot(self.Ur, self.h[-1]) + self.br)

      # GRU units.
      hh = tanh.fn(np.dot(self.Wh, x) + \
            np.dot(self.Uh, np.multiply(r, self.h[-1])) + self.bh)
      h = np.multiply(z, self.h[-1]) + np.multiply((1-z), hh)
      y = np.dot(self.Wy, h) + self.by

      # Write internal states.
      self.x.append(x)
      self.z.append(z)
      self.r.append(r)
      self.hh.append(hh)
      self.y.append(y)
      self.h.append(h)

      return y


   def backprop(self, dy):
      '''The parameter 'dy' must be a list of gradients fed
      into the GRU layer.'''

      self.zerograd()
      dhn = np.zeros(self.hsz)

      T = len(dy)
      dx = list()

      for t in range(T-1,-1,-1):
         # Layer output.
         self.dby += dy[t]
         self.dWy += np.outer(dy[t], self.h[t+1])

         # Intermediates.
         dh   = np.dot(self.Wy.T, dy[t]) + dhn
         dhh  = np.multiply(dh, (1-self.z[t]))
         dhhl = np.multiply(dhh, tanh.df(self.hh[t]))

         # Gradient for 'hhat' parameters.
         self.dWh += np.outer(dhhl, self.x[t])
         self.dUh += np.outer(dhhl, np.multiply(self.r[t], self.h[t]))
         self.dbh += dhhl

         # Intermediates
         drh = np.dot(self.Uh.T, dhhl)
         dr  = np.multiply(drh, self.h[t])
         drl = dr * sigmoid.df(self.r[t])

         # Gradient for 'r' parameters.
         self.dWr += np.outer(drl, self.x[t])
         self.dUr += np.outer(drl, self.h[t])
         self.dbr += drl

         # Intermediates
         dz  = np.multiply(dh, self.h[t] - self.hh[t])
         dzl = dz * sigmoid.df(self.z[t])

         # Gradient for 'z' parameters.
         self.dWz += np.outer(dzl, self.x[t])
         self.dUz += np.outer(dzl, self.h[t].T)
         self.dbz += dzl

         # Gradient for next 'h'
         dh1 = np.dot(self.Uz.T, dzl)
         dh2 = np.dot(self.Ur.T, drl)
         dh3 = np.multiply(dh, self.z[t])
         dh4 = np.multiply(drh, self.r[t])

         dhn = dh1 + dh2 + dh3 + dh4

         # Send down the gradient.
         dxr = np.dot(self.Wr.T, drl)
         dxz = np.dot(self.Wz.T, dzl)
         dx.append(dxr+dxz)

      # Erase internal variables.
      self.x  = list()
      self.z  = list()
      self.r  = list()
      self.hh = list()
      self.h  = self.h[-1:]
      self.y  = list()

      return dx


   def update(self, learning_rate=0.1):
      # Update model with adagrad (stochastic) gradient descent
      plist = [self.Wy, self.Wh, self.Wr, self.Wz, self.Uh,
            self.Ur, self.Uz, self.by, self.bh, self.br, self.bz]
      dlist = [self.dWy, self.dWh, self.dWr, self.dWz, self.dUh,
            self.dUr, self.dUz, self.dby, self.dbh, self.dbr, self.dbz]
      mlist = [self.mdWy, self.mdWh, self.mdWr, self.mdWz, self.mdUh,
            self.mdUr, self.mdUz, self.mdby, self.mdbh, self.mdbr, self.mdbz]
      for i in range(len(plist)):
          np.clip(dlist[i], -5, 5, out=dlist[i])
          mlist[i] += dlist[i] * dlist[i]
          # Add small term for numerical stability.
          plist[i] += -learning_rate * dlist[i] / np.sqrt(mlist[i] + 1e-8)
      return


def sample(RNN, x):
   charidx = list()
   intlstate = RNN.get_state()
   for t in range(1000):
      p = softmax.fn(RNN.forward(x))
      # Choose next char according to the distribution
      idx = np.random.choice(range(len(x)), p=p.ravel())
      x = np.zeros_like(x)
      x[idx] = 1
      charidx.append(idx)
   RNN.set_state(intlstate)
   return charidx



def main(fname):
   '''Read data from an input text and train a GRU.'''
   
   np.random.seed(123)

   with open(fname) as f: 
      txt = f.read()

   alphabet = sorted(list(set(txt)))
   txtlen, sigma = len(txt), len(alphabet)
   print('data has %d characters, %d unique.' % (txtlen, sigma))
   char_to_idx = { ch:i for i,ch in enumerate(alphabet) }
   idx_to_char = { i:ch for i,ch in enumerate(alphabet) }

   hsz = sigma
   G = GRULayer(sigma, hsz, sigma)

   seq_length = 25
   Si = range(seq_length)
   learning_rate = 1e-1
   print_interval = 1000

   p = 1
   n = 0
   seq = [onehot(sigma, char_to_idx[txt[0]])]
   while True:
      # Check if end of text is reached.
      if p + seq_length > txtlen:
         G.flush()
         seq = [onehot(sigma, char_to_idx[txt[0]])]
         
      # Get input and target sequence
      for i in range(p, p+seq_length):
         seq.append(onehot(sigma, char_to_idx[txt[i]]))

      out =  [G.forward(seq[i]) for i in Si]
      losses = sum([loss.fn(out[i], seq[i+1]) for i in Si])
      G.backprop([loss.dx(out[i], seq[i+1]) for i in Si])
      G.update()

      # Keep last character for next round.
      seq = seq[-1:]

      if n % print_interval == 0:
         x = np.zeros(sigma)
         x[np.random.randint(sigma) % sigma] = 1
         stxt = sample(G, x)
         print ''.join([idx_to_char[x] for x in stxt])

      # Prepare for next iteration
      p += seq_length
      n += 1


if __name__ == '__main__':
   main(sys.argv[1])


