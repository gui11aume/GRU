##!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys


def onehot(sz, pos):
   o = np.zeros(sz); o[pos] = 1
   return o


class xent_loss:
   @staticmethod
   def fn(x,y):
      return - np.dot(x, np.log(y))
   @staticmethod
   def dy(x,y):
      # Here 'y' is a probability.
      return - np.divide(x, y)
   @staticmethod
   def softmax_dy(x, y):
      # Here 'y' is an energy.
      return y - x
      
class softmax:
   @staticmethod
   def fn(x):
      e = np.exp(x - np.max(x))
      return e / e.sum()
   @staticmethod
   def jac(x):
      tmp = x.reshape((-1,1))
      return np.diag(x) - np.outer(tmp, tmp.T)

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


class FwdParams:
   def __init__(self, x = [], h = []):
      self.x  =  x    # Input.
      self.y  = [ ]   # Output.
      self.z  = [ ]   # Update.
      self.r  = [ ]   # Reset
      self.hh = [ ]   # Hidden (hat).
      self.h  =  h    # Hidden.

   def apply_softmax(self):
      self.y  = [softmax.fn(x) for x in self.y]
      return self


class GradParams:
   def __init__(self, xsz, hsz, ysz):
      self.dWz = np.zeros((hsz, xsz))
      self.dWr = np.zeros((hsz, xsz))
      self.dWh = np.zeros((hsz, xsz))
      self.dWy = np.zeros((ysz, hsz))

      self.dUz = np.zeros((hsz, hsz))
      self.dUr = np.zeros((hsz, hsz))
      self.dUh = np.zeros((hsz, hsz))

      self.dbz = np.zeros(hsz)
      self.dbr = np.zeros(hsz)
      self.dbh = np.zeros(hsz)
      self.dby = np.zeros(ysz)
   
   def apply_softmax_grad(self, L):
      self.dx = [np.dot(softmax.jac(x), y) for (x,y) in zip(L, self.dx)]
      return self



class GRULayer:
   def __init__(self, xsz, hsz, ysz):
      # Dimensions.
      self.xsz = xsz
      self.hsz = hsz
      self.ysz = ysz

      self.h  = np.zeros(hsz)  # Hidden state.

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


   def get_h(self):
      return self.h

   def set_h(self, h):
      self.h = h

   def reset(self):
      self.h  = np.zeros(self.hsz)



   def forward(self, x):
      '''The parameter x is a list of 'np.array'. Outputs
      a 'FwdParams' object.'''

      fwd = FwdParams(x = x[:], h = [self.h])
      prevh = self.h

      for t in range(len(x)):
         # Equations below are the definition of GRU.
         z = sigmoid.fn(np.dot(self.Wz, x[t]) + \
               np.dot(self.Uz, prevh) + self.bz)
         r = sigmoid.fn(np.dot(self.Wr, x[t]) + \
               np.dot(self.Ur, prevh) + self.br)
         hh = tanh.fn(np.dot(self.Wh, x[t]) + \
               np.dot(self.Uh, np.multiply(r, prevh)) + self.bh)
         h = np.multiply(z, prevh) + np.multiply((1-z), hh)
         y = np.dot(self.Wy, h) + self.by

         fwd.z.append(z)
         fwd.r.append(r)
         fwd.hh.append(hh)
         fwd.h.append(h)
         fwd.y.append(y)

         prevh = h

      self.h = h

      # Return 'FwdParams' object.
      return fwd


   def backprop(self, fwd, dy):
      '''dy is a list of gradients in output of the GRU layer.'''

      grd = GradParams(self.xsz, self.hsz, self.ysz)
      grd.dx = [None] * len(dy)

      dhn = np.zeros(self.hsz)

      for t in reversed(range(len(dy))):
         # Layer output.
         grd.dby += dy[t]
         grd.dWy += np.outer(dy[t], fwd.h[t+1])

         # Intermediates.
         dh   = np.dot(self.Wy.T, dy[t]) + dhn
         dhh  = np.multiply(dh, (1-fwd.z[t]))
         dhhl = np.multiply(dhh, tanh.df(fwd.hh[t]))

         # Gradient for 'hhat' parameters.
         grd.dWh += np.outer(dhhl, fwd.x[t])
         grd.dUh += np.outer(dhhl, np.multiply(fwd.r[t], fwd.h[t]))
         grd.dbh += dhhl

         # Intermediates
         drh = np.dot(self.Uh.T, dhhl)
         dr  = np.multiply(drh, fwd.h[t])
         drl = dr * sigmoid.df(fwd.r[t])

         # Gradient for 'r' parameters.
         grd.dWr += np.outer(drl, fwd.x[t])
         grd.dUr += np.outer(drl, fwd.h[t])
         grd.dbr += drl

         # Intermediates
         dz  = np.multiply(dh, fwd.h[t] - fwd.hh[t])
         dzl = dz * sigmoid.df(fwd.z[t])

         # Gradient for 'z' parameters.
         grd.dWz += np.outer(dzl, fwd.x[t])
         grd.dUz += np.outer(dzl, fwd.h[t].T)
         grd.dbz += dzl

         # Gradient for next 'h'
         dh1 = np.dot(self.Uz.T, dzl)
         dh2 = np.dot(self.Ur.T, drl)
         dh3 = np.multiply(dh, fwd.z[t])
         dh4 = np.multiply(drh, fwd.r[t])

         dhn = dh1 + dh2 + dh3 + dh4

         # Send down the gradient.
         dxr = np.dot(self.Wr.T, drl)
         dxz = np.dot(self.Wz.T, dzl)
         grd.dx[t] = dxr + dxz

      return grd


   # TODO: write this is a cleaner way.
   def update(self, grd, learning_rate=0.02):
      # Update model with adagrad (stochastic) gradient descent
      plist = [self.Wy, self.Wh, self.Wr, self.Wz, self.Uh,
         self.Ur, self.Uz, self.by, self.bh, self.br, self.bz]
      dlist = [grd.dWy, grd.dWh, grd.dWr, grd.dWz, grd.dUh,
         grd.dUr, grd.dUz, grd.dby, grd.dbh, grd.dbr, grd.dbz]
      mlist = [self.mdWy, self.mdWh, self.mdWr, self.mdWz, self.mdUh,
         self.mdUr, self.mdUz, self.mdby, self.mdbh, self.mdbr, self.mdbz]
      for i in range(len(plist)):
          np.clip(dlist[i], -5, 5, out=dlist[i])
          mlist[i] += dlist[i] * dlist[i]
          # Add small term for numerical stability.
          plist[i] += -learning_rate * dlist[i] / np.sqrt(mlist[i] + 1e-8)
      return


def sample(G1, G2, G3, x):
   charidx = list()
   initl1 = G1.get_h()
   initl2 = G2.get_h()
   initl3 = G3.get_h()
   for t in range(4100):
      # Three-layer version.
      oG1 = G1.forward([x])
      oG2 = G2.forward(oG1.y)
      oG3 = G3.forward(oG2.y).apply_softmax()
      (p,) = oG3.y
      # Choose next char according to the distribution
      idx = np.random.choice(range(len(x)), p=p.ravel())
      x = np.zeros_like(x)
      x[idx] = 1
      charidx.append(idx)
   G1.set_h(initl1)
   G2.set_h(initl2)
   G3.set_h(initl3)
   return charidx[100:]



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

   G1 = GRULayer(sigma, sigma, sigma)
   G2 = GRULayer(sigma, sigma, sigma)
   G3 = GRULayer(sigma, sigma, sigma)

   seq_length = 25
   Si = range(seq_length)
   print_interval = 200


   # Pre-training 1
   p = 1
   seq = [onehot(sigma, char_to_idx[txt[0]])]
   for n in range(50000):
      # Check if end of text is reached.
      if p + seq_length > txtlen:
         G1.reset()
         G2.reset()
         G3.reset()
         p = 1
         seq = [onehot(sigma, char_to_idx[txt[0]])]
         
      # Get input and target sequence
      for i in range(p, p+seq_length):
         seq.append(onehot(sigma, char_to_idx[txt[i]]))
      
      # Three-layer version.
      oG1 = G1.forward(seq[:-1]).apply_softmax()

      loss_grad = [xent_loss.softmax_dy(seq[i], oG1.y[i]) for i in Si]
      dG1 = G1.backprop(oG1, loss_grad)

      G1.update(dG1)

      if n % print_interval == 0:
         # Three-layer version.
         #losses = sum([xent_loss.fn(seq[i+1], oG3.y[i]) for i in Si])
         losses = sum([xent_loss.fn(seq[i], oG1.y[i]) for i in Si])
         print 'iteration: %d, loss: %f' % (n, losses)
         x = np.zeros(sigma)
         x[np.random.randint(sigma) % sigma] = 1
         #stxt = sample(G1, G2, G3, x)
         #print ''.join([idx_to_char[x] for x in stxt])

      # Keep last character for next round.
      seq = seq[-1:]

      # Prepare for next iteration
      p += seq_length
      n += 1

   # Pre-training 1+2
   seq = [onehot(sigma, char_to_idx[txt[0]])]
   for n in range(50000):
      # Check if end of text is reached.
      if p + seq_length > txtlen:
         G1.reset()
         G2.reset()
         G3.reset()
         p = 1
         seq = [onehot(sigma, char_to_idx[txt[0]])]
         
      # Get input and target sequence
      for i in range(p, p+seq_length):
         seq.append(onehot(sigma, char_to_idx[txt[i]]))
      
      # Three-layer version.
      oG1 = G1.forward(seq[:-1])
      oG2 = G2.forward(oG1.y).apply_softmax()

      loss_grad = [xent_loss.softmax_dy(seq[i], oG2.y[i]) for i in Si]
      dG2 = G1.backprop(oG2, loss_grad)
      dG1 = G1.backprop(oG1, dG2.dx)

      G1.update(dG1)
      G2.update(dG2)

      if n % print_interval == 0:
         # Three-layer version.
         #losses = sum([xent_loss.fn(seq[i+1], oG3.y[i]) for i in Si])
         losses = sum([xent_loss.fn(seq[i], oG2.y[i]) for i in Si])
         print 'iteration: %d, loss: %f' % (n, losses)
         x = np.zeros(sigma)
         x[np.random.randint(sigma) % sigma] = 1
         #stxt = sample(G1, G2, G3, x)
         #print ''.join([idx_to_char[x] for x in stxt])

      # Keep last character for next round.
      seq = seq[-1:]

      # Prepare for next iteration
      p += seq_length
      n += 1


   # Training proper.
   print_interval = 10000
   p = 1
   n = 0
   seq = [onehot(sigma, char_to_idx[txt[0]])]
   while True:
      # Check if end of text is reached.
      if p + seq_length > txtlen:
         G1.reset()
         G2.reset()
         G3.reset()
         p = 1
         seq = [onehot(sigma, char_to_idx[txt[0]])]
         
      # Get input and target sequence
      for i in range(p, p+seq_length):
         seq.append(onehot(sigma, char_to_idx[txt[i]]))
      
      # Three-layer version.
      oG1 = G1.forward(seq[:-1])
      oG2 = G2.forward(oG1.y)
      oG3 = G3.forward(oG2.y).apply_softmax()


      loss_grad = [xent_loss.softmax_dy(seq[i+1], oG3.y[i]) for i in Si]
      dG3 = G3.backprop(oG3, loss_grad)
      dG2 = G2.backprop(oG2, dG3.dx)
      dG1 = G1.backprop(oG1, dG2.dx)

      G1.update(dG1)
      G2.update(dG2)
      G3.update(dG3)

      if n % print_interval == 0:
         # Three-layer version.
         losses = sum([xent_loss.fn(seq[i+1], oG3.y[i]) for i in Si])
         print 'iteration: %d, loss: %f' % (n, losses)
         x = np.zeros(sigma)
         x[np.random.randint(sigma) % sigma] = 1
         stxt = sample(G1, G2, G3, x)
         print ''.join([idx_to_char[x] for x in stxt])

      # Keep last character for next round.
      seq = seq[-1:]

      # Prepare for next iteration
      p += seq_length
      n += 1


if __name__ == '__main__':
   main(sys.argv[1])


