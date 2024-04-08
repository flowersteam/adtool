import numpy as np
import typing as t

import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

  def show(self, **kw):
      self.close()
      fn = self.params['filename']
      display(mvp.ipython_display(fn, **kw))


import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

torch.set_default_dtype(torch.float32)

def sigmoid(x):
    return 0.5 * (torch.tanh(x / 2) + 1)

def ker_f(x, a, w, b):
    return (b * torch.exp( - (x[..., None] - a)**2 / w)).sum(-1)

def bell(x, m, s):
    return torch.exp(-((x-m)/s)**2 / 2)

def growth(U, m, s):
    return bell(U, m, s)*2-1

kx = torch.tensor([
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
])[None, None, :, :].double()

ky = torch.transpose(kx, 2, 3)



def sobel_x(A):
    A_bat = A[None, None, ...]
    return torch.stack([F.conv2d(A_bat[..., c], kx, padding='same')[...,None]
                    for c in range(A.shape[-1])], dim=-1)[0, 0, ...]

def sobel_y(A):
    A_bat = A[None, None, ...]
    #get type of A_bat[..., c]

    return torch.stack([F.conv2d(A_bat[..., c], ky, padding='same')[...,None]
                    for c in range(A.shape[-1])], dim=-1)[0, 0, ...]

def sobel(A):
  #  A_3d = A[:, :, ,;None]  # add an extra dimension to make A a 3D tensor
    return torch.cat((sobel_y(A), sobel_x(A)), dim=-2)



def get_kernels(SX, SY, nb_k, params):
    mid = SX//2
    Ds = [ torch.linalg.norm(torch.mgrid[-mid:mid, -mid:mid], axis=0) /
          ((params['R']+15) * params['r'][k]) for k in range(nb_k) ]  # (x,y,k)
    K = torch.stack([sigmoid(-(D-1)*10) * ker_f(D, params["a"][k], params["w"][k], params["b"][k])
                    for k, D in zip(range(nb_k), Ds)], dim=-1)
    nK = K / torch.sum(K, axis=(0,1), keepdims=True)
    return nK

def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s]*n
                c1[t] = c1[t] + list(range(i, i+n))
            i+=n
    return c0, c1

def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for c in range(C)]
