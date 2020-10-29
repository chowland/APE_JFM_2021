######
# Import necessary libraries
######
import numpy as np
import h5py
import pyfftw

fname = "out.h5"

######
# Set up FFT-powered derivatives
######

# Read size of buoyancy field
with h5py.File(fname,"r") as f:
    nz, ny, nx = f["TH1"].shape

# Vertical coordinate vector
yvec = 2*np.pi*np.arange(ny)/ny

# Imaginary wavenumber vectors
CIKX = np.reshape(1j*np.arange(nx//2+1),(nx//2+1,1))*ny/nx
CIKY = 1j*np.arange(ny//2+1)
CIKZ = 1j*np.arange(nz//2+1)*ny/nz

# x-derivative
Ax=pyfftw.empty_aligned((nx,ny),dtype='float64')
Bx=pyfftw.empty_aligned((nx//2+1,ny),dtype='complex128')
ftx=pyfftw.FFTW(Ax, Bx, axes=(0,), direction='FFTW_FORWARD',threads=8)
iftx=pyfftw.FFTW(Bx, Ax, axes=(0,), direction='FFTW_BACKWARD',threads=8)
def ddx(S1):
    Ax[:,:] = S1
    ftx()
    Bx[:,:] = CIKX*Bx
    iftx()
    return Ax.copy()

# y-derivative (vertical)
Ay=pyfftw.empty_aligned((nx,ny),dtype='float64')
By=pyfftw.empty_aligned((nx,ny//2+1),dtype='complex128')
fty=pyfftw.FFTW(Ay, By, axes=(1,), direction='FFTW_FORWARD',threads=8)
ifty=pyfftw.FFTW(By, Ay, axes=(1,), direction='FFTW_BACKWARD',threads=8)
def ddy(S1):
    Ay[:,:] = S1
    fty()
    By[:,:] = CIKY*By
    ifty()
    return Ay.copy()

# z-derivative
Az=pyfftw.empty_aligned((nx,nz),dtype='float64')
Bz=pyfftw.empty_aligned((nx,nz//2+1),dtype='complex128')
ftz=pyfftw.FFTW(Az, Bz, axes=(1,), direction='FFTW_FORWARD',threads=8)
iftz=pyfftw.FFTW(Bz, Az, axes=(1,), direction='FFTW_BACKWARD',threads=8)
def ddz(S1):
    Az[:,:] = S1
    ftz()
    Bz[:,:] = CIKZ*Bz
    iftz()
    return Az.copy()

# 1D vertical derivative
As=pyfftw.empty_aligned(ny,dtype='float64')
Bs=pyfftw.empty_aligned(ny//2+1,dtype='complex128')
fts=pyfftw.FFTW(As, Bs, direction='FFTW_FORWARD')
ifts=pyfftw.FFTW(Bs, As, direction='FFTW_BACKWARD')
def dds(S1):
    As[:] = S1
    fts()
    Bs[:] = CIKY*Bs
    ifts()
    return As.copy()

######
# Preallocate values to be recorded
######
z0, dBdy, tmpFd = np.zeros((nx,nz)), np.zeros((nx,nz)), np.zeros((nx,nz))
EP, EB, ES, EA = 0, 0, 0, 0
Ea, Eb = 0, 0
b_mean, z1_mean = 0, 0
Kd, M, Fd = 0, 0, 0
tii = 0

Re = 8000   # Prescribe the Reynolds number
Dp = 1/Re

######
# Load buoyancy field and calculate boundary
######
with h5py.File(fname,"r") as f:
    TH = f["TH1"][()].T
    tii = f.attrs.__getitem__('Time')

# Pick the buoyancy contour to use as the boundary
con = np.pi

# Calculate buoyancy contour height
for j in range(nx):
    for k in range(nz):
        z0[j,k] = np.interp(con,yvec+TH[j,:,k],yvec)

# Shift contour to top of domain
z2 = z0 + 2*np.pi - con
# Record mean vertical deviation of isopycnal
z1_mean = np.mean(z2) - 2*np.pi

######
# Construct reference profile Z*(s)
######
# Prescribe bin width as grid spacing
bw = 2*np.pi/ny
# Obtain histogram of the buoyancy field
for k in range(nz):     # Loop over z to ease memory demand
    B = np.mod(yvec + TH[:,:,k] - con, 2*np.pi)
    hist, s_edges = np.histogram(B.flatten(), bins=bw*np.arange(ny+1))
    if k==0:
        H = hist
    else:
        H += hist
H = H/bw/np.sum(H)
z_r = np.zeros(ny)
for j in range(ny):
    z_r[j] = 2*np.pi*bw*np.sum(H[:j])
# Calculate derivative of reference profile
dzds = 1 + dds(z_r-yvec)

######
# Compute potential energy quantities
######
# Extend vectors to interpolate accurately at high buoyancy values
s_int = np.concatenate((yvec, [2*np.pi]))
z_int = np.concatenate((z_r, [2*np.pi]))
dz_int = np.concatenate((dzds, [dzds[0]]))

for k in range(nz):
    B = np.mod(yvec + TH[:,:,k] - con, 2*np.pi)
    Z = np.mod(yvec-con,2*np.pi) + 2*np.pi*(TH[:,:,k] + yvec < con)*(yvec>=con) - 2*np.pi*(TH[:,:,k] + yvec >= con)*(yvec<con)
    Zstar = np.interp(B, s_int, z_int)
    b_mean += np.mean(B)
    EP += np.mean(-B*Z)
    EB += np.mean(-B*Zstar)
b_mean = b_mean/nz
EP = EP/nz
EB = EB/nz
EB = EB - b_mean*z1_mean + 0.5*(2*np.pi+z1_mean)**2
ES = np.mean(z2**2)/2
EA = EP + ES - EB

######
# Local APE calculation
######
# Use periodicity to extend reference profile
xnum = 128
svec = np.arange(ny + 2*xnum + 1)*bw - bw*xnum
Zrvec = np.concatenate((z_r[-xnum:]-2*np.pi, z_r, z_r[:xnum+1]+2*np.pi))
dZrvec = np.concatenate((dzds[-xnum:], dzds, dzds[:xnum+1]))

# Construct integral over Z*
iZrvec = np.zeros(svec.size)
for j in range(svec.size):
    iZrvec[j] = np.trapz(Zrvec[:j+1], x=svec[:j+1])

for k in range(nz):
    B = np.mod(yvec + TH[:,:,k] - con, 2*np.pi)
    Z = np.mod(yvec-con,2*np.pi) + 2*np.pi*(TH[:,:,k] + yvec < con)*(yvec>=con) - 2*np.pi*(TH[:,:,k] + yvec >= con)*(yvec<con)
    Bstar = np.interp(np.mod(yvec-con,2*np.pi), z_int, s_int) \
        + 2*np.pi*(TH[:,:,k]+yvec<con)*(yvec>=con) - 2*np.pi*(TH[:,:,k]+yvec>=con)*(yvec<con)
    iB = np.interp(B, svec, iZrvec)
    iBstar = np.interp(Bstar, svec, iZrvec)
    locAPE =  Z*(Bstar - B) + iB - iBstar
    if k==0:
        np.save("locAPE", locAPE)
    Ea += np.mean(locAPE)

Ea = Ea/nz
Eb = EP + ES - Ea

######
# Compute diapycnal mixing and diffusivity
######
# Compute local buoyancy gradients
gradB2 = np.zeros(TH.shape)
for k in range(nz):
    gradB2[:,:,k] += 1 + ddy(TH[:,:,k])
    for j in range(nx):
        dBdy[j,k] = np.interp(z0[j,k], yvec, gradB2[j,:,k])
gradB2 = gradB2**2
for j in range(ny):
    gradB2[:,j,:] += ddz(TH[:,j,:])**2
for k in range(nz):
    gradB2[:,:,k] += ddx(TH[:,:,k])**2
    for j in range(nx):
        tmpFd[j,k] = np.interp(z0[j,k], yvec, gradB2[j,:,k])

# Record diffusive flux at the isopycnal boundary
tmpFd = tmpFd/dBdy/Re
Fd = np.mean(tmpFd)

# Record mean mixing rate and diffusivity
for k in range(nz):
    B = np.mod(yvec + TH[:,:,k] - con, 2*np.pi)
    dZdB = np.interp(B, s_int, dz_int)
    M += np.mean(gradB2[:,:,k]*dZdB)
    Kd += np.mean(gradB2[:,:,k]*dZdB**2)
    if k==0:
        np.save("gradB2", gradB2[:,:,k])
        np.save("dZdB", dZdB)
M = M/nz/Re - Dp
Kd = Kd/nz/Re

######
# Remove noise
# (This seemingly arises from taking modulo for buoyancy on the discrete grid)
######
noise = np.pi*(b_mean - z1_mean - np.pi)
EP = EP + 2*noise
EA = EA + noise
EB = EB + noise
Eb = Eb + 2*noise

######
# Save data to file
######
np.savez(
    "APE_data",
    P=EP, S=ES, B=EB, A=EA, E_B=Eb, E_A=Ea,
    M=M, Dp=Dp, Kd=Kd, b_mean=b_mean, z1_mean=z1_mean,
    z_r=z_r, tii=tii, Fd=Fd
)
# Overview of data saved
# P     "internal" potential energy
# S     surface potential energy
# B     background potential energy
# A     available potential energy
# E_B   average local BPE
# E_A   average local APE
# M     diapycnal mixing rate
# Dp    internal energy conversion rate
# Kd    diapycnal diffusivity
# b_mean    volume-averaged buoyancy
# z1_mean   mean deviation of boundary isopycnal
# z_r   background profile Z*(s)
# tii   time
# Fd    diffusive boundary flux