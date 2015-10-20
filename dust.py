"""
Written by Jacob N. McLane
Last Modified Oct. 20th, 2015

Code for generating a "toy model" of dust rings in a star system.
Later parts (parts 4 and 5) assume that the dust ring you're analyzing lies at
either 10 or 130 AU, and that the star being modeled is Fomalhaut, because I
got lazy.

Code works in,and returns values in (unless otherwise noted), mks units.

*********Synatax*********

python dust.py Star_radius Effective_temp Distance_from_star

ex.

python dust.py 1.842 8590 10

    INPUTS

    Star_radius = Radius of star in solar radii

    Effective_temp = Effective temperature of star

    Distance_from_star = Distance from star of dust in AU

===================================================

    ******   CODE CITED  **********************************

    This code makes use of Bruce T. Draine's dust models which are
    avaliable at : http://www.astro.princeton.edu/~draine/dust/dust.diel.html

    Specifically, the Smoothed UV Astronomical Silicate  (su-vSil_21) models
    for 0.1, 1, and 10 micron dust particles. Copies of the relavent dust models
    *should* be present in the same github directory as this file.

    *******************************************************************
    *  THIS PROGRAM WAS WRITTEN FOR USE WITH PYTHON 3.4,    *
    *   COMPATIBILITY WITH OTHER VERSIONS NOT GUARANTEED   * 
    ********************************************************************
    
"""
###############################
### IMPORT PACKAGES USED ########
###############################

from math import *
import numpy as np
import scipy.integrate as integrate
import sys
import time

start_time = time.time() #Timer to time program

###############################
### END PACKAGES USED ##########
###############################

# Read in user-defined values and set constants to mks values

R=float(sys.argv[1])
Teff=float(sys.argv[2])
d=float(sys.argv[3])

R=R*0.0046491                                        #convert stellar radius to AU
h=6.626070040e-34                                #Planck constant
k=1.3806488e-23                                    #Boltzmann Constant
c=2.99792458e8                                      #speed of light
sigma=5.67036713e-8                            #Stefan-Boltzmann constant
b=0.0028977729                                     #Wein's displacement constant
G=6.67408e-11                                       #Gravitational constant

###############################
### BEGIN FUNCTIONS #############
###############################
#Function to extract Q and wavelengths from our dust parameter files

def fext(x):
    fdust=np.loadtxt(str(x)+'_micron_dust.txt', dtype={'names':('wav','Q','j1','j2'),'formats' : ('f4','f4','f4','f4')},skiprows=2)
    wav=fdust['wav']
    wav=wav*1e-6
    wav=list(reversed(wav))
    Q=fdust['Q']
    Q=list(reversed(Q))
    return wav,Q

###############################
#Wavelength formulation for a Blackbody

def p(x,T):
    return ((2*h*pow(c,2))/pow(x,5))*1/(exp((h*c)/(x*k*T))-1)

plank=np.vectorize(p) #Must be vectorized to work with arrays

###############################
#Convert blackbody values from W m^-3 to janskys

def mjan(bb,wavl):
    cor=np.multiply(wavl,wavl)/c
    j=np.multiply(cor,bb)*1e26
    return j

###############################
#Calculate power absorbed by a dust particle

def edust(x,bb,rat,wavelength):
    if x == 'abs':
        wav,Q=fext(1)
        Q=np.ones(len(Q))
        r=1e-3
    else:
        wav,Q=fext(x)
        r=x*1e-6

    Qint=np.interp(wavelength,wav,Q)
    flux=np.multiply(bb,rat)
    eabs=np.multiply(Qint,flux)
    en=integrate.trapz(eabs,wavelength)*pow(r,2)/4
    return en

###############################
#Find equilibrium temperature of a grain

def tgrain(power,x,ratio):
    t_increase=100 #Temperature increase stepsize
    if x == 'abs':
        wav,Q=fext(1)
        Q=np.ones(len(Q))
        r=1e-3
        itt=500 #max number of fitting itterations
    else:
        wav,Q=fext(x)
        r=x*1e-6
        itt=500
    
    T4=power/(4*pi*pow(r,2)*sigma)
    temp=pow(T4,1/4)

    print('Starting temperature fitting.')
    for i in range(500):
        wavelength,ang=findwav(-5,10)
        Qint=np.interp(wavelength,wav,Q)
        black_body=plank(wavelength,temp)
        eabs=np.multiply(Qint,black_body)
        energy=integrate.trapz(eabs,wavelength)*pi*pow(r,2)
        energy_ratio=energy/power
        
        if abs(energy_ratio-1) <= 1e-7: #stop if power in and pover out differ by 1 part per 1e7 or less
            print('Fitting complete. Convergence took '+str(i+1)+' iterations.')
            break
        elif energy_ratio < 1:
            temp=temp+t_increase
        else:
            t_increase=t_increase/2
            temp=temp-t_increase

    print('Energy out is '+str(energy)+' watts.')
    
    jansky1=mjan(black_body,wavelength)
    jansky2=mjan(eabs,wavelength)
    micron=ang/1e4
    
    log=open('bbcurve_'+str(x)+'_dust_'+str(int(d))+'AU.txt','w')

    for i in range(len(jansky1)):
        print(micron[i],jansky1[i],jansky2[i],Qint[i], file=log) #wavelength of output file is in microns

    log.close()
    return temp
    
###############################
#Generate a logspace vector of wavelength values, return in meters and angstroms 

def findwav(factor1,factor2):
    wavelength=np.logspace(factor1,factor2,100000)
    angstroms=wavelength*1e10
    return wavelength,angstroms

###############################
#Find mass of dust grains necessary to generate observed Fomalhaut SED

def findM(peak,model,Distance,r):
    ly=9.4607e15
    Distance=Distance*ly
    flux=peak/pow(Distance,2)*pow(r,2)*pi
    N=model/flux
    rho=.002*1e6
    V=(4/3)*pi*pow(r,3)
    M=N*rho*V
    return N,M

###############################
#Calculate radiation pressure, gravitational force, and infall time

def findF(Pin,R,r):
    V=(4/3)*pi*pow(r,3)
    rho=.002*1e6
    mp=rho*V
    Ms=1.92*1.98855e30
    Rd=R*1.4960e11
    Fgr=(G*Ms*mp)/pow(Rd,2)
    Frad=Pin/c
    beta=Frad/Fgr
    To=(4e2/beta)*pow(R,2)
    return To
    
###############################
### END FUNCTIONS ##############
###############################

#___________Part 1__________________
#__________________________________
#Finds the observed blackbody spectrum of star at input radius

lam,ang=findwav(-8,10)

bbcurve=plank(lam,Teff)
jan=mjan(bbcurve,lam)
ratio=pi*pow(R/d,2)

out=np.multiply(ratio,jan)

#-----Make Output File -----------

log=open('out_'+str(int(d))+'AU.txt','w')

for i in range(len(out)):
    print(ang[i],out[i], file=log)

log.close()

#------------------------------

print('Part 1 Finished')

#__________________________________

#___________Part 2__________________
#__________________________________
#Finds power absorbed by a 0.1, 1, and 10 micron dust grain at
#given radius. Also Finds power absorbed by 1 mm perfect absorber.

en01=edust(0.1,bbcurve,ratio,lam)
en1=edust(1,bbcurve,ratio,lam)
en10=edust(10,bbcurve,ratio,lam)
ena=edust('abs',bbcurve,ratio,lam)

print('Part 2 Finished')

#__________________________________

#___________Part 3__________________
#__________________________________
#Finds equilibrium temperature of each grain type such that
#power in = power out for the given orbital radius.

T01=tgrain(en01,0.1,ratio)
T1=tgrain(en1,1,ratio)
T10=tgrain(en10,10,ratio)
Ta=tgrain(ena,'abs',ratio)

#-----Make Output File -----------

log=open('Eabsorbed_'+str(int(d))+'AU.txt','w')

print('A 0.1 micron grain at '+str(d)+' AU absorbed '+str(en01)+' watts.', file=log)
print('The temp of a 0.1 micron grain at '+str(d)+' AU is '+str(T01)+' kelvin.', file=log)
print(' ', file=log)
print('A 1 micron grain at '+str(d)+' AU absorbed '+str(en1)+' watts.', file=log)
print('The temp of a 1 micron grain at '+str(d)+' AU is '+str(T1)+' kelvin.', file=log)
print(' ', file=log)
print('A 10 micron grain at '+str(d)+' AU absorbed '+str(en10)+' watts.', file=log)
print('The temp of a 10 micron grain at '+str(d)+' AU is '+str(T10)+' kelvin.', file=log)
print(' ', file=log)
print('A 1 millimeter perfect absorber at '+str(d)+' AU absorbed '+str(ena)+' watts.', file=log)
print('The temp of a 1 mm perfect absorber at '+str(d)+' AU is '+str(Ta)+' kelvin.', file=log)

log.close()

#------------------------------

print('Part 3 Finished')

#__________________________________

#___________Part 4__________________
#__________________________________
#Finds the approximate number of dust grains necessary to produce the SED
#presented in Su et al. 2013. Only 0.1 and 1 micron dust grains are considered
#for the 10 AU distance, and only 10 micron dust and the perfect absorber are
#considered for the 130 AU distance. Reasoning for this is explained in the report.

if d==10:
    N01,M01=findM(2e13,0.5,25.13,r=0.1e-6)
    N1,M1=findM(7e13,0.5,25.13,r=1e-6)
    log=open('mass_'+str(int(d))+'AU.txt','w')
    print('For a 0.1 micron at '+str(d)+'AU, there are '+str(N01)+' grains with a mass of '+str(M01)+'kg.' , file=log)
    print(' ', file=log)
    print('For a 1 micron at '+str(d)+'AU, there are '+str(N1)+' grains with a mass of '+str(M1)+'kg.' , file=log)
    log.close()
    
if d==130:
    N10,M10=findM(3e12,10,25.13,r=10e-6)
    Na,Ma=findM(2e12,10,25.13,r=1e-3)
    log=open('mass_'+str(int(d))+'AU.txt','w')
    print('For a 10 micron at '+str(d)+'AU, there are '+str(N10)+' grains with a mass of '+str(M10)+'kg.' , file=log)
    print(' ', file=log)
    print('For a 1 mm perfect absorber at '+str(d)+'AU, there are '+str(Na)+' grains with a mass of '+str(Ma)+'kg.' , file=log)
    log.close()
    
#__________________________________

print('Part 4 Finished')

#___________Part 5__________________
#__________________________________
#Radiation pressure and gravitational force for each particle type at the given
#distance is calculated in order to find infall time.

T01=findF(en01,d,0.1e-6)
T1=findF(en1,d,1e-6)
T10=findF(en10,d,10e-6)
Ta=findF(ena,d,1e-3)

#-----Make Output File -----------

log=open('infall_'+str(int(d))+'AU.txt','w')

print('A 0.1 micron grain at '+str(d)+' AU spirals in in '+str(T01)+' years', file=log)
print(' ', file=log)
print('A 1 micron grain at '+str(d)+' AU spirals in in '+str(T1)+' years', file=log)
print(' ', file=log)
print('A 10 micron grain at '+str(d)+' AU spirals in in '+str(T10)+' years', file=log)
print(' ', file=log)
print('A 1 mm perfect absober at '+str(d)+' AU spirals in in '+str(Ta)+' years', file=log)

log.close()

#------------------------------

print('Part 5 Finished')

print("--- %s seconds ---" % (time.time() - start_time))
