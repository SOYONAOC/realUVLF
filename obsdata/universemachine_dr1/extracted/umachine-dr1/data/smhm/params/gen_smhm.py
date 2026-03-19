#!/usr/bin/python
import sys
import math
import re

if (len(sys.argv) < 3):
    print("Usage: %s z smhm_parameter_file.txt" % sys.argv[0])
    quit()

z = sys.argv[1]

if (re.search('^uncertainties', sys.argv[2])):
    print("Use gen_smhm_uncertainties.py instead for uncertainties_* files.");
    quit()

#Load params
param_file = open(sys.argv[2], "r")
param_list = []
allparams = []
for line in param_file:
    param_list.append(float((line.split(" "))[1]))
    allparams.append(line.split(" "))

if (len(param_list) != 20):
    print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
    quit()

names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
params = dict(zip(names, param_list))

#Decide whether to print tex or evaluate SMHM parameter
try:
    z = float(z)
except:
    #print TeX
    for x in allparams[0:10:1]:
        x[3] = -float(x[3])
        sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
    sys.stdout.write("\\\\\n & & & ")
    for x in allparams[10:19:1]:
        x[3] = -float(x[3])
        sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
#    sys.stdout.write("\\\\\n & & & ")    
#    for x in allparams[16:19:1]:
#        x[3] = -float(x[3])
#        sys.stdout.write('& $%.3f^{%+.3f}_{%+.3f}$' % tuple(float(y) for y in x[1:4]))
    sys.stdout.write(' & %.0f' % float(allparams[19][1]))
    if (float(allparams[19][1])>200):
        sys.stdout.write('$\dag$')
    print('\\\\[2ex]')
    quit()

#Print SMHM relation
a = 1.0/(1.0+z)
a1 = a - 1.0
lna = math.log(a)
zparams = {}
zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
zparams['delta'] = params['DELTA']
zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])

smhm_max = 14.5-0.35*z
print('#Log10(Mpeak/Msun) Log10(Median_SM/Msun) Log10(Median_SM/Mpeak)')
print('#Mpeak: peak historical halo mass, using Bryan & Norman virial overdensity.')
print('#Overall fit chi^2: %f' % params['CHI2'])
if (params['CHI2']>200):
    print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
for m in [x*0.05 for x in range(int(10.5*20),int(smhm_max*20+1),1)]:
    dm = m-zparams['m_1'];
    dm2 = dm/zparams['delta'];
    sm = zparams['sm_0'] - math.log10(10**(-zparams['alpha']*dm) + 10**(-zparams['beta']*dm)) + zparams['gamma']*math.exp(-0.5*(dm2*dm2));
    print("%.2f %.6f %.6f" % (m,sm,sm-m))


    
