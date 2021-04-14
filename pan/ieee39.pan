ground electrical gnd

parameter PRAND=1M
parameter TSTOP=600
parameter FRAND=10
parameter D=0
parameter DZA=0
parameter LAMBDA=0
parameter COEFF=0

options outintnodes=yes ; pivcaching=0

Al_dummy_tstop  alter param="TSTOP"  rt=yes
Al_dummy_frand  alter param="FRAND"  rt=yes
Al_dummy_d      alter param="D"      rt=yes
Al_dummy_dza    alter param="DZA"    rt=yes
Al_dummy_lambda alter param="LAMBDA" rt=yes
Al_dummy_coeff  alter param="COEFF"  rt=yes

#ifdef PAN

Dc dc nettype=1 print=yes sparse=1
Pz pz nettype=yes

Rnd control begin

    dt = 1/FRAND;
    T  = [dt:dt:TSTOP+dt];

    noise_samples_bus_8  = [ T, randn( length(T) ) ];
    noise_samples_bus_21 = [ T, randn( length(T) ) ];
    noise_samples_bus_26 = [ T, randn( length(T) ) ];

endcontrol

Tr tran stop=TSTOP nettype=1 method=2 maxord=2 iabstol=1u devvars=1 tmax=0.1 \
        noisefmax=FRAND/2 noiseinj=2 seed=12345 

#endif


begin power

; +---------------------------------------+
; | This file was automatically generated |
; +---------------------------------------+

; +------------------+
; | Power generators |
; +------------------+

E30  bus30  avr30 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
	       ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

T30   pm30  omega30 powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg30" dza=DZA

Pg30 bus30 avr30 pm30 omega30 powergenerator pg=(1+LAMBDA)*2.5 \
    vg=1.047500e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \ 
    type=4 ra=0 xdp=0.0060 xqp=0.0080 xd=0.0200 xq=0.019 \
    td0p=7.00 tq0p=0.70 xl=0.0030 h=500 d=D qlimits=no

E31  bus31  avr31 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
               ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

T31   pm31  omega31 powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg31" dza=DZA

Pg31 bus31 avr31 pm31 omega31 powergenerator pg=(1+LAMBDA)*5.729300e+00 \
    vg=1.040000e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0697 xqp=0.1700 xd=0.2950 xq=0.282 \
    td0p=6.56 tq0p=1.50 xl=0.0350 h=30.3 d=D slack=yes qlimits=no

E32  bus32  avr32 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
               ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

T32   pm32  omega32 powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg32" dza=DZA

Pg32 bus32 avr32 pm32 omega32 powergenerator pg=(1+LAMBDA)*6.500000e+00 \
    vg=9.831000e-01 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0531 xqp=0.0876 xd=0.2495 xq=0.237 \
    td0p=5.70 tq0p=1.50 xl=0.0304 h=35.8 d=D qlimits=no

Pg33 bus33 powergenerator pg=(1+LAMBDA)*6.320000e+00 vg=9.972000e-01 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0436 xqp=0.1660 xd=0.2620 xq=0.258 \
    td0p=5.69 tq0p=1.50 xl=0.0295 h=28.6 d=D qlimits=no

Pg34 bus34 powergenerator pg=(1+LAMBDA)*5.080000e+00 vg=1.012300e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=4.000000e+00 \
    qmin=-3.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.1320 xqp=0.1660 xd=0.6700 xq=0.620 \
    td0p=5.40 tq0p=0.44 xl=0.0540 h=26.0 d=D qlimits=no

E35  bus35  avr35 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
	       ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

T35   pm35  omega35 powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg35" dza=DZA

Pg35 bus35 avr35 pm35 omega35 powergenerator pg=(1+LAMBDA)*6.500000e+00 \
    vg=1.049300e+00 prating=1.000000e+08 vrating=1.000000e+03 \
    qmax=8.000000e+00 qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0500 xqp=0.0814 xd=0.2540 xq=0.241 \
    td0p=7.30 tq0p=0.40 xl=0.0224 h=34.8 d=D qlimits=no

Pg36 bus36 powergenerator pg=(1+LAMBDA)*5.600000e+00 vg=1.063500e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0490 xqp=0.1860 xd=0.2950 xq=0.292 \
    td0p=5.66 tq0p=1.50 xl=0.0322 h=26.4 d=D qlimits=no

Pg37 bus37 powergenerator pg=(1+LAMBDA)*5.400000e+00 vg=1.027800e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=8.000000e+00 \
    qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0570 xqp=0.0911 xd=0.2900 xq=0.280 \
    td0p=6.70 tq0p=0.41 xl=0.0280 h=24.3 d=D

E38  bus38  avr38 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
	       ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

T38   pm38  omega38 powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg38" dza=DZA

Pg38 bus38 avr38 pm38 omega38 powergenerator pg=(1+LAMBDA)*8.300000e+00 \
    vg=1.026500e+00 prating=1.000000e+08 vrating=1.000000e+03 \
    qmax=8.000000e+00 qmin=-5.000000e+00 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0570 xqp=0.0587 xd=0.2106 xq=0.205 \
    td0p=4.79 tq0p=1.69 xl=0.0298 h=34.5 d=D

E39  bus39  avr39 poweravr vrating=1k type=2 vmax=4.38 vmin=0 ka=20 \
               ta=0.02 kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

parameter MEX=3.4216*100
parameter MHP=9.2897*100
parameter MIP=1.55589*1000
parameter MLP=8.58670*1000
parameter KHP=7.277
parameter KIP=13.168
parameter KLP=19.618
parameter KEX=1.064
parameter DHP=1.0/10
parameter DIP=1.0/10
parameter DLP=1.0/10
parameter DEX=1.0/10
parameter D12=0.0
parameter D23=0.0
parameter D34=0.0
parameter D45=0.0

T39   tg39  omega39 powertg type=1 omegaref=1 r=0.02 pmax=12 pmin=0 ts=0.1 \
                            tc=0.45 t3=0 t4=12 t5=50 gen="Pg39"  dza=DZA

Sh39  tg39  pm39  omega39  x39  y39  powershaft \
	    mex=MEX mhp=MHP mip=MIP mlp=MLP khp=KHP kip=KIP klp=KLP kex=KEX \
	    dip=DIP dlp=DLP dhp=DHP d12=D12 d23=D23 d34=D34 d45=D45 dex=DEX \
	    gen="Pg39"

Pg39 bus39 avr39 pm39 omega39 x39 y39 powergenerator pg=(1+LAMBDA)*1.005729e+01 vg=1.030000e+00 \
    prating=1.000000e+08 vrating=1.000000e+03 qmax=1.500000e+01 \
    qmin=-1.000000e+01 pmax=9.999900e+01 pmin=0.000000e+00 \
    type=4 ra=0 xdp=0.0310 xqp=0.0080 \
    xd=0.0100 xq=0.069 td0p=10.2 tq0p=0.00 xl=0.0125 h=42.0 d=D qlimits=no

; +-------------+
; | Power loads |
; +-------------+
Lo3  bus3  powerload pc=(1+LAMBDA)*3.220000e+00 qc=(1+LAMBDA)*2.400000e-02 vrating=1.000000e+03 prating=1.000000e+08
Lo4  bus4  powerload pc=(1+LAMBDA)*5.000000e+00 qc=(1+LAMBDA)*1.840000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo7  bus7  powerload pc=(1+LAMBDA)*2.338000e+00 qc=(1+LAMBDA)*8.400000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo8  bus8  powerload pc=(1+LAMBDA)*5.220000e+00 qc=(1+LAMBDA)*1.760000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo12 bus12 powerload pc=(1+LAMBDA)*8.500000e-02 qc=(1+LAMBDA)*8.800000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo15 bus15 powerload pc=(1+LAMBDA)*3.200000e+00 qc=(1+LAMBDA)*1.530000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo16 bus16 powerload pc=(1+LAMBDA)*3.294000e+00 qc=(1+LAMBDA)*3.230000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo18 bus18 powerload pc=(1+LAMBDA)*1.580000e+00 qc=(1+LAMBDA)*3.000000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo20 bus20 powerload pc=(1+LAMBDA)*6.800000e+00 qc=(1+LAMBDA)*1.030000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo21 bus21 powerload pc=(1+LAMBDA)*2.740000e+00 qc=(1+LAMBDA)*1.150000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo23 bus23 powerload pc=(1+LAMBDA)*2.475000e+00 qc=(1+LAMBDA)*8.460000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo24 bus24 powerload pc=(1+LAMBDA)*3.086000e+00 qc=(1+LAMBDA)*-9.220000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo25 bus25 powerload pc=(1+LAMBDA)*2.240000e+00 qc=(1+LAMBDA)*4.720000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo26 bus26 powerload pc=(1+LAMBDA)*1.390000e+00 qc=(1+LAMBDA)*1.700000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo27 bus27 powerload pc=(1+LAMBDA)*2.810000e+00 qc=(1+LAMBDA)*7.550000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo28 bus28 powerload pc=(1+LAMBDA)*2.060000e+00 qc=(1+LAMBDA)*2.760000e-01 vrating=1.000000e+03 prating=1.000000e+08
Lo29 bus29 powerload pc=(1+LAMBDA)*2.835000e+00 qc=(1+LAMBDA)*1.269000e+00 vrating=1.000000e+03 prating=1.000000e+08
Lo31 bus31 powerload pc=(1+LAMBDA)*9.200000e-02 qc=(1+LAMBDA)*4.600000e-02 vrating=1.000000e+03 prating=1.000000e+08
Lo39 bus39 powerload pc=(1+LAMBDA)*1.104000e+01 qc=(1+LAMBDA)*2.500000e+00 vrating=1.000000e+03 prating=1.000000e+08

; +-------------+
; | Power lines |
; +-------------+
Line1  bus1   bus39b powerline r=1.000000e-03 x=2.500000e-02 b=7.500000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line2  bus2   bus3   powerline r=1.300000e-03 x=1.510000e-02 b=2.572000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line3  bus2   bus25  powerline r=7.000000e-03 x=8.600000e-03 b=1.460000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line4  bus3b  bus4   powerline r=1.300000e-03 x=2.130000e-02 b=2.214000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line5  bus3   bus18  powerline r=1.100000e-03 x=1.330000e-02 b=2.138000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line6  bus4   bus5   powerline r=8.000000e-04 x=1.280000e-02 b=1.342000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line7  bus4   bus14  powerline r=8.000000e-04 x=1.290000e-02 b=1.382000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line8  bus5   bus6   powerline r=2.000000e-04 x=2.600000e-03 b=4.340000e-02 prating=1.000000e+08 vrating=1.000000e+03
Line9  bus5   bus8   powerline r=8.000000e-04 x=1.120000e-02 b=1.476000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line10 bus6   bus7   powerline r=6.000000e-04 x=9.200000e-03 b=1.130000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line11 bus6   bus11  powerline r=7.000000e-04 x=8.200000e-03 b=1.389000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line12 bus7   bus8   powerline r=4.000000e-04 x=4.600000e-03 b=7.800000e-02 prating=1.000000e+08 vrating=1.000000e+03
Line13 bus8   bus9   powerline r=2.300000e-03 x=3.630000e-02 b=3.804000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line14 bus9   bus39  powerline r=1.000000e-03 x=2.500000e-02 b=1.200000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line15 bus10  bus11  powerline r=4.000000e-04 x=4.300000e-03 b=7.290000e-02 prating=1.000000e+08 vrating=1.000000e+03
Line16 bus10  bus13  powerline r=4.000000e-04 x=4.300000e-03 b=7.290000e-02 prating=1.000000e+08 vrating=1.000000e+03
Line17 bus13  bus14  powerline r=9.000000e-04 x=1.010000e-02 b=1.723000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line18 bus14b bus15  powerline r=1.800000e-03 x=2.170000e-02 b=3.660000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line19 bus15  bus16  powerline r=9.000000e-04 x=9.400000e-03 b=1.710000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line20 bus16  bus17b powerline r=7.000000e-04 x=8.900000e-03 b=1.342000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line21 bus16  bus19  powerline r=1.600000e-03 x=1.950000e-02 b=3.040000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line22 bus16  bus21  powerline r=8.000000e-04 x=1.350000e-02 b=2.548000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line23 bus16  bus24  powerline r=3.000000e-04 x=5.900000e-03 b=6.800000e-02 prating=1.000000e+08 vrating=1.000000e+03
Line24 bus17  bus18  powerline r=7.000000e-04 x=8.200000e-03 b=1.319000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line25 bus17  bus27  powerline r=1.300000e-03 x=1.730000e-02 b=3.216000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line26 bus21  bus22  powerline r=8.000000e-04 x=1.400000e-02 b=2.565000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line27 bus22  bus23  powerline r=6.000000e-04 x=9.600000e-03 b=1.846000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line28 bus23  bus24  powerline r=2.200000e-03 x=3.500000e-02 b=3.610000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line29 bus25  bus26  powerline r=3.200000e-03 x=3.230000e-02 b=5.130000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line30 bus26  bus27  powerline r=1.400000e-03 x=1.470000e-02 b=2.396000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line31 bus26  bus28  powerline r=4.300000e-03 x=4.740000e-02 b=7.802000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line32 bus26  bus29  powerline r=5.700000e-03 x=6.250000e-02 b=1.029000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line33 bus28  bus29  powerline r=1.400000e-03 x=1.510000e-02 b=2.490000e-01 prating=1.000000e+08 vrating=1.000000e+03
Line34 bus2   bus30  powerline r=0.000000e+00 x=1.810000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line35 bus31  bus6   powerline r=0.000000e+00 x=2.500000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line36 bus10  bus32  powerline r=0.000000e+00 x=2.000000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line37 bus12  bus11  powerline r=1.600000e-03 x=4.350000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line38 bus12  bus13  powerline r=1.600000e-03 x=4.350000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line39 bus19  bus20  powerline r=7.000000e-04 x=1.380000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line40 bus19  bus33  powerline r=7.000000e-04 x=1.420000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line41 bus20  bus34  powerline r=9.000000e-04 x=1.800000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line42 bus22  bus35  powerline r=0.000000e+00 x=1.430000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line43 bus23  bus36  powerline r=5.000000e-04 x=2.720000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line44 bus25  bus37  powerline r=6.000000e-04 x=2.320000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03
Line45 bus29  bus38  powerline r=8.000000e-04 x=1.560000e-02 b=0.000000e+00 prating=1.000000e+08 vrating=1.000000e+03

; +--------------------+
; | Power transformers |
; +--------------------+

; +--------+
; | Busses |
; +--------+

Bus1  bus1  powerbus vb=1k v0=+1.045500e+00 theta0=-9.431200e+00
Bus2  bus2  powerbus vb=1k v0=+1.043600e+00 theta0=-6.864400e+00
Bus3  bus3  powerbus vb=1k v0=+1.020380e+00 theta0=-9.743300e+00
Bus3b bus3b powerbus vb=1k v0=+1.020380e+00 theta0=-9.743300e+00
Bus4  bus4  powerbus vb=1k v0=+9.978000e-01 theta0=-1.056710e+01
Bus5  bus5  powerbus vb=1k v0=+9.935500e-01 theta0=-9.340900e+00
Bus6  bus6  powerbus vb=1k v0=+9.965100e-01 theta0=-8.616900e+00
Bus7  bus7  powerbus vb=1k v0=+9.408300e-01 theta0=-1.076130e+01
Bus8  bus8  powerbus vb=1k v0=+9.546000e-01 theta0=-1.137220e+01
Bus9  bus9  powerbus vb=1k v0=+1.011140e+00 theta0=-1.117840e+01
Bus10 bus10 powerbus vb=1k v0=+1.008610e+00 theta0=-6.203700e+00
Bus11 bus11 powerbus vb=1k v0=+1.003260e+00 theta0=-7.025400e+00
Bus12 bus12 powerbus vb=1k v0=+9.905600e-01 theta0=-7.043400e+00
Bus13 bus13 powerbus vb=1k v0=+1.005070e+00 theta0=-6.928800e+00
Bus14 bus14 powerbus vb=1k v0=+1.001020e+00 theta0=-8.628100e+00
Bus15 bus15 powerbus vb=1k v0=+9.939300e-01 theta0=-9.014500e+00
Bus16 bus16 powerbus vb=1k v0=+1.006030e+00 theta0=-7.520500e+00
Bus17 bus17 powerbus vb=1k v0=+1.013460e+00 theta0=-8.588000e+00
Bus18 bus18 powerbus vb=1k v0=+1.014740e+00 theta0=-9.473300e+00
Bus19 bus19 powerbus vb=1k v0=+1.040360e+00 theta0=-2.810900e+00
Bus20 bus20 powerbus vb=1k v0=+9.856900e-01 theta0=-4.255200e+00
Bus21 bus21 powerbus vb=1k v0=+1.013630e+00 theta0=-5.035500e+00
Bus22 bus22 powerbus vb=1k v0=+1.040090e+00 theta0=-4.920000e-01
Bus23 bus23 powerbus vb=1k v0=+1.034740e+00 theta0=-6.935000e-01
Bus24 bus24 powerbus vb=1k v0=+1.013840e+00 theta0=-7.401400e+00
Bus25 bus25 powerbus vb=1k v0=+1.051830e+00 theta0=-5.469800e+00
Bus26 bus26 powerbus vb=1k v0=+1.037590e+00 theta0=-6.698300e+00
Bus27 bus27 powerbus vb=1k v0=+1.020480e+00 theta0=-8.754200e+00
Bus28 bus28 powerbus vb=1k v0=+1.034350e+00 theta0=-3.080400e+00
Bus29 bus29 powerbus vb=1k v0=+1.034070e+00 theta0=-2.364000e-01
Bus30 bus30 powerbus vb=1k v0=+1.047500e+00 theta0=-4.432700e+00
Bus31 bus31 powerbus vb=1k v0=+1.040000e+00 theta0=-1.586800e+00
Bus32 bus32 powerbus vb=1k v0=+9.831000e-01 theta0=+1.861000e+00
Bus33 bus33 powerbus vb=1k v0=+9.972000e-01 theta0=+2.428500e+00
Bus34 bus34 powerbus vb=1k v0=+1.012300e+00 theta0=+9.475000e-01
Bus35 bus35 powerbus vb=1k v0=+1.049300e+00 theta0=+4.516200e+00
Bus36 bus36 powerbus vb=1k v0=+1.063500e+00 theta0=+7.225800e+00
Bus37 bus37 powerbus vb=1k v0=+1.027800e+00 theta0=+1.343600e+00
Bus38 bus38 powerbus vb=1k v0=+1.026500e+00 theta0=+6.890500e+00
Bus39 bus39 powerbus vb=1k v0=+1.030000e+00 theta0=-1.096000e+01

Pec1  bus3  pe3  gnd qe3  gnd bus3b  powerec type=4
Pec2  bus14 pe14 gnd qe14 gnd bus14b powerec type=4
Pec3  bus17 pe17 gnd qe17 gnd bus17b powerec type=4
Pec4  bus39 pe39 gnd qe39 gnd bus39b powerec type=4

Pec5  bus3  omegael03 gnd powerec type=2
Pec6  bus14 omegael14 gnd powerec type=2
Pec7  bus17 omegael17 gnd powerec type=2
Pec8  bus39 omegael39 gnd powerec type=2

Pec9  bus8  d8  gnd q8  gnd powerec type=0
Pec10 bus21 d21 gnd q21 gnd powerec type=0
Pec11 bus26 d26 gnd q26 gnd powerec type=0

end

Rnd8 d8 q8 rand8  RAND_L P=PRAND VRATING=1k VMAX=1.2*345k VMIN=0.8*345k
Wav8 rand8 gnd port noisesamples="noise_samples_bus_8"
Rnd21 d21 q21 rand21  RAND_L P=PRAND VRATING=1k VMAX=1.2*345k VMIN=0.8*345k
Wav21 rand21 gnd port noisesamples="noise_samples_bus_21"
Rnd26 d26 q26 rand26  RAND_L P=PRAND VRATING=1k VMAX=1.2*345k VMIN=0.8*345k
Wav26 rand26 gnd port noisesamples="noise_samples_bus_26"

model RAND_L nport veriloga="randl.va" verilogaprotected=1

