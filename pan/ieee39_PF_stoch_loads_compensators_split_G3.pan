ground electrical gnd

parameter PRAND=1M
parameter TSTOP=300
parameter SRATE=400
parameter D=0

parameter Hcomp11 = 0.1 ; inertia of compensator 1 in area 1
parameter Hcomp21 = 0.1 ; inertia of compensator 1 in area 2
parameter Hcomp31 = 0.1 ; inertia of compensator 1 in area 3

options outintnodes=yes ; pivcaching=0

Al_dummy_tstop   alter param="TSTOP"  rt=yes
Al_dummy_srate   alter param="SRATE"  rt=yes
Al_dummy_damping alter param="D"      rt=yes

#ifdef PAN

Ctrl control begin
    load("mat5", "OU.mat");
endcontrol

Al1 alter param="h" value=4.33 instance="G02" invalidate=false
Dc1 dc nettype=1 print=yes sparse=1
Pz1 pz nettype=1 mem=["invmtrx"] pf=1
TrA tran stop=10/SRATE   nettype=1 restart=1 annotate=3 method=1 timepoints=1/SRATE saman=yes sparse=2
Tr1 tran stop=60 nettype=1 restart=0 annotate=3 method=1 timepoints=1/SRATE forcetps=1 \
         maxiter=65 saman=yes sparse=2 devvars=yes

#endif // PAN

begin power

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

AVR02 bus31 avr31 poweravr type=2 vrating=1.650000e+04 \
		ka=6.2 ta=0.05 kf=0.057 tf=0.5 ke=-0.633 \
		te=0.405 tr=0.0 vmin=-1.0 vmax=1.0 e1=3.036437 \
		e2=4.048583 se1=0.66 se2=0.88 

GOV02 php31 omega31 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G02 bus31 avr31 php31 omega31 powergenerator type=6 qlimits=no phtype=1 \
		slack=yes vg=0.982 prating=7.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.214286 \
		xdp=0.4879 xd=2.065 xq=1.974 xds=0.35 xqs=0.35 ra=0 \
		td0p=6.56 td0s=0.05 tq0s=0.035 xl=0.245 h=4.33 d=0 xqp=1.19 tq0p=1.5

AVR03 bus32 avr32 poweravr type=2 vrating=1.650000e+04 \
		ka=5.0 ta=0.06 kf=0.08 tf=1.0 ke=-0.0198 \
		te=0.5 tr=0.0 vmin=-1.0 vmax=1.0 e1=2.342286 \
		e2=3.123048 se1=0.13 se2=0.34 

GOV03 php32 omega32 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G03 bus32 avr32 php32 omega32 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.8125 vg=0.9831 prating=8.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.25 \
		xdp=0.4248 xd=1.996 xq=1.896 xds=0.36 xqs=0.36 ra=0 \
		td0p=5.7 td0s=0.05 tq0s=0.035 xl=0.2432 h=4.47 d=0 xqp=0.7008 tq0p=1.5

G03b bus32b gnd gnd omega32b powergenerator type=2 qlimits=no phtype=1 \
		pg=0.8125 vg=0.9831 prating=8.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.25 \
		xdp=0.4248 xd=1.996 xq=1.896 xds=0.36 xqs=0.36 ra=0 \
		td0p=5.7 td0s=0.05 tq0s=0.035 xl=0.2432 h=4.47 d=0 xqp=0.7008 tq0p=1.5

AVR04 bus33 avr33 poweravr type=2 vrating=1.650000e+04 \
		ka=5.0 ta=0.06 kf=0.08 tf=1.0 ke=-0.0525 \
		te=0.5 tr=0.0 vmin=-1.0 vmax=1.0 e1=2.868069 \
		e2=3.824092 se1=0.08 se2=0.314 

GOV04 php33 omega33 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G04 bus33 avr33 php33 omega33 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.79 vg=0.9972 prating=8.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.25 \
		xdp=0.3488 xd=2.096 xq=2.064 xds=0.28 xqs=0.28 ra=0 \
		td0p=5.69 td0s=0.05 tq0s=0.035 xl=0.236 h=3.57 d=0 xqp=1.328 tq0p=1.5

AVR05 bus34 avr34 poweravr type=2 vrating=1.650000e+04 \
		ka=40.0 ta=0.02 kf=0.03 tf=1.0 ke=1.0 \
		te=0.785 tr=0.0 vmin=-10.0 vmax=10.0 e1=3.926702 \
		e2=5.235602 se1=0.07 se2=0.91 

GOV05 php34 omega34 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G05 bus34 avr34 php34 omega34 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.846667 vg=1.0123 prating=6.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.233333 \
		xdp=0.396 xd=2.01 xq=1.86 xds=0.267 xqs=0.267 ra=0 \
		td0p=5.4 td0s=0.05 tq0s=0.035 xl=0.162 h=4.33 d=0 xqp=0.498 tq0p=0.44

AVR06 bus35 avr35 poweravr type=2 vrating=1.650000e+04 \
		ka=5.0 ta=0.02 kf=0.0754 tf=1.246 ke=-0.0419 \
		te=0.471 tr=0.0 vmin=-1.0 vmax=1.0 e1=3.586801 \
		e2=4.782401 se1=0.064 se2=0.251 

GOV06 php35 omega35 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G06 bus35 avr35 php35 omega35 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.8125 vg=1.0493 prating=8.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.25 \
		xdp=0.4 xd=2.032 xq=1.928 xds=0.32 xqs=0.32 ra=0 \
		td0p=7.3 td0s=0.05 tq0s=0.035 xl=0.1792 h=4.35 d=0 xqp=0.6512 tq0p=0.4

AVR07 bus36 avr36 poweravr type=2 vrating=1.650000e+04 \
		ka=40.0 ta=0.02 kf=0.03 tf=1.0 ke=1.0 \
		te=0.73 tr=0.0 vmin=-6.5 vmax=6.5 e1=2.801724 \
		e2=3.735632 se1=0.53 se2=0.74 

GOV07 php36 omega36 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G07 bus36 avr36 php36 omega36 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.8 vg=1.0635 prating=7.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.214286 \
		xdp=0.343 xd=2.065 xq=2.044 xds=0.308 xqs=0.308 ra=0 \
		td0p=5.66 td0s=0.05 tq0s=0.035 xl=0.2254 h=3.77 d=0 xqp=1.302 tq0p=1.5

AVR08 bus37 avr37 poweravr type=2 vrating=1.650000e+04 \
		ka=5.0 ta=0.02 kf=0.0854 tf=1.26 ke=-0.047 \
		te=0.528 tr=0.0 vmin=-1.0 vmax=1.0 e1=3.191489 \
		e2=4.255319 se1=0.072 se2=0.282 

GOV08 php37 omega37 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G08 bus37 avr37 php37 omega37 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.771429 vg=1.0278 prating=7.000000e+08 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.214286 \
		xdp=0.399 xd=2.03 xq=1.96 xds=0.315 xqs=0.315 ra=0 \
		td0p=6.7 td0s=0.05 tq0s=0.035 xl=0.196 h=3.47 d=0 xqp=0.6377 tq0p=0.41

AVR09 bus38 avr38 poweravr type=2 vrating=1.650000e+04 \
		ka=40.0 ta=0.02 kf=0.03 tf=1.0 ke=1.0 \
		te=1.4 tr=0.0 vmin=-10.5 vmax=10.5 e1=4.256757 \
		e2=5.675676 se1=0.62 se2=0.85 

GOV09 php38 omega38 powertg type=3 \
		r=0.2 t1=0.2 t2=1.0 t3=0.6 k1=0.3 \
		k2=0.0 t5=0.5 k3=0.25 k4=0.0 t6=0.8 \
		k5=0.3 k6=0.0 t4=0.6 t7=1.0 k7=0.15 \
		k8=0.0 uc=-0.3 uo=0.3 pmin=0.0 pmax=1.0 

G09 bus38 avr38 php38 omega38 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.83 vg=1.0265 prating=1.000000e+09 vrating=1.650000e+04 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0.25 \
		xdp=0.57 xd=2.106 xq=2.05 xds=0.45 xqs=0.45 ra=0 \
		td0p=4.79 td0s=0.05 tq0s=0.035 xl=0.298 h=3.45 d=0 xqp=0.587 tq0p=1.96

AVR10 bus30 avr30 poweravr type=2 vrating=1.650000e+04 \
		ka=5.0 ta=0.06 kf=0.04 tf=1.0 ke=-0.0485 \
		te=0.25 tr=0.0 vmin=-1.0 vmax=1.0 e1=3.546099 \
		e2=4.728132 se1=0.08 se2=0.26 

GOV10 pm30 omega30 powertg type=4 \
		tg=0.05 tp=0.04 sigma=0.04 delta=0.2 tr=10.0 \
		a11=0.5 a13=1.0 a21=1.5 a23=1.0 tw=0.75 \
		uc=-0.1 uo=0.1 pmin=0.0 pmax=1.0 

G10 bus30 avr30 pm30 omega30 powergenerator type=52 qlimits=no phtype=1 \
		pg=0.25 vg=1.0475 prating=1.000000e+09 vrating=1.650000e+04 \
		qmax=0.6 qmin=-0.5 pmax=0.85 pmin=0 \
		xdp=0.31 xd=1 xq=0.69 xds=0.25 xqs=0.25 ra=0 \
		td0p=10.2 td0s=0.05 tq0s=0.035 xl=0.125 h=4.2 d=0

G01 bus39 gnd gnd omega39 powergenerator type=6 qlimits=no phtype=1 \
		pg=0.1 vg=1.03 prating=1.000000e+10 vrating=3.450000e+05 \
		qmax=0.7 qmin=-0.3 pmax=0.85 pmin=0 \
		xdp=0.6 xd=2 xq=1.9 xds=0.4 xqs=0.4 ra=0 \
		td0p=7 td0s=0.05 tq0s=0.035 xl=0.3 h=5 d=0 xqp=0.8 tq0p=0.7


;;;;;;;;;; COMPENSATORS ;;;;;;;;;;

; in area 1
Comp11 bus8  gnd gnd omegacomp11 powergenerator pg=0 vg=1.030000e+00 \
                prating=1.000000e+08 vrating=345k qmax=1.500000e+01 \
                qmin=-1.000000e+01 pmax=9.999900e+01 pmin=0.000000e+00 \
                type=2 ra=1m xdp=0.00310 h=Hcomp11 d=D qlimits=no phtype=1

; in area 2
Comp21 bus24 gnd gnd omegacomp21 powergenerator pg=0 vg=1.030000e+00 \
                prating=1.000000e+08 vrating=345k qmax=1.500000e+01 \
                qmin=-1.000000e+01 pmax=9.999900e+01 pmin=0.000000e+00 \
                type=2 ra=1m xdp=0.00310 h=Hcomp21 d=D qlimits=no phtype=1

; in area 3
Comp31 bus27 gnd gnd omegacomp31 powergenerator pg=0 vg=1.030000e+00 \
                prating=1.000000e+08 vrating=345k qmax=1.500000e+01 \
                qmin=-1.000000e+01 pmax=9.999900e+01 pmin=0.000000e+00 \
                type=2 ra=1m xdp=0.00310 h=Hcomp31 d=D qlimits=no phtype=1



Load03 bus3  powerload utype=1 pc=3.220000e+08 qc=2.400000e+06 vrating=3.450000e+05
Load04 bus4  powerload utype=1 pc=5.000000e+08 qc=1.840000e+08 vrating=3.450000e+05
Load07 bus7  powerload utype=1 pc=2.338000e+08 qc=8.400000e+07 vrating=3.450000e+05
Load08 bus8  powerload utype=1 pc=5.220000e+08 qc=1.760000e+08 vrating=3.450000e+05
Load12 bus12 powerload utype=1 pc=7.500000e+06 qc=8.800000e+07 vrating=1.380000e+05
Load15 bus15 powerload utype=1 pc=3.200000e+08 qc=1.530000e+08 vrating=3.450000e+05
Load16 bus16 powerload utype=1 pc=3.290000e+08 qc=3.230000e+07 vrating=3.450000e+05
Load18 bus18 powerload utype=1 pc=1.580000e+08 qc=3.000000e+07 vrating=3.450000e+05
Load20 bus20 powerload utype=1 pc=6.280000e+08 qc=1.030000e+08 vrating=2.300000e+05
Load21 bus21 powerload utype=1 pc=2.740000e+08 qc=1.150000e+08 vrating=3.450000e+05
Load23 bus23 powerload utype=1 pc=2.475000e+08 qc=8.460000e+07 vrating=3.450000e+05
Load24 bus24 powerload utype=1 pc=3.086000e+08 qc=-9.220000e+07 vrating=3.450000e+05
Load25 bus25 powerload utype=1 pc=2.240000e+08 qc=4.720000e+07 vrating=3.450000e+05
Load26 bus26 powerload utype=1 pc=1.390000e+08 qc=1.700000e+07 vrating=3.450000e+05
Load27 bus27 powerload utype=1 pc=2.810000e+08 qc=7.550000e+07 vrating=3.450000e+05
Load28 bus28 powerload utype=1 pc=2.060000e+08 qc=2.760000e+07 vrating=3.450000e+05
Load29 bus29 powerload utype=1 pc=2.835000e+08 qc=2.690000e+07 vrating=3.450000e+05
Load31 bus31 powerload utype=1 pc=9.200000e+06 qc=4.600000e+06 vrating=1.650000e+04
Load39 bus39 powerload utype=1 pc=1.104000e+09 qc=2.500000e+08 vrating=3.450000e+05

Line0102 bus1   bus2  powerline utype=1 r=4.165876e+00 x=4.891928e+01 b=5.870192e-04 vrating=3.450000e+05
Line0139 bus1c  bus39 powerline utype=1 r=1.190250e+00 x=2.975625e+01 b=6.301211e-04 vrating=3.450000e+05
Line0203 bus2   bus3  powerline utype=1 r=1.547325e+00 x=1.797278e+01 b=2.160890e-04 vrating=3.450000e+05
Line0225 bus2   bus25 powerline utype=1 r=8.331750e+00 x=1.023615e+01 b=1.226628e-04 vrating=3.450000e+05
Line0304 bus3c  bus4  powerline utype=1 r=1.547325e+00 x=2.535233e+01 b=1.860100e-04 vrating=3.450000e+05
Line0318 bus3   bus18 powerline utype=1 r=1.309275e+00 x=1.583033e+01 b=1.796256e-04 vrating=3.450000e+05
Line0405 bus4   bus5  powerline utype=1 r=9.522000e-01 x=1.523520e+01 b=1.127494e-04 vrating=3.450000e+05
Line0414 bus4   bus14 powerline utype=1 r=9.521999e-01 x=1.535423e+01 b=1.161096e-04 vrating=3.450000e+05
Line0506 bus5   bus6  powerline utype=1 r=2.380500e-01 x=3.094650e+00 b=3.646304e-05 vrating=3.450000e+05
Line0508 bus5   bus8  powerline utype=1 r=9.522000e-01 x=1.333080e+01 b=1.240082e-04 vrating=3.450000e+05
Line0607 bus6   bus7  powerline utype=1 r=7.141501e-01 x=1.095030e+01 b=9.493818e-05 vrating=3.450000e+05
Line0611 bus6   bus11 powerline utype=1 r=8.331751e-01 x=9.760051e+00 b=1.166987e-04 vrating=3.450000e+05
Line0708 bus7   bus8  powerline utype=1 r=4.761001e-01 x=5.475150e+00 b=6.553257e-05 vrating=3.450000e+05
Line0809 bus8   bus9  powerline utype=1 r=2.737574e+00 x=4.320608e+01 b=3.195983e-04 vrating=3.450000e+05
Line0939 bus9   bus39 powerline utype=1 r=1.190250e+00 x=2.975625e+01 b=1.008192e-03 vrating=3.450000e+05
Line1011 bus10  bus11 powerline utype=1 r=4.761000e-01 x=5.118075e+00 b=6.124777e-05 vrating=3.450000e+05
Line1013 bus10  bus13 powerline utype=1 r=4.761000e-01 x=5.118075e+00 b=6.124777e-05 vrating=3.450000e+05
Line1314 bus13  bus14 powerline utype=1 r=1.071225e+00 x=1.202153e+01 b=1.447599e-04 vrating=3.450000e+05
Line1415 bus14c bus15 powerline utype=1 r=2.142450e+00 x=2.582843e+01 b=3.074972e-04 vrating=3.450000e+05
Line1516 bus15  bus16 powerline utype=1 r=1.071225e+00 x=1.118835e+01 b=1.436676e-04 vrating=3.450000e+05
Line1617 bus16c bus17 powerline utype=1 r=8.331751e-01 x=1.059323e+01 b=1.127499e-04 vrating=3.450000e+05
Line1619 bus16  bus19 powerline utype=1 r=1.904400e+00 x=2.320988e+01 b=2.554072e-04 vrating=3.450000e+05
Line1621 bus16  bus21 powerline utype=1 r=9.522001e-01 x=1.606838e+01 b=2.140728e-04 vrating=3.450000e+05
Line1624 bus16  bus24 powerline utype=1 r=3.570751e-01 x=7.022476e+00 b=5.713112e-05 vrating=3.450000e+05
Line1718 bus17  bus18 powerline utype=1 r=8.331751e-01 x=9.760051e+00 b=1.108165e-04 vrating=3.450000e+05
Line1727 bus17  bus27 powerline utype=1 r=1.547325e+00 x=2.059133e+01 b=2.701953e-04 vrating=3.450000e+05
Line2122 bus21  bus22 powerline utype=1 r=9.522001e-01 x=1.666350e+01 b=2.155016e-04 vrating=3.450000e+05
Line2223 bus22  bus23 powerline utype=1 r=7.141501e-01 x=1.142640e+01 b=1.550941e-04 vrating=3.450000e+05
Line2324 bus23  bus24 powerline utype=1 r=2.618550e+00 x=4.165875e+01 b=3.032998e-04 vrating=3.450000e+05
Line2526 bus25  bus26 powerline utype=1 r=3.808800e+00 x=3.844508e+01 b=4.310015e-04 vrating=3.450000e+05
Line2627 bus26  bus27 powerline utype=1 r=1.666350e+00 x=1.749668e+01 b=2.013017e-04 vrating=3.450000e+05
Line2628 bus26  bus28 powerline utype=1 r=5.118075e+00 x=5.641785e+01 b=6.554902e-04 vrating=3.450000e+05
Line2629 bus26  bus29 powerline utype=1 r=6.784425e+00 x=7.439063e+01 b=8.645217e-04 vrating=3.450000e+05
Line2829 bus28  bus29 powerline utype=1 r=1.666350e+00 x=1.797278e+01 b=2.092004e-04 vrating=3.450000e+05

Trf0230 bus30 bus2  powertransformer r=0.000000e+00 x=1.810000e-01 a=2.090909e+01 kt=1.025000e+00 prating=1.000000e+09 vrating=1.650000e+04
Trf0631 bus31 bus6  powertransformer r=0.000000e+00 x=1.750000e-01 a=2.090909e+01 kt=1.070000e+00 prating=7.000000e+08 vrating=1.650000e+04
Trf1032 bus32 bus10 powertransformer r=0.000000e+00 x=1.600000e-01 a=2.090909e+01 kt=1.070000e+00 prating=8.000000e+08 vrating=1.650000e+04
Trf1032b bus32b bus10 powertransformer r=0.000000e+00 x=1.600000e-01 a=2.090909e+01 kt=1.070000e+00 prating=8.000000e+08 vrating=1.650000e+04
Trf1112 bus12 bus11 powertransformer r=4.800000e-03 x=1.304999e-01 a=2.500000e+00 kt=1.006000e+00 prating=3.000000e+08 vrating=1.380000e+05
Trf1312 bus12 bus13 powertransformer r=4.800000e-03 x=1.304999e-01 a=2.500000e+00 kt=1.006000e+00 prating=3.000000e+08 vrating=1.380000e+05
Trf1920 bus20 bus19 powertransformer r=6.999999e-03 x=1.380000e-01 a=1.500000e+00 kt=1.060000e+00 prating=1.000000e+09 vrating=2.300000e+05
Trf2034 bus34 bus20 powertransformer r=5.400000e-03 x=1.080000e-01 a=1.393939e+01 kt=1.009000e+00 prating=6.000000e+08 vrating=1.650000e+04
Trf1933 bus33 bus19 powertransformer r=5.600000e-03 x=1.135999e-01 a=2.090909e+01 kt=1.070000e+00 prating=8.000000e+08 vrating=1.650000e+04
Trf2336 bus36 bus23 powertransformer r=3.500000e-03 x=1.904000e-01 a=2.090909e+01 kt=1.000000e+00 prating=7.000000e+08 vrating=1.650000e+04
Trf2235 bus35 bus22 powertransformer r=0.000000e+00 x=1.144000e-01 a=2.090909e+01 kt=1.025000e+00 prating=8.000000e+08 vrating=1.650000e+04
Trf2938 bus38 bus29 powertransformer r=8.000000e-03 x=1.560000e-01 a=2.090909e+01 kt=1.025000e+00 prating=1.000000e+09 vrating=1.650000e+04
Trf2537 bus37 bus25 powertransformer r=4.200000e-03 x=1.624000e-01 a=2.090909e+01 kt=1.025000e+00 prating=7.000000e+08 vrating=1.650000e+04

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

Pec1  bus3  pe3  gnd qe3  gnd bus3b  powerec type=4
Pec2  bus14 pe14 gnd qe14 gnd bus14b powerec type=4
Pec3  bus16 pe16 gnd qe16 gnd bus16b powerec type=4
Pec4  bus1  pe1  gnd qe1  gnd bus1b  powerec type=4

Pec5  bus3  omegael03 gnd powerec type=2
Pec6  bus14 omegael14 gnd powerec type=2
Pec7  bus17 omegael17 gnd powerec type=2
Pec8  bus39 omegael39 gnd powerec type=2

Pec9  bus3b  id3  gnd iq3  gnd bus3c  powerec type=5
Pec10 bus14b id14 gnd iq14 gnd bus14c powerec type=5
Pec11 bus16b id16 gnd iq16 gnd bus16c powerec type=5
Pec12 bus1b  id1  gnd iq1  gnd bus1c  powerec type=5

; used to give a reference to the electric angular frequency of each bus
Coi omegacoi powercoi type=2 gen="G02" \
                  attach="G01" attach="G03" attach="G03b" attach="G04" \
                  attach="G05" attach="G06" attach="G07" \
                  attach="G08" attach="G09" attach="G10"

; the powerecs used for connecting the stochastic loads
PecStochLoad03 bus3  dload3  gnd qload3  gnd powerec type=0
PecStochLoad04 bus4  dload4  gnd qload4  gnd powerec type=0
PecStochLoad07 bus7  dload7  gnd qload7  gnd powerec type=0
PecStochLoad08 bus8  dload8  gnd qload8  gnd powerec type=0
PecStochLoad12 bus12 dload12 gnd qload12 gnd powerec type=0
PecStochLoad15 bus15 dload15 gnd qload15 gnd powerec type=0
PecStochLoad16 bus16 dload16 gnd qload16 gnd powerec type=0
PecStochLoad18 bus18 dload18 gnd qload18 gnd powerec type=0
PecStochLoad20 bus20 dload20 gnd qload20 gnd powerec type=0
PecStochLoad21 bus21 dload21 gnd qload21 gnd powerec type=0
PecStochLoad23 bus23 dload23 gnd qload23 gnd powerec type=0
PecStochLoad24 bus24 dload24 gnd qload24 gnd powerec type=0
PecStochLoad25 bus25 dload25 gnd qload25 gnd powerec type=0
PecStochLoad26 bus26 dload26 gnd qload26 gnd powerec type=0
PecStochLoad27 bus27 dload27 gnd qload27 gnd powerec type=0
PecStochLoad28 bus28 dload28 gnd qload28 gnd powerec type=0
PecStochLoad29 bus29 dload29 gnd qload29 gnd powerec type=0
PecStochLoad31 bus31 dload31 gnd qload31 gnd powerec type=0
PecStochLoad39 bus39 dload39 gnd qload39 gnd powerec type=0

end

; the stochastic loads
Rnd03 dload3  qload3  rndload3  RAND_L P=3.220*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd04 dload4  qload4  rndload4  RAND_L P=5.000*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd07 dload7  qload7  rndload7  RAND_L P=2.338*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd08 dload8  qload8  rndload8  RAND_L P=5.220*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd12 dload12 qload12 rndload12 RAND_L P=0.075*PRAND VRATING=138k  VMAX=1.2*138k  VMIN=0.8*138k
Rnd15 dload15 qload15 rndload15 RAND_L P=3.200*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd16 dload16 qload16 rndload16 RAND_L P=3.290*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd18 dload18 qload18 rndload18 RAND_L P=1.580*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd20 dload20 qload20 rndload20 RAND_L P=6.280*PRAND VRATING=230k  VMAX=1.2*230k  VMIN=0.8*230k
Rnd21 dload21 qload21 rndload21 RAND_L P=2.740*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd23 dload23 qload23 rndload23 RAND_L P=2.475*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd24 dload24 qload24 rndload24 RAND_L P=3.086*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd25 dload25 qload25 rndload25 RAND_L P=2.240*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd26 dload26 qload26 rndload26 RAND_L P=1.390*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd27 dload27 qload27 rndload27 RAND_L P=2.810*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd28 dload28 qload28 rndload28 RAND_L P=2.060*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd29 dload29 qload29 rndload29 RAND_L P=2.835*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k
Rnd31 dload31 qload31 rndload31 RAND_L P=0.092*PRAND VRATING=16.5k VMAX=1.2*16.5k VMIN=0.8*16.5k
Rnd39 dload39 qload39 rndload39 RAND_L P=11.04*PRAND VRATING=345k  VMAX=1.2*345k  VMIN=0.8*345k

Wav03 rndload3  gnd vsource wave="load_samples_bus_3"
Wav04 rndload4  gnd vsource wave="load_samples_bus_4"
Wav07 rndload7  gnd vsource wave="load_samples_bus_7"
Wav08 rndload8  gnd vsource wave="load_samples_bus_8"
Wav12 rndload12 gnd vsource wave="load_samples_bus_12"
Wav15 rndload15 gnd vsource wave="load_samples_bus_15"
Wav16 rndload16 gnd vsource wave="load_samples_bus_16"
Wav18 rndload18 gnd vsource wave="load_samples_bus_18"
Wav20 rndload20 gnd vsource wave="load_samples_bus_20"
Wav21 rndload21 gnd vsource wave="load_samples_bus_21"
Wav23 rndload23 gnd vsource wave="load_samples_bus_23"
Wav24 rndload24 gnd vsource wave="load_samples_bus_24"
Wav25 rndload25 gnd vsource wave="load_samples_bus_25"
Wav26 rndload26 gnd vsource wave="load_samples_bus_26"
Wav27 rndload27 gnd vsource wave="load_samples_bus_27"
Wav28 rndload28 gnd vsource wave="load_samples_bus_28"
Wav29 rndload29 gnd vsource wave="load_samples_bus_29"
Wav31 rndload31 gnd vsource wave="load_samples_bus_31"
Wav39 rndload39 gnd vsource wave="load_samples_bus_39"

model RAND_L   nport veriloga="randl.va"
; model IEEEG1Tg nport veriloga="ieeeg1tg.va"
; model IEEEG3Tg nport veriloga="ieeeg3tg.va"

