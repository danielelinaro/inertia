ground electrical gnd

parameters RL=0.1m XL=1m/10 BC=1.75m 
parameters L25=25 L10=10 L110=110
parameters F0=60 TSTOP=60
parameters TYPE=4 D=2 DZA=1
parameters PRAND=1M FRAND=10
parameter LAMBDA=0 ; coefficient for overloading area 2
parameter COEFF=0  ; coefficient for varying the active power of the loads in the two areas

options outintnodes=yes 

Al_dummy_tstop  alter param="TSTOP"  rt=yes
Al_dummy_frand  alter param="FRAND"  rt=yes
Al_dummy_d      alter param="D"      rt=yes
Al_dummy_dza    alter param="DZA"    rt=yes
Al_dummy_lambda alter param="LAMBDA" rt=yes
Al_dummy_coeff  alter param="COEFF"  rt=yes

#ifdef PAN

Rnd control begin

    dt = 1/FRAND;
    T  = [dt:dt:TSTOP+dt];

    noise_samples = [ T, randn( length(T) ) ];

endcontrol

;Dc dc nettype=yes print=yes printnodes=yes ireltol=1m vreltol=1m
;Pz pz nettype=yes
Tr tran stop=TSTOP nettype=yes method=2 maxord=2 noisefmax=FRAND/2 \
        noiseinj=2 seed=12345 iabstol=1u devvars=1 tmax=0.1 

#endif

begin power

//
// Synchronous machines, avr and tg
//

Tg1 pm01 omega01     powertg type=1 omegaref=1 r=0.002 pmax=1 pmin=0.0 \
			     ts=0.1 tc=0.45 t3=0 t4=12 t5=50 gen="G1" dza=DZA

Av1      g1   ex1 poweravr vmax=5 vmin=-5 ka=20 ta=0.055 te=0.36 kf=0.125 \
                           tf=1.8 tr=0.05 vrating=20k type=2 ae=0.0056 be=1.075

G1       g1   ex1  pm01 omega01 powergenerator slack=yes vg=1 type=TYPE \
                    omegab=F0*2*pi \
		    vrating=20k prating=1G pg=0.7 \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.5

Av2      g2   ex2 poweravr vmax=5 vmin=-5 ka=20 ta=0.055 te=0.36 kf=0.125 \
                           tf=1.8 tr=0.05 vrating=20k type=2 ae=0.0056 be=1.075

Tg2 pm02 omega02     powertg type=1 omegaref=1 r=0.02 pmax=1 pmin=0.0 \
			     ts=0.1 tc=0.45 t3=0 t4=12 t5=50 dza=DZA

G2       g2   ex2 pm02 omega02 powergenerator vg=1 type=TYPE omegab=F0*2*pi \
		    vrating=20k prating=1G pg=0.7 \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.5

Tg3 pm03 omega03     powertg type=1 omegaref=1 r=0.02 pmax=1 pmin=0.0 \
			     ts=0.1 tc=0.45 t3=0 t4=12 t5=50 dza=DZA

Av3      g3   ex3 poweravr vmax=5 vmin=-5 ka=20 ta=0.055 te=0.36 kf=0.125 \
                           tf=1.8 tr=0.05 vrating=20k type=2 ae=0.0056 be=1.075

G3       g3   ex3 pm03 omega03 powergenerator vg=1 type=TYPE omegab=F0*2*pi \
		    vrating=20k prating=1G pg=0.719*(1+LAMBDA)  \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.175

Tg4 pm04 omega04     powertg type=1 omegaref=1 r=0.002 pmax=1 pmin=0.0 \
			     ts=0.1 tc=0.45 t3=0 t4=12 t5=50 dza=DZA

Av4      g4   ex4 poweravr vmax=5 vmin=-5 ka=20 ta=0.055 te=0.36 kf=0.125 \
                           tf=1.8 tr=0.05 vrating=20k type=2 ae=0.0056 be=1.075

G4       g4   ex4 pm04 omega04 powergenerator vg=1 type=TYPE omegab=F0*2*pi \
		    vrating=20k prating=1G pg=0.7*(1+LAMBDA) \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.175 

//
// Measure the power flowing at buses 6, 7, 9 and 10
//
Pec1 bus6a  pe6  gnd qe6  gnd bus6b  powerec type=4
Pec2 bus7a  pe7  gnd qe7  gnd bus7b  powerec type=4
Pec3 bus9a  pe9  gnd qe9  gnd bus9b  powerec type=4
Pec4 bus10a pe10 gnd qe10 gnd bus10b powerec type=4

//
// Measure the electrical angular frequency and phase angles at buses 6, 7, 9 and 10
//
Pec5  bus6a  omegael06 gnd powerec type=2
Pec6  bus7a  omegael07 gnd powerec type=2
Pec7  bus9a  omegael09 gnd powerec type=2
Pec8  bus10a omegael10 gnd powerec type=2
Pec9  bus6a  deltael06 gnd powerec type=1
Pec10 bus7a  deltael07 gnd powerec type=1
Pec11 bus9a  deltael09 gnd powerec type=1
Pec12 bus10a deltael10 gnd powerec type=1

//
// Transformers connecting machine to busses
//
T15      g1  bus5   powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T26      g2  bus6a  powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T311     g3  bus11  powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T410     g4  bus10a powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M

//
// Lines 
//
L56    bus5   bus6a  powerline prating=100M r=L25*RL x=L25*XL b=L25*BC vrating=230k
L67    bus6b  bus7a  powerline prating=100M r=L10*RL x=L10*XL b=L10*BC vrating=230k
L910   bus9a  bus10b powerline prating=100M r=L10*RL x=L10*XL b=L10*BC vrating=230k
L1011  bus10a bus11  powerline prating=100M r=L25*RL x=L25*XL b=L25*BC vrating=230k

// 
// Lines connecting the two areas
//
L78    bus7b  bus8   powerline prating=100M r=L110*RL x=L110*XL b=L110*BC vrating=230k 
L89    bus8   bus9b  powerline prating=100M r=L110*RL x=L110*XL b=L110*BC vrating=230k

//
// Loads
//
Lo7    bus7a  powerload pc=0.967*(1+COEFF)                qc=-0.1             vrating=230k prating=1G
Lo9    bus9a  powerload pc=1.767/1.3*(1+COEFF)*(1+LAMBDA) qc=-0.25*(1+LAMBDA) vrating=230k prating=1G
;Lo7    bus7a      cntp powerload pc=0.967*(1+COEFF)                qc=-0.1             vrating=230k prating=1G
;Lo9    bus9a      cntp powerload pc=1.767/1.3*(1+COEFF)*(1+LAMBDA) qc=-0.25*(1+LAMBDA) vrating=230k prating=1G

;Pec13   bus5  d5  gnd  q5  gnd  powerec type=0
Pec14   bus8  d8  gnd  q8  gnd  powerec type=0
;Pec15   bus11 d11 gnd  q11 gnd  powerec type=0

Coi powercoi gen="G1" gen="G2" gen="G3" gen="G4" type=2

end

;CntLo cntp gnd vsource vsin=0.02 freq=1/(24*3600/2)

//
// Stochastic load(s)
//
;Rnd5       d5  q5   rand5  RAND_L P=PRAND VRATING=230k VMAX=1.2*230k VMIN=0.8*230k
;Wav5    rand5  gnd   vsource wave="noise_samples_bus_5"
Rnd8       d8  q8   rand8  RAND_L P=PRAND VRATING=230k VMAX=1.2*230k VMIN=0.8*230k
Wav8    rand8  gnd   vsource wave="noise_samples_bus_8"
;Rnd11      d11 q11  rand11 RAND_L P=PRAND VRATING=230k VMAX=1.2*230k VMIN=0.8*230k
;Wav11    rand11  gnd   vsource wave="noise_samples_bus_11"

model RAND_L nport veriloga="randl.va" verilogaprotected=1

