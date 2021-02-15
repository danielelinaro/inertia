ground electrical gnd

parameters RL=0.1m XL=1m/10 BC=1.75m 
parameters L25=25 L10=10 L110=110
parameters F0=60 TSTOP=3600
parameters TYPE=4 D=2 DZA=1
parameters PRAND=1M FRAND=10

options outintnodes=yes 

Al_dummy_tstop alter param="TSTOP" rt=yes
Al_dummy_frand alter param="FRAND" rt=yes
Al_dummy_d     alter param="D"     rt=yes
Al_dummy_dza   alter param="DZA"   rt=yes

#ifdef PAN

Step control begin

    alpha = 0.5;     %relaxation time  tau = 1/alpha
    mu    = 0;       %mean of stochatic process x
    c     = 0.5;     %diffusion constant  --> b   = c^2
    dt    = 1/FRAND;

    seed(12345);

    T = dt:dt:TSTOP+dt;

    x = zeros(length(T)); % Allocate output vector, set initial condition
    x(1) = 0; %set initial condition

    for i = 1:length(T)-1
	    x(i+1) = (x(i) + alpha * mu * dt + c * sqrt(dt) * randn()) / (1 + alpha * dt);
    end

    noise_samples = [T, x];
    Time = [1200, 2400];
    idx = 1;
    max_idx = length(Time);

    Tr tran tstop=TSTOP nettype=1 method=2 maxord=2 noisefmax=FRAND/2 noiseinj=2 \
                  seed=5061983 iabstol=1u tmax=0.1 annotate=4 devvars=yes begin 
    
        if( idx <= max_idx && time > Time(idx) )
            if ( idx == 1 )
                AlterMG1 alter instance="G1" param="m" value=2*5.5 invalidate=false
                AlterMG2 alter instance="G2" param="m" value=2*5.5 invalidate=false
            else
                AlterMG3 alter instance="G3" param="m" value=2*6.5 invalidate=false
                AlterMG4 alter instance="G4" param="m" value=2*6.5 invalidate=false
            end
            idx = idx + 1;
        end
    
    end

endcontrol

#endif

begin power

//
// Synchronous maxchines, avr and tg
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
		    vrating=20k prating=1G pg=0.719/1  \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.175

Tg4 pm04 omega04     powertg type=1 omegaref=1 r=0.002 pmax=1 pmin=0.0 \
			     ts=0.1 tc=0.45 t3=0 t4=12 t5=50 dza=DZA

Av4      g4   ex4 poweravr vmax=5 vmin=-5 ka=20 ta=0.055 te=0.36 kf=0.125 \
                           tf=1.8 tr=0.05 vrating=20k type=2 ae=0.0056 be=1.075

G4       g4   ex4 pm04 omega04 powergenerator vg=1 type=TYPE omegab=F0*2*pi \
		    vrating=20k prating=1G pg=0.7 \
		    xd=1.8 xq=1.7 xl=0.2 xdp=0.3 xqp=0.55 xds=0.25 xqs=0.25 \
		    ra=2.5m td0p=8 tq0p=0.4 td0s=0.03 tq0s=0.05 \
		    d=D m=2*6.175 

//
// Measure the power flowing at buses 7 and 9
//
Pec1 bus6a  pe7 gnd qe7 gnd bus6b  powerec type=4
Pec2 bus10a pe9 gnd qe9 gnd bus10b powerec type=4

//
// Measure the electrical angular frequency at buses 7 and 9
//
Pec3 bus7 omegael07 gnd powerec type=2
Pec4 bus9 omegael09 gnd powerec type=2

//
// Transformers connecting machine to busses
//
T15      g1  bus5  powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T26      g2 bus6a  powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T311     g3 bus11  powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M
T410     g4 bus10a powertransformer kt=230/20 x=0.15/11 vrating=20k prating=900M

//
// Lines 
//
L56    bus5  bus6a powerline prating=100M r=L25*RL x=L25*XL b=L25*BC vrating=230k
L67    bus6b  bus7 powerline prating=100M r=L10*RL x=L10*XL b=L10*BC vrating=230k

// 
// Lines connecting the two areas
//

L78    bus7   bus8   powerline prating=100M r=L110*RL x=L110*XL b=L110*BC vrating=230k 
L89    bus8   bus9   powerline prating=100M r=L110*RL x=L110*XL b=L110*BC vrating=230k
L910   bus9   bus10b powerline prating=100M r=L10*RL  x=L10*XL  b=L10*BC  vrating=230k
L1011  bus10a bus11  powerline prating=100M r=L25*RL  x=L25*XL  b=L25*BC  vrating=230k

//
// Loads
//
Lo7    bus7       powerload pc=0.967 qc=0.1+1*-0.2  vrating=230k prating=1G
Lo9    bus9       powerload pc=1.767/1.3 qc=0.1+1*-0.35 vrating=230k prating=1G

Pe8    bus8  d8  gnd  q8  gnd  powerec type=0

end

//
// Random load
//
Rnd13       d8  q8   rand8  RAND_L P=PRAND \
                            VRATING=230k VMAX=1.2*230k VMIN=0.8*230k
Wav13    rand8  gnd   port noisesamples="noise_samples"

model RAND_L nport veriloga="randl.va" verilogaprotected=1
