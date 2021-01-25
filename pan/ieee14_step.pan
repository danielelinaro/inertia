parameters TSTOP=7200 VDIG=1 LAMBDA=0 FRAND=10 F0=60 H_COEFF=1 
parameters PRAND=1M

; Synthetic inertia
parameters k0=100 t_stop_iniz=3 t_stop_fin=20 i0_limit=1e5 serie=39 \
    par=304 kt=4 vpv_ref=700 vc_ref=680 kp_dcdc=1000 ki_dcdc=1e6/10 \
    kp_DC_1=600*0 ki_DC_1=1*46875 Kd=33892 ic_limit=2e3 deltavc_limit=vc_ref*0.4 \
    Ta_wu=1e-6 CDC=3528/5 MATF="simSyn1.mat"

parameter MAXDCGAIN=1M

#ifndef SYNT
    parameters SaveSet = ["agc02","drift14","omega*","pm","bus13"]
#endif

parameter D=2 
parameter DZA=60.0/F0

options outintnodes=yes topcheck=2

Al_dummy_tstop alter param="TSTOP" rt=yes
Al_dummy_frand alter param="FRAND" rt=yes

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

    idx = 1;
    nsteps = 101;
    H_min = 3.5;
    H_max = 6.5;
    if (nsteps == 1)
        Time = [3600];
        H = [H_max];
    else
        Time = linspace(3550, 3650, nsteps);
        H    = linspace(H_min, H_max, nsteps);
    end
    max_idx = length(Time);
    
    Tr tran tstop=TSTOP nettype=1 method=2 maxord=2 noisefmax=FRAND/2 noiseinj=2 \
                  seed=5061983 iabstol=1u tmax=0.1 annotate=4 devvars=yes \
                  savelist=["omega*"] begin
    
        if( idx <= max_idx && time > Time( idx ) )
            AlterM alter instance="G1" param="m" value=2*H(idx) invalidate=false
            idx = idx + 1;
        end
    
    end

endcontrol

begin power

// Busses
;Bus1  bus01 powerbus vb=73.140000000000000k v0=1 theta0= 0.0000000000000000 
;Bus2  bus02 powerbus vb=71.782698983756112k v0=1 theta0=-6.2869206276416563 
;Bus3  bus03 powerbus vb=69.554289139197397k v0=1 theta0=-15.834946946171080 
;Bus4  bus04 powerbus vb=69.751433298936638k v0=1 theta0=-12.779797203427005 
;Bus5  bus05 powerbus vb=69.881892460201343k v0=1 theta0=-10.893841421687798 
;Bus6  bus06 powerbus vb=14.765999999999998k v0=1 theta0=-17.546618622818546 
;Bus7  bus07 powerbus vb=14.596552994087133k v0=1 theta0=-16.453088955903098 
;Bus8  bus08 powerbus vb=19.733116026409898k v0=1 theta0=-16.453088955903095 
;Bus9  bus09 powerbus vb=14.462500932767573k v0=1 theta0=-18.358002889061741 
;Bus10 bus10 powerbus vb=14.392019342276615k v0=1 theta0=-18.558489042520556 
;Bus11 bus11 powerbus vb=14.518646989246892k v0=1 theta0=-18.206955363491065 
;Bus12 bus12 powerbus vb=14.513945971068255k v0=1 theta0=-18.576083416919595 
;Bus13 bus13 powerbus vb=14.428621543734493k v0=1 theta0=-18.665426396397759 
;Bus14 bus14 powerbus vb=14.146304626782379k v0=1 theta0=-19.711269068884647 

; Lines

Li02_05   bus02   bus05  powerline prating=100M    vrating=69k   r=0.05695 \
                                         x=0.17388       b=0.034
Li06_12   bus06   bus12  powerline prating=100M    vrating=13.8k r=0.12291 \
                                         x=0.25581       b=0
Li12_13   bus12   bus13  powerline prating=100M    vrating=13.8k r=0.22092 \
                                         x=0.19988       b=0
Li06_13   bus06   bus13  powerline prating=100M    vrating=13.8k r=0.06615 \
                                         x=0.13027       b=0
Li06_11   bus06   bus11  powerline prating=100M    vrating=13.8k r=0.09498 \
                                         x=0.1989        b=0
Li11_10   bus11   bus10  powerline prating=100M    vrating=13.8k r=0.08205 \
                                         x=0.19207       b=0
Li09_10   bus09   bus10  powerline prating=100M    vrating=13.8k r=0.03181 \
                                         x=0.0845        b=0
Li09_14   bus09   bus14  powerline prating=100M    vrating=13.8k r=0.12711 \
                                         x=0.27038       b=0
Li14_13   bus14   bus13  powerline prating=100M    vrating=13.8k r=0.17093 \
                                         x=0.34802       b=0
Li07_09   bus07   bus09  powerline prating=100M    vrating=13.8k r=0       \
                                         x=0.11001       b=0
Li01_02   bus01   bus02  powerline prating=100M    vrating=69.0k r=0.01938 \
                                         x=0.05917       b=0.0528
Li03_02   bus03   bus02  powerline prating=100M    vrating=69.0k r=0.04699 \
                                         x=0.19797       b=0.0438
Li03_04   bus03   bus04  powerline prating=100M    vrating=69.0k r=0.06701 \
                                         x=0.17103       b=0.0346
Li01_05   bus01   bus05  powerline prating=100M    vrating=69.0k r=0.05403 \
                                         x=0.22304       b=0.0492
Li05_04   bus05   bus04  powerline prating=100M    vrating=69.0k r=0.01335 \
                                         x=0.04211       b=0.0128
Li02_04   bus02   bus04  powerline prating=100M    vrating=69.0k r=0.05811 \
                                         x=0.17632       b=0.0374

;Breaker.con = [ ... 
;  16  2  100  69  60  1  1  200;
; ];

; First synchronous machine - voltage regualator - turbine governor
Tg1      pm01  omega01  powertg type=1 omegaref=1 r=0.02 pmax=10 pmin=0 \
                               ts=0.1 tc=0.45 t3=0 t4=12 t5=50 gen="G1" \
			       dza=DZA

E1  bus01  avr01 poweravr vrating=69k type=2 vmax=20 vmin=-20 ka=200 ta=0.02 \
                          kf=0.002 tf=1 ke=1 te=0.2  tr=0.001 ae=0.0006 be=0.9

G1   bus01  avr01  pm01  omega01  powergenerator slack=yes \
           prating=610M \
	    vrating=69k        vg=1.06    qmax=9.9 \
               qmin=-9.9     vmax=1.2     vmin=0.8    omegab=F0*2*pi     \
	       type=52   \
		 xl=0.2396     ra=0         xd=0.8979    xdp=0.2998  xds=0.23 \
	       td0p=7.4      td0s=0.03      xq=0.646     xqp=0.646   xqs=0.4  \
	       tq0p=0        tq0s=0.033     m=7           d=D 
               ; m=H_COEFF_G1*10.296  h = m/2 h = [2,10]

; Second synchronous machine - voltage regualator - turbine governor
Tg2      pm02  omega02 agc02 powertg type=1 omegaref=1 r=0.02 pmax=4 pmin=0.3 \
                               ts=0.1 tc=0.45 t3=0 t4=12 t5=50 gen="G2" \
			       dza=DZA

E2   bus02  avr02 poweravr vrating=69k type=2 vmax=4.38 vmin=0 ka=20 ta=0.02 \
                          kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9

G2   bus02  avr02   pm02  omega02  powergenerator \
           prating=60M \
	    vrating=69k      vg=1.045     qmax=0.5 \
	       qmin=-0.4     vmax=1.2    vmin=0.8     omegab=F0*2*pi     \
	       type=6  pg=40/60 \
		 xl=0          ra=0.0031   xd=1.05       xdp=0.185   xds=0.13  \
	       td0p=6.1      td0s=0.04     xq=0.98       xqp=0.36   xqs=0.13 \
	       tq0p=0.3      tq0s=0.099     m=H_COEFF*13.08        d=D

E3  bus03  avr03 poweravr vrating=69k type=2 vmax=4.38 vmin=0 ka=20 ta=0.02 \
                          kf=0.001 tf=1 ke=1 te=1.98 tr=0.001 ae=0.0006 be=0.9
G3  bus03  avr03 powergenerator \
           prating=60M \
	    vrating=69k      vg=1.01      qmax=0.4  \
	       qmin=0        vmax=1.2    vmin=0.8     omegab=F0*2*pi     \
	       type=6  pg=0   \
		 xl=0          ra=0.0031   xd=1.05       xdp=0.185   xds=0.13 \
               td0p=6.1      td0s=0.04     xq=0.98       xqp=0.36    xqs=0.13 \
	       tq0p=0.3     tq0s=0.099      m=H_COEFF*13.08        d=D

E6  bus06  avr06 poweravr vrating=13.8k type=2 vmax=6.81 vmin=1.395 ka=20 \
                 ta=0.02 kf=0.001 tf=1 ke=1 te=0.7  tr=0.001 ae=0.0006 be=0.9
G6  bus06  avr06 powergenerator \
            prating=25M \
	    vrating=13.8k    vg=1.07      qmax=0.24 \
	       qmin=-0.06    vmax=1.2    vmin=0.8     omegab=F0*2*pi     \
	       type=6  pg=0 \
		 xl=0.134      ra=0.0014   xd=1.25       xdp=0.232   xds=0.12  \
	       td0p=4.75     td0s=0.06     xq=1.22       xqp=0.715   xqs=0.12  \
	       tq0p=1.5      tq0s=0.21      m=H_COEFF*10.12         d=D

E8  bus08  avr08 poweravr vrating=18k type=2 vmax=10 vmin=-1 ka=20 \
                 ta=0.02 kf=0.001 tf=1 ke=1 te=0.7  tr=0.001 ae=0.0006 be=0.9
G8  bus08  avr08 powergenerator \
           prating=25M \
	    vrating=18k      vg=1.09      qmax=0.24 \
	       qmin=-0.06    vmax=1.2    vmin=0.8     omegab=F0*2*pi     \
	       type=6  pg=0 \
	         xl=0.134      ra=0.0014   xd=1.25       xdp=0.232   xds=0.12  \
	       td0p=4.75     td0s=0.06     xq=1.22       xqp=0.715   xqs=0.12  \
	       tq0p=1.5      tq0s=0.21      m=H_COEFF*10.12        d=D

; Center of inertia

Coi  omegacoi powercoi gen="G1" gen="G2" gen="G3" gen="G6" gen="G8" type=0

; Transformers

Tr05_06   bus06   bus05  powertransformer \
                          prating=100M  vrating=13.8k   kt=5 \
                                r=0           x=0.25202  a=0.932

Tr04_09   bus09   bus04  powertransformer \
                          prating=100M  vrating=13.8k   kt=5 \
			        r=0           x=0.55618  a=0.969

Tr04_07   bus07   bus04  powertransformer \
                          prating=100M  vrating=13.8k   kt=5 \
			        r=0           x=0.20912  a=0.978

Tr08_07   bus07   bus08  powertransformer \
                          prating=100M  vrating=13.8k   kt=1.304348 \
			        r=0           x=0.17615  a=1

; Loads

Lo11     bus11  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.035  qc=(1+LAMBDA)*0.018 \
                          vmax=1.2  vmin=0.8
Lo13     bus13  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.135  qc=(1+LAMBDA)*0.058 \
                          vmax=1.2  vmin=0.8
Lo03     bus03  powerload prating=100M vrating=69k   \
                          pc=(1+LAMBDA)*0.942 qc=(1+LAMBDA)*0.19  \
                          vmax=1.2  vmin=0.8
Lo05     bus05  powerload prating=100M vrating=69k   \
                          pc=(1+LAMBDA)*0.076 qc=(1+LAMBDA)*0.016 \
                          vmax=1.2  vmin=0.8
Lo02     bus02  powerload prating=100M vrating=69k   \
                          pc=(1+LAMBDA)*0.217 qc=(1+LAMBDA)*0.127 \
                          vmax=1.2  vmin=0.8
Lo06     bus06  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.112 qc=(1+LAMBDA)*0.075  \
                          vmax=1.2  vmin=0.8
Lo04     bus04  powerload prating=100M vrating=69k   \
                          pc=(1+LAMBDA)*0.478 qc=(1+LAMBDA)*0.04  \
                          vmax=1.2  vmin=0.8
Lo14     bus14  drift14 powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.149 qc=(1+LAMBDA)*0.05   \
                          vmax=1.2  vmin=0.8
Lo12     bus12  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.061 qc=(1+LAMBDA)*0.016 \
                          vmax=1.2  vmin=0.8
Lo10     bus10  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.09  qc=(1+LAMBDA)*0.058 \
                          vmax=1.2  vmin=0.8
Lo09     bus09  powerload prating=100M vrating=13.8k \
                          pc=(1+LAMBDA)*0.295  qc=(1+LAMBDA)*0.166 \
                          vmax=1.2  vmin=0.8

Pe13     bus13  d13  gnd  q13  gnd  powerec type=0

#ifdef SYNT
Ec bus11 dp gnd qp gnd powerec type=0 vrating=13.8k
#endif

end

AgcG1  omegacoi agc02  AGC K=0*10m

;Dfr14   drift14  gnd  vsource t=0 v=0 t=TSTOP/2 v=0.24 t=TSTOP+1m v=0
Dfr14   drift14  gnd  vsource vsin=-0*0.12 freq=1/(TSTOP)

#ifdef SYNT
f_PLL omegacoi pll gnd powerec type=0
Ix pll dp qp SYNT
#endif

Rnd13       d13  q13   rand13    RAND_L P=PRAND \
			    VRATING=13.8k VMAX=1.2*13.8k VMIN=0.8*13.8k
Wav13    rand13  gnd   port noisesamples="noise_samples"

Iac         d13  gnd   isource mag=1

subckt SYNT pll dp qp
;RETE ASSE DIRETTO E INVERSO
Xr  dp gnd qp pp qq vccs func=(-v(dp)*v(pp)+v(qp)*v(qq))/max(k0,v(dp)^2+v(qp)^2)
Xi  qp gnd dp pp qq vccs func=(-v(qp)*v(pp)-v(dp)*v(qq))/max(k0,v(dp)^2+v(qp)^2)

Ep  pp gnd nport sensen="i0_meas" sensen="vc_meas" func1=v(p1)-v(i0_meas)*v(vc_meas)
Eq  qq gnd vsource vdc=0 ; sempre nulla.

;BUS DC

Tpv    T   gnd        vsource vdc=25
Spv    S   gnd        vsource vdc=1000

I1   pv1   gnd T   S   PV NS=36*serie RS=0.005/par RSH=1000*par \
                          IS0=1.16e-8*par CT=3.25m*par ISC0=5*par
Tf   pv2   gnd   pv1   gnd  kt transformer t1=kt t2=1 minkt=0.00001 maxkt=100000

Pi   deltakt   gnd    pv1  ref  svcvs numer=[ki_dcdc,kp_dcdc] denom=[0 1] maxdcgain=MAXDCGAIN
Vrf  ref   gnd        vsource vdc=vpv_ref
kt   kt    gnd deltakt vcvs func=v(deltakt)+vc_ref/(4*vpv_ref)

ipvcto  pv2 pv3 vsource     vdc=0
cdc     pv3 gnd capacitor   c=CDC
i0cto   pv3 pv4 vsource     vdc=0
i0      pv4 gnd ipv_meas gnd ic gnd  vccs func=limit(v(ipv_meas,gnd)-v(ic,gnd),-i0_limit*0-1,i0_limit)

;MISURE
ipv_meas    ipv_meas    gnd ccvs sensedev="ipvcto"  gain1=1
i0_meas     i0_meas     gnd ccvs sensedev="i0cto"   gain1=1
vc_meas     vc_meas     gnd pv3 gnd vcvs gain1=1

ppv_meas    ppv_meas    gnd nport sensen="vc_meas" sensen="ipv_meas" \
                                  func1=v(p1)-v(ipv_meas)*v(vc_meas)
p0_meas     p0_meas     gnd nport sensen="vc_meas" sensen="i0_meas" \
                                  func1=v(p1)-v(i0_meas)*v(vc_meas)

controlloconv vc_meas pll ic controllo

model PV nport veriloga="pvMod.va"

ends

subckt controllo tensionedc frequenza correnteic
    
;MISURE DI RIFERIMENTO
vc_ref vc_ref gnd vsource vdc=vc_ref
    
f_error f_error gnd frequenza vcvs func=(v(frequenza)-1)*F0
delta_vdc_ref  delta_vdc_ref gnd f_error vcvs \
    func=limit(v(f_error)*Kd,-deltavc_limit,deltavc_limit)

vc_error vc_error gnd vc_ref gnd tensionedc gnd delta_vdc_ref gnd vcvs \
    gain1=1 gain2=-1 gain3=1
antiwindup_interno vc_error correnteic antiwu kp=kp_DC_1 ki=ki_DC_1 \
    limit=ic_limit Ta=Ta_wu
ends

subckt  antiwu  error1 out
parameters kp ki limit Ta

prop    prop    gnd     error1  gnd     vcvs    gain1=kp
int     int     gnd     error2  gnd     svcvs   numer=[ki]   denom=[0,1] maxdcgain=MAXDCGAIN
out     out     gnd     prop    int     vcvs    func=limit(v(prop)+v(int),-limit,limit)
corr    corr    gnd     out     prop    int     vcvs    func=(v(prop)+v(int)-v(out))/Ta
error2  error2  gnd     corr    error1  vcvs    func=v(error1)-v(corr)
ends

define AGC omega pm
parameters OMEGA0=1 K=1m 

I1  x10   gnd  omega vcvs func=K*(OMEGA0-v(omega))
I2   pm   gnd  x10  gnd  svcvs numer=[1] denom=[0,1] maxdcgain=MAXDCGAIN

end

model RAND_L nport veriloga="randl.va" 

ground electrical gnd
