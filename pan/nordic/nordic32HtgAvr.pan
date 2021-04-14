ground electrical gnd

parameters F0=50 LAMBDA=0 TCHECK=120
parameters PERIOD=1/F0 PWM_FREQ=213*F0 PWM_PERIOD=1/PWM_FREQ \
           L1=1.8m R1=0.1
parameters P=1G*1.216/4 Q=0 K=1 VSC_DC=1100*1k/K
parameters TR_TIME=1p V_DIG=1 TSTOP=15*PERIOD
parameters G_KI=50 G_KP=20

parameters RA=0 XL=0.15 D=0 XDS=0.2 XQS=0.2
parameters PG1=600.0/800 PG2=300.0/600 PG3=550.0/700 PG4=400.0/600 PG5=200.0/250
parameters PG8=750.0/850 PG9=668.5/1000 PG10=600.0/800 PG11=250.0/300 PG12=310.0/350
parameters PG19=300.0/500 PG20=2137.4/4500

parameter ALPHAG=1 ALPHAL=1 BETAL=1 BETAT=1

options outintnodes=1 maxcputime=200

#ifdef HVDC

TrI   tran    stop=TSTOP uic=1 restart=1 ireltol=1m iabstol=1u nettype=no \
              printnodes=0
ShI   shooting fund=F0 restart=0 solver=0 floquet=yes \
	      method=2 maxord=2 annotate=5 damping=0.4 tmax=1m/F0 \
	      iabstol=1n nettype=no ereltol=5m

Dc dc printnodes=0 nettype=2 print=no sparse=2 gminstepping=false \
                   ggroundstepping=no

ShPf shooting fund=F0 solver=0 floquet=true printnodes=no \
              fft=no fftharms=1024 method=2 maxord=2 nettype=3 restart=no \
	      tmax=1m/F0 ereltol=1m eabstol=1m devvars=yes \
	      printmo=0 trabstol=10n iabstol=1n \
	      tmin=100n

#else

Init control begin
 
     Open  = NO;

endcontrol

Dc dc nettype=1 print=0 printnodes=0 sparse=2 gminstepping=false \
      ggroundstepping=no
Pz pz nettype=1 shift=3.276152e-01
Tr tran stop=400 devvars=true nettype=1 annotate=5 saman=1 tinc=1.5 \
        ireltol=1m iabstol=1u vreltol=1m sparse=2 acntrl=3 ltefactor=1 \
	method=2 maxord=2 savelist=["[Oo]mega","tap"] begin

    if( time > 50 && Open == NO )
        Open = YES;
        AlLine alter param="open" instance=Ln#4032#4044 value=Open
    end

    if( time > TCHECK && get("Tr.G#g14:omega",-1) > 1.025 )
        break;
    end

end

Test control begin
    Poles = get("Pz.poles");

    MaxReal = max( real( Poles ) )/(2*pi);

    printf( "Max real pole: %e\n", MaxReal );

    if( MaxReal < 300p )
	printf( "+------------------+\n" );
	printf( "| POLE TEST PASSED |\n" );
	printf( "| STABLE SYSTEM    |\n" );
	printf( "+------------------+\n" );
    else
	printf( "+--------------------------+\n" );
	printf( "| POLE TEST **NOT** PASSED |\n" );
	printf( "+--------------------------+\n" );
	abort();
    endif

    Idx = find( get("Tr.time") > TCHECK );
    Omega = get("Tr.G#g6:omega");

    if( 1 - min( Omega(Idx) ) > 6m && max( Omega(Idx) ) - 1 > 6m )
	printf( "+-------------+\n" );
	printf( "| TEST PASSED |\n" );
	printf( "+-------------+\n" );
    else
	printf( "+---------------------+\n" );
	printf( "| TEST **NOT** PASSED |\n" );
	printf( "+---------------------+\n" );
	abort();
    endif
endcontrol

#endif

begin power

; +------------+
; | Generators |
; +------------+

G#g1   g1 avr1 pm1 omega1 powergenerator vrating=15k pg=ALPHAG*PG1 vg=1.0684 \
            prating=800M xd=1.1 xq=0.70 xdp=0.25 \
	    xqp=0.0 xds=0.2 xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g2   g2 avr2 pm2 omega2 powergenerator vrating=15k pg=ALPHAG*PG2 vg=1.0565 prating=600M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g3   g3 avr3 pm3 omega3 powergenerator vrating=15k pg=ALPHAG*PG3 vg=1.0595 prating=700M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g4   g4 avr4 pm4 omega4 powergenerator vrating=15k pg=ALPHAG*PG4 vg=1.0339 prating=600M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g5   g5 avr5 pm5 omega5 powergenerator vrating=15k pg=ALPHAG*PG5 vg=1.0294 prating=250M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g6   g6 avr6 powergenerator vrating=15k pg=ALPHAG*360.0/400  vg=1.0084 prating=400M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g7   g7 avr7 powergenerator vrating=15k pg=ALPHAG*180.0/200  vg=1.0141 prating=200M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g8   g8 avr8 pm8 omega8 powergenerator vrating=15k pg=ALPHAG*PG8  vg=1.0498 prating=850M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g9   g9 avr9 pm9 omega9 powergenerator vrating=15k pg=ALPHAG*PG9  vg=0.9988 prating=1000M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g10 g10 avr10 pm10 omega10 powergenerator vrating=15k pg=ALPHAG*PG10 vg=1.0157 prating=800M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g11 g11 avr11 pm11 omega11 powergenerator vrating=15k pg=ALPHAG*PG11  vg=1.0211 prating=300M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g12 g12 avr12 pm12 omega12 powergenerator vrating=15k pg=ALPHAG*PG12 vg=1.0200 prating=350M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g13 g13 avr13 powergenerator vrating=15k pg=ALPHAG*0.0/300    vg=1.0170 prating=300M \
            xd=1.55 xq=1.00 xdp=0.30 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=7 tq0p=0.0 td0s=0.05 tq0s=0.10 h=2 type=52 ra=RA xl=XL d=D phtype=2

G#g14 g14 avr14 powergenerator vrating=15k pg=ALPHAG*630.0/700  vg=1.0454 prating=700M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g15 g15 avr15 powergenerator vrating=15k pg=ALPHAG*1080.0/1200 vg=1.0455 prating=1200M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g16 g16 avr16 powergenerator vrating=15k pg=ALPHAG*600.0/700  vg=1.0531 prating=700M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

#ifdef g16b

G#g16b g16b powergenerator vrating=15k pg=ALPHAG*600.0/700  vg=1.0531 prating=700M \
             xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	     td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

#endif

G#g17 g17 avr17 powergenerator vrating=15k pg=ALPHAG*530.0/600  vg=1.0092 prating=600M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g18 g18 avr18 powergenerator vrating=15k pg=ALPHAG*1060.0/1200 vg=1.0307 prating=1200M \
            xd=2.2 xq=2.00 xdp=0.30 xqp=0.40 xds=XDS xqs=XQS \
	    td0p=7 tq0p=1.5 td0s=0.05 tq0s=0.05 h=6 type=6 ra=RA xl=XL d=D phtype=2

G#g19 g19 avr19 pm19 omega19 powergenerator vrating=15k pg=ALPHAG*PG19 vg=1.0300 prating=500M \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 ra=RA xl=XL d=D phtype=2

G#g20 g20 avr20 pm20 omega20 powergenerator vrating=15k pg=PG20 vg=1.0185 prating=4500M slack=1 \
            xd=1.1 xq=0.70 xdp=0.25 xqp=0.0 xds=XDS xqs=XQS \
	    td0p=5 tq0p=0.0 td0s=0.05 tq0s=0.10 h=3 type=52 d=D ra=RA xl=XL phtype=2

; +-------+
; | Lines |
; +-------+
Ln#1011#1013  bus1011 bus1013 powerline r=1.69*BETAL x=11.83*BETAL b=40.841u*2*BETAL utype=1  
Ln#1011#1013b bus1011 bus1013 powerline r=1.69*BETAL x=11.83*BETAL b=40.841u*2*BETAL utype=1  
Ln#1012#1014  bus1012 bus1014 powerline r=2.37*BETAL x=15.21*BETAL b=53.407u*2*BETAL utype=1  
Ln#1012#1014b bus1012 bus1014 powerline r=2.37*BETAL x=15.21*BETAL b=53.407u*2*BETAL utype=1  
Ln#1013#1014  bus1013 bus1014 powerline r=1.18*BETAL x=8.450*BETAL b=29.845u*2*BETAL utype=1  
Ln#1013#1014b bus1013 bus1014 powerline r=1.18*BETAL x=8.450*BETAL b=29.845u*2*BETAL utype=1  
Ln#1021#1022  bus1021 bus1022 powerline r=5.07*BETAL x=33.80*BETAL b=89.535u*2*BETAL utype=1  
Ln#1021#1022b bus1021 bus1022 powerline r=5.07*BETAL x=33.80*BETAL b=89.535u*2*BETAL utype=1  
Ln#1041#1043  bus1041 bus1043 powerline r=1.69*BETAL x=10.14*BETAL b=36.128u*2*BETAL utype=1  
Ln#1041#1043b bus1041 bus1043 powerline r=1.69*BETAL x=10.14*BETAL b=36.128u*2*BETAL utype=1  
Ln#1041#1045  bus1041 bus1045 powerline r=2.53*BETAL x=20.28*BETAL b=73.827u*2*BETAL utype=1  
Ln#1041#1045b bus1041 bus1045 powerline r=2.53*BETAL x=20.28*BETAL b=73.827u*2*BETAL utype=1  
Ln#1042#1044  bus1042 bus1044 powerline r=6.42*BETAL x=47.32*BETAL b=177.50u*2*BETAL utype=1  
Ln#1042#1044b bus1042 bus1044 powerline r=6.42*BETAL x=47.32*BETAL b=177.50u*2*BETAL utype=1  
Ln#1042#1045  bus1042 bus1045 powerline r=8.45*BETAL x=50.70*BETAL b=177.50u*2*BETAL utype=1  
Ln#1043#1044  bus1043 bus1044 powerline r=1.69*BETAL x=13.52*BETAL b=47.124u*2*BETAL utype=1  
Ln#1043#1044b bus1043 bus1044 powerline r=1.69*BETAL x=13.52*BETAL b=47.124u*2*BETAL utype=1  
Ln#2031#2032  bus2031 bus2032 powerline r=5.81*BETAL x=43.56*BETAL b=15.708u*2*BETAL utype=1  
Ln#2031#2032b bus2031 bus2032 powerline r=5.81*BETAL x=43.56*BETAL b=15.708u*2*BETAL utype=1  
Ln#4011#4012  bus4011 bus4012 powerline r=1.60*BETAL x=12.80*BETAL b=62.832u*2*BETAL utype=1  
Ln#4011#4021  bus4011 bus4021 powerline r=9.60*BETAL x=96.00*BETAL b=562.34u*2*BETAL utype=1  
Ln#4011#4022  bus4011 bus4022 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4011#4071  bus4011 bus4071 powerline r=8.00*BETAL x=72.00*BETAL b=438.25u*2*BETAL utype=1  
Ln#4012#4022  bus4012 bus4022 powerline r=6.40*BETAL x=56.00*BETAL b=328.30u*2*BETAL utype=1  
Ln#4012#4071  bus4012 bus4071 powerline r=8.00*BETAL x=80.00*BETAL b=468.10u*2*BETAL utype=1  
Ln#4021#4032  bus4021 bus4032 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4021#4042  bus4021 bus4042 powerline r=16.0*BETAL x=96.00*BETAL b=937.77u*2*BETAL utype=1  
Ln#4022#4031  bus4022 bus4031 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4022#4031b bus4022 bus4031 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4031#4032  bus4031 bus4032 powerline r=1.60*BETAL x=16.00*BETAL b=94.248u*2*BETAL utype=1  
Ln#4031#4041  bus4031 bus4041 powerline r=9.60*BETAL x=64.00*BETAL b=749.27u*2*BETAL utype=1  
Ln#4031#4041b bus4031 bus4041 powerline r=9.60*BETAL x=64.00*BETAL b=749.27u*2*BETAL utype=1  
Ln#4032#4042  bus4032 bus4042 powerline r=16.0*BETAL x=64.00*BETAL b=625.18u*2*BETAL utype=1  
Ln#4032#4044  bus4032 bus4044 powerline r=9.60*BETAL x=80.00*BETAL b=749.27u*2*BETAL utype=1  
Ln#4041#4044  bus4041 bus4044 powerline r=4.80*BETAL x=48.00*BETAL b=281.17u*2*BETAL utype=1  
Ln#4041#4061  bus4041 bus4061 powerline r=9.60*BETAL x=72.00*BETAL b=406.84u*2*BETAL utype=1  
Ln#4042#4043  bus4042 bus4043 powerline r=3.20*BETAL x=24.00*BETAL b=155.51u*2*BETAL utype=1  
Ln#4042#4044  bus4042 bus4044 powerline r=3.20*BETAL x=32.00*BETAL b=186.93u*2*BETAL utype=1  
Ln#4043#4044  bus4043 bus4044 powerline r=1.60*BETAL x=16.00*BETAL b=94.248u*2*BETAL utype=1  
Ln#4043#4046  bus4043 bus4046 powerline r=1.60*BETAL x=16.00*BETAL b=94.248u*2*BETAL utype=1  
Ln#4043#4047  bus4043 bus4047 powerline r=3.20*BETAL x=32.00*BETAL b=186.93u*2*BETAL utype=1  
Ln#4044#4045  bus4044 bus4045 powerline r=3.20*BETAL x=32.00*BETAL b=186.93u*2*BETAL utype=1  
Ln#4044#4045b bus4044 bus4045 powerline r=3.20*BETAL x=32.00*BETAL b=186.93u*2*BETAL utype=1  
Ln#4045#4051  bus4045 bus4051 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4045#4051b bus4045 bus4051 powerline r=6.40*BETAL x=64.00*BETAL b=375.42u*2*BETAL utype=1  
Ln#4045#4062  bus4045 bus4062 powerline r=17.6*BETAL x=128.0*BETAL b=749.27u*2*BETAL utype=1  
Ln#4046#4047  bus4046 bus4047 powerline r=1.60*BETAL x=24.00*BETAL b=155.51u*2*BETAL utype=1  
Ln#4061#4062  bus4061 bus4062 powerline r=3.20*BETAL x=32.00*BETAL b=186.93u*2*BETAL utype=1  
Ln#4062#4063  bus4062 bus4063 powerline r=4.80*BETAL x=48.00*BETAL b=281.17u*2*BETAL utype=1  
Ln#4062#4063b bus4062 bus4063 powerline r=4.80*BETAL x=48.00*BETAL b=281.17u*2*BETAL utype=1  
Ln#4071#4072  bus4071 bus4072 powerline r=4.80*BETAL x=48.00*BETAL b=937.77u*2*BETAL utype=1  
Ln#4071#4072b bus4071 bus4072 powerline r=4.80*BETAL x=48.00*BETAL b=937.77u*2*BETAL utype=1  

; +----------------------+
; | Step-up transformers |
; +----------------------+
Tr#g1   g1 bus1012 powertransformer x=0.15*BETAT kt=1.00 prating=800.0M  vrating=15k a=130/15
Tr#g2   g2 bus1013 powertransformer x=0.15*BETAT kt=1.00 prating=600.0M  vrating=15k a=130/15
Tr#g3   g3 bus1014 powertransformer x=0.15*BETAT kt=1.00 prating=700.0M  vrating=15k a=130/15
Tr#g4   g4 bus1021 powertransformer x=0.15*BETAT kt=1.00 prating=600.0M  vrating=15k a=130/15
Tr#g5   g5 bus1022 powertransformer x=0.15*BETAT kt=1.05 prating=250.0M  vrating=15k a=130/15
Tr#g6   g6 bus1042 powertransformer x=0.15*BETAT kt=1.05 prating=400.0M  vrating=15k a=130/15
Tr#g7   g7 bus1043 powertransformer x=0.15*BETAT kt=1.05 prating=200.0M  vrating=15k a=130/15
Tr#g8   g8 bus2032 powertransformer x=0.15*BETAT kt=1.05 prating=850.0M  vrating=15k a=220/15
Tr#g9   g9 bus4011 powertransformer x=0.15*BETAT kt=1.05 prating=1000.0M vrating=15k a=400/15
Tr#g10 g10 bus4012 powertransformer x=0.15*BETAT kt=1.05 prating=800.0M  vrating=15k a=400/15
Tr#g11 g11 bus4021 powertransformer x=0.15*BETAT kt=1.05 prating=300.0M  vrating=15k a=400/15
Tr#g12 g12 bus4031 powertransformer x=0.15*BETAT kt=1.05 prating=350.0M  vrating=15k a=400/15
Tr#g13 g13 bus4041 powertransformer x=0.10*BETAT kt=1.05 prating=300.0M  vrating=15k a=400/15
Tr#g14 g14 bus4042 powertransformer x=0.15*BETAT kt=1.05 prating=700.0M  vrating=15k a=400/15
Tr#g15 g15 bus4047 powertransformer x=0.15*BETAT kt=1.05 prating=1200.0M vrating=15k a=400/15
Tr#g16 g16 bus4051 powertransformer x=0.15*BETAT kt=1.05 prating=700.0M  vrating=15k a=400/15
Tr#g17 g17 bus4062 powertransformer x=0.15*BETAT kt=1.05 prating=600.0M  vrating=15k a=400/15
Tr#g18 g18 bus4063 powertransformer x=0.15*BETAT kt=1.05 prating=1200.0M vrating=15k a=400/15
Tr#g19 g19 bus4071 powertransformer x=0.15*BETAT kt=1.05 prating=500.0M  vrating=15k a=400/15
Tr#g20 g20 bus4072 powertransformer x=0.15*BETAT kt=1.05 prating=4500.0M vrating=15k a=400/15

; +----------------------------------+
; | 400/220 and 400/130 transformers |
; +----------------------------------+
Tr#1011#4011  bus1011 bus4011 powertransformer x=0.10*BETAT kt=0.95 prating=1250.0M vrating=130k a=400/130
Tr#1012#4012  bus1012 bus4012 powertransformer x=0.10*BETAT kt=0.95 prating=1250.0M vrating=130k a=400/130
Tr#1022#4022  bus1022 bus4022 powertransformer x=0.10*BETAT kt=0.93 prating=833.3M  vrating=130k a=400/130
Tr#2031#4031  bus2031 bus4031 powertransformer x=0.10*BETAT kt=1.00 prating=833.3M  vrating=220k a=400/220
Tr#1044#4044  bus1044 bus4044 powertransformer x=0.10*BETAT kt=1.03 prating=1000.0M vrating=130k a=400/130
Tr#1044#4044b bus1044 bus4044 powertransformer x=0.10*BETAT kt=1.03 prating=1000.0M vrating=130k a=400/130
Tr#1045#4045  bus1045 bus4045 powertransformer x=0.10*BETAT kt=1.04 prating=1000.0M vrating=130k a=400/130
Tr#1045#4045b bus1045 bus4045 powertransformer x=0.10*BETAT kt=1.04 prating=1000.0M vrating=130k a=400/130

; +-------------------------+
; | Step-downp transformers |
; +-------------------------+
Tr#11#1011 bus11 bus1011 tap11 powertransformer x=0.10*BETAT kt=1.00 prating=400.0M  vrating=20k a=1.04*130/20 minkt=0.88 maxkt=1.2
Tr#12#1012 bus12 bus1012 tap12 powertransformer x=0.10*BETAT kt=1.00 prating=600.0M  vrating=20k a=1.05*130/20 minkt=0.88 maxkt=1.2
Tr#13#1013 bus13 bus1013 tap13 powertransformer x=0.10*BETAT kt=1.00 prating=200.0M  vrating=20k a=1.04*130/20 minkt=0.88 maxkt=1.2
Tr#22#1022 bus22 bus1022 tap22 powertransformer x=0.10*BETAT kt=1.00 prating=560.0M  vrating=20k a=1.04*130/20 minkt=0.88 maxkt=1.2
Tr#1#1041  bus1  bus1041 tap1  powertransformer x=0.10*BETAT kt=1.00 prating=1200.0M vrating=20k a=1.00*130/20 minkt=0.88 maxkt=1.2
Tr#2#1042  bus2  bus1042 tap2  powertransformer x=0.10*BETAT kt=1.00 prating=600.0M  vrating=20k a=1.00*130/20 minkt=0.88 maxkt=1.2
Tr#3#1043  bus3  bus1043 tap3  powertransformer x=0.10*BETAT kt=1.00 prating=460.0M  vrating=20k a=1.01*130/20 minkt=0.88 maxkt=1.2
Tr#4#1044  bus4  bus1044 tap4  powertransformer x=0.10*BETAT kt=1.00 prating=1600.0M vrating=20k a=0.99*130/20 minkt=0.88 maxkt=1.2
Tr#5#1045  bus5  bus1045 tap5  powertransformer x=0.10*BETAT kt=1.00 prating=1400.0M vrating=20k a=1.00*130/20 minkt=0.88 maxkt=1.2
Tr#31#2031 bus31 bus2031 tap31 powertransformer x=0.10*BETAT kt=1.00 prating=200.0M  vrating=20k a=1.01*220/20 minkt=0.88 maxkt=1.2
Tr#32#2032 bus32 bus2032 tap32 powertransformer x=0.10*BETAT kt=1.00 prating=400.0M  vrating=20k a=1.06*220/20 minkt=0.88 maxkt=1.2
Tr#41#4041 bus41 bus4041 tap41 powertransformer x=0.10*BETAT kt=1.00 prating=1080.0M vrating=20k a=1.04*400/20 minkt=0.88 maxkt=1.2
Tr#42#4042 bus42 bus4042 tap42 powertransformer x=0.10*BETAT kt=1.00 prating=800.0M  vrating=20k a=1.03*400/20 minkt=0.88 maxkt=1.2
Tr#43#4043 bus43 bus4043 tap43 powertransformer x=0.10*BETAT kt=1.00 prating=1800.0M vrating=20k a=1.02*400/20 minkt=0.88 maxkt=1.2
Tr#46#4046 bus46 bus4046 tap46 powertransformer x=0.10*BETAT kt=1.00 prating=1400.0M vrating=20k a=1.02*400/20 minkt=0.88 maxkt=1.2
Tr#47#4047 bus47 bus4047 tap47 powertransformer x=0.10*BETAT kt=1.00 prating=200.0M  vrating=20k a=1.04*400/20 minkt=0.88 maxkt=1.2
Tr#51#4051 bus51 bus4051 tap51 powertransformer x=0.10*BETAT kt=1.00 prating=1600.0M vrating=20k a=1.05*400/20 minkt=0.88 maxkt=1.2
Tr#61#4061 bus61 bus4061 tap61 powertransformer x=0.10*BETAT kt=1.00 prating=1000.0M vrating=20k a=1.03*400/20 minkt=0.88 maxkt=1.2
Tr#62#4062 bus62 bus4062 tap62 powertransformer x=0.10*BETAT kt=1.00 prating=600.0M  vrating=20k a=1.04*400/20 minkt=0.88 maxkt=1.2
Tr#63#4063 bus63 bus4063 tap63 powertransformer x=0.10*BETAT kt=1.00 prating=1180.0M vrating=20k a=1.03*400/20 minkt=0.88 maxkt=1.2
Tr#71#4071 bus71 bus4071 tap71 powertransformer x=0.10*BETAT kt=1.00 prating=600.0M  vrating=20k a=1.03*400/20 minkt=0.88 maxkt=1.2
Tr#72#4072 bus72 bus4072 tap72 powertransformer x=0.10*BETAT kt=1.00 prating=4000.0M vrating=20k a=1.05*400/20 minkt=0.88 maxkt=1.2

; +--------+
; | Shunts |
; +--------+
Sh#1022 bus1022 powershunt b=  50M/130k^2 utype=1
Sh#1041 bus1041 powershunt b= 250M/130k^2 utype=1
Sh#1043 bus1043 powershunt b= 200M/130k^2 utype=1
Sh#1044 bus1044 powershunt b= 200M/130k^2 utype=1
Sh#1045 bus1045 powershunt b= 200M/130k^2 utype=1
Sh#4012 bus4012 powershunt b=-100M/400k^2 utype=1
Sh#4041 bus4041 powershunt b= 200M/400k^2 utype=1
Sh#4043 bus4043 powershunt b= 200M/400k^2 utype=1
Sh#4046 bus4046 powershunt b= 100M/400k^2 utype=1
Sh#4051 bus4051 powershunt b= 100M/400k^2 utype=1
Sh#4071 bus4071 powershunt b=-400M/400k^2 utype=1

; +-------+
; | Loads |
; +-------+
Lo#1  bus1  powerload vrating=20k prating=1M pc=600  qc=148.2 \
                      ap=1 aq=2
Lo#2  bus2  powerload vrating=20k prating=1M pc=330  qc=71.0  \
                      ap=1 aq=2
Lo#3  bus3  powerload vrating=20k prating=1M pc=260  qc=83.8  \
                      ap=1 aq=2
Lo#4  bus4  powerload vrating=20k prating=1M pc=840  qc=252.0 \
                      ap=1 aq=2
Lo#5  bus5  powerload vrating=20k prating=1M pc=720  qc=190.4 \
                      ap=1 aq=2
Lo#11 bus11 powerload vrating=20k prating=1M pc=200  qc=68.8  \
                      ap=1 aq=2
Lo#12 bus12 powerload vrating=20k prating=1M pc=300  qc=83.8  \
                      ap=1 aq=2
Lo#13 bus13 powerload vrating=20k prating=1M pc=100  qc=34.4  \
                      ap=1 aq=2
Lo#22 bus22 powerload vrating=20k prating=1M pc=280  qc=79.9  \
                      ap=1 aq=2
Lo#31 bus31 powerload vrating=20k prating=1M pc=100  qc=24.7  \
                      ap=1 aq=2
Lo#32 bus32 powerload vrating=20k prating=1M pc=200  qc=39.6  \
                      ap=1 aq=2
Lo#41 bus41 powerload vrating=20k prating=1M pc=540  qc=131.4 \
                      ap=1 aq=2
Lo#42 bus42 powerload vrating=20k prating=1M pc=400  qc=127.4 \
                      ap=1 aq=2
Lo#43 bus43 powerload vrating=20k prating=1M pc=900  qc=254.6 \
                      ap=1 aq=2
Lo#46 bus46 powerload vrating=20k prating=1M pc=700  qc=211.8 \
                      ap=1 aq=2
Lo#47 bus47 powerload vrating=20k prating=1M pc=100  qc=44.0  \
                      ap=1 aq=2
Lo#51 bus51 powerload vrating=20k prating=1M pc=800  qc=258.2 \
                      ap=1 aq=2
Lo#61 bus61 powerload vrating=20k prating=1M pc=500  qc=122.5 \
                      ap=1 aq=2
Lo#62 bus62 powerload vrating=20k prating=1M pc=300  qc=83.8  \
                      ap=1 aq=2
Lo#63 bus63 powerload vrating=20k prating=1M pc=590  qc=264.6 \
                      ap=1 aq=2
Lo#71 bus71 powerload vrating=20k prating=1M pc=300  qc=83.8  \
                      ap=1 aq=2
Lo#72 bus72 powerload vrating=20k prating=1M pc=2000 qc=396.1 \
                      ap=1 aq=2


; +--------+
; | Busses |
; +--------+
B#g1   g1 powerbus vb=15k v0=1.0684 theta0= 2.59
B#g2   g2 powerbus vb=15k v0=1.0565 theta0= 5.12
B#g3   g3 powerbus vb=15k v0=1.0595 theta0= 10.27
B#g4   g4 powerbus vb=15k v0=1.0339 theta0= 8.03
B#g5   g5 powerbus vb=15k v0=1.0294 theta0=-12.36
B#g6   g6 powerbus vb=15k v0=1.0084 theta0=-59.42
B#g7   g7 powerbus vb=15k v0=1.0141 theta0=-68.95
B#g8   g8 powerbus vb=15k v0=1.0498 theta0=-16.81
B#g9   g9 powerbus vb=15k v0=0.9988 theta0=-1.63
B#g10 g10 powerbus vb=15k v0=1.0157 theta0= 0.99
B#g11 g11 powerbus vb=15k v0=1.0211 theta0=-29.04
B#g12 g12 powerbus vb=15k v0=1.0200 theta0=-31.88
B#g13 g13 powerbus vb=15k v0=1.0170 theta0=-54.30
B#g14 g14 powerbus vb=15k v0=1.0454 theta0=-49.90
B#g15 g15 powerbus vb=15k v0=1.0455 theta0=-52.19
B#g16 g16 powerbus vb=15k v0=1.0531 theta0=-64.10
B#g17 g17 powerbus vb=15k v0=1.0092 theta0=-46.85
B#g18 g18 powerbus vb=15k v0=1.0307 theta0=-43.32
B#g19 g19 powerbus vb=15k v0=1.0300 theta0= 0.03
B#g20 g20 powerbus vb=15k v0=1.0185 theta0= 0.00    

B#1011 bus1011 powerbus vb=130.0k v0=1.0618 theta0=-6.65
B#1012 bus1012 powerbus vb=130.0k v0=1.0634 theta0=-3.10
B#1013 bus1013 powerbus vb=130.0k v0=1.0548 theta0= 1.26
B#1014 bus1014 powerbus vb=130.0k v0=1.0611 theta0= 4.26
B#1021 bus1021 powerbus vb=130.0k v0=1.0311 theta0= 2.64
B#1022 bus1022 powerbus vb=130.0k v0=1.0512 theta0=-19.05
B#1041 bus1041 powerbus vb=130.0k v0=1.0124 theta0=-81.87
B#1042 bus1042 powerbus vb=130.0k v0=1.0145 theta0=-67.38
B#1043 bus1043 powerbus vb=130.0k v0=1.0274 theta0=-76.77
B#1044 bus1044 powerbus vb=130.0k v0=1.0066 theta0=-67.71
B#1045 bus1045 powerbus vb=130.0k v0=1.0111 theta0=-71.66
B#2031 bus2031 powerbus vb=220.0k v0=1.0279 theta0=-36.66
B#2032 bus2032 powerbus vb=220.0k v0=1.0695 theta0=-23.92
B#4011 bus4011 powerbus vb=400.0k v0=1.0224 theta0=-7.55
B#4012 bus4012 powerbus vb=400.0k v0=1.0235 theta0=-5.54
B#4021 bus4021 powerbus vb=400.0k v0=1.0488 theta0=-36.08
B#4022 bus4022 powerbus vb=400.0k v0=0.9947 theta0=-20.86
B#4031 bus4031 powerbus vb=400.0k v0=1.0367 theta0=-39.46
B#4032 bus4032 powerbus vb=400.0k v0=1.0487 theta0=-44.54
B#4041 bus4041 powerbus vb=400.0k v0=1.0506 theta0=-54.30
B#4042 bus4042 powerbus vb=400.0k v0=1.0428 theta0=-57.37
B#4043 bus4043 powerbus vb=400.0k v0=1.0370 theta0=-63.51
B#4044 bus4044 powerbus vb=400.0k v0=1.0395 theta0=-64.23
B#4045 bus4045 powerbus vb=400.0k v0=1.0533 theta0=-68.88
B#4046 bus4046 powerbus vb=400.0k v0=1.0357 theta0=-64.11
B#4047 bus4047 powerbus vb=400.0k v0=1.0590 theta0=-59.55
B#4051 bus4051 powerbus vb=400.0k v0=1.0659 theta0=-71.01
B#4061 bus4061 powerbus vb=400.0k v0=1.0387 theta0=-57.93
B#4062 bus4062 powerbus vb=400.0k v0=1.0560 theta0=-54.36
B#4063 bus4063 powerbus vb=400.0k v0=1.0536 theta0=-50.68
B#4071 bus4071 powerbus vb=400.0k v0=1.0484 theta0=-4.99
B#4072 bus4072 powerbus vb=400.0k v0=1.0590 theta0=-3.98

B#1  bus1  powerbus vb=20k v0=0.9988 theta0=-84.71
B#2  bus2  powerbus vb=20k v0=1.0012 theta0=-70.49
B#3  bus3  powerbus vb=20k v0=0.9974 theta0=-79.97
B#4  bus4  powerbus vb=20k v0=0.9996 theta0=-70.67
B#5  bus5  powerbus vb=20k v0=0.9961 theta0=-74.59
B#11 bus11 powerbus vb=20k v0=1.0026 theta0=-9.45
B#12 bus12 powerbus vb=20k v0=0.9975 theta0=-5.93
B#13 bus13 powerbus vb=20k v0=0.9957 theta0=-1.58
B#22 bus22 powerbus vb=20k v0=0.9952 theta0=-21.89
B#31 bus31 powerbus vb=20k v0=1.0042 theta0=-39.47
B#32 bus32 powerbus vb=20k v0=0.9978 theta0=-26.77
B#41 bus41 powerbus vb=20k v0=0.9967 theta0=-57.14
B#42 bus42 powerbus vb=20k v0=0.9952 theta0=-60.22
B#43 bus43 powerbus vb=20k v0=1.0013 theta0=-66.33
B#46 bus46 powerbus vb=20k v0=0.9990 theta0=-66.93
B#47 bus47 powerbus vb=20k v0=0.9950 theta0=-62.38
B#51 bus51 powerbus vb=20k v0=0.9978 theta0=-73.84
B#61 bus61 powerbus vb=20k v0=0.9949 theta0=-60.78
B#62 bus62 powerbus vb=20k v0=1.0002 theta0=-57.18
B#63 bus63 powerbus vb=20k v0=0.9992 theta0=-53.49
B#71 bus71 powerbus vb=20k v0=1.0028 theta0=-7.80
B#72 bus72 powerbus vb=20k v0=0.9974 theta0=-6.83

; +----------------+
; | AVR converters |
; +----------------+

Pec1    g1   d1  gnd   q1  gnd  powerec type=0  
Pec2    g2   d2  gnd   q2  gnd  powerec type=0  
Pec3    g3   d3  gnd   q3  gnd  powerec type=0  
Pec4    g4   d4  gnd   q4  gnd  powerec type=0  
Pec5    g5   d5  gnd   q5  gnd  powerec type=0  

Pec6    g6   d6  gnd   q6  gnd  powerec type=0  
Pec7    g7   d7  gnd   q7  gnd  powerec type=0  

Pec8    g8   d8  gnd   q8  gnd  powerec type=0  
Pec9    g9   d9  gnd   q9  gnd  powerec type=0  
Pec10  g10  d10  gnd  q10  gnd  powerec type=0  
Pec11  g11  d11  gnd  q11  gnd  powerec type=0  
Pec12  g12  d12  gnd  q12  gnd  powerec type=0  

Pec13  g13  d13  gnd  q13  gnd  powerec type=0  
Pec14  g14  d14  gnd  q14  gnd  powerec type=0  
Pec15  g15  d15  gnd  q15  gnd  powerec type=0  
Pec16  g16  d16  gnd  q16  gnd  powerec type=0  
Pec17  g17  d17  gnd  q17  gnd  powerec type=0  
Pec18  g18  d18  gnd  q18  gnd  powerec type=0  

Pec19  g19  d19  gnd  q19  gnd  powerec type=0  
Pec20  g20  d20  gnd  q20  gnd  powerec type=0  

; +-------------------------+
; | Hydro turbine governors |
; +-------------------------+

Htg1    omega1  pm1  HTG
Htg2    omega2  pm2  HTG
Htg3    omega3  pm3  HTG
Htg4    omega4  pm4  HTG
Htg5    omega5  pm5  HTG

Htg8    omega8  pm8  HTG
Htg9    omega9  pm9  HTG
Htg10  omega10 pm10  HTG
Htg11  omega11 pm11  HTG
Htg12  omega12 pm12  HTG

Htg19  omega19 pm19  HTG
Htg20  omega20 pm20  HTG

model HTG   nport veriloga="htg.va" verilogaprotected=no \
                  verilogatrace=["OmegaErr","PiOut","Reset"]

; +------------------------------+
; | Automatic voltage regulators |
; +------------------------------+

Avr1   d1  q1  avr1 AVR G=70  TA=10 TB=20 
Avr2   d2  q2  avr2 AVR G=70  TA=10 TB=20 
Avr3   d3  q3  avr3 AVR G=70  TA=10 TB=20 
Avr4   d4  q4  avr4 AVR G=70  TA=10 TB=20 
Avr5   d5  q5  avr5 AVR G=70  TA=10 TB=20 

Avr6   d6  q6  avr6 AVR G=120 TA=5  TB=12.5 
Avr7   d7  q7  avr7 AVR G=120 TA=5  TB=12.5 

Avr8   d8  q8  avr8 AVR G=70  TA=10 TB=20 
Avr9   d9  q9  avr9 AVR G=70  TA=10 TB=20 
Avr10 d10 q10 avr10 AVR G=70  TA=10 TB=20 
Avr11 d11 q11 avr11 AVR G=70  TA=10 TB=20 
Avr12 d12 q12 avr12 AVR G=70  TA=10 TB=20 

Avr13 d13 q13 avr13 AVR G=50  TA=4  TB=20 
Avr14 d14 q14 avr14 AVR G=120 TA=5  TB=12.5 
Avr15 d15 q15 avr15 AVR G=120 TA=5  TB=12.5 
Avr16 d16 q16 avr16 AVR G=120 TA=5  TB=12.5 
Avr17 d17 q17 avr17 AVR G=120 TA=5  TB=12.5 
Avr18 d18 q18 avr18 AVR G=120 TA=5  TB=12.5 

Avr19 d19 q19 avr19 AVR G=70  TA=10 TB=20 
Avr20 d20 q20 avr20 AVR G=70  TA=10 TB=20 

model AVR   nport veriloga="avr.va" verilogaprotected=yes

; +--------------+
; | Tap changers |
; +--------------+

Uec1    bus1   ud1  gnd   uq1  gnd  powerec type=0  
Uec2    bus2   ud2  gnd   uq2  gnd  powerec type=0  
Uec3    bus3   ud3  gnd   uq3  gnd  powerec type=0  
Uec4    bus4   ud4  gnd   uq4  gnd  powerec type=0  
Uec5    bus5   ud5  gnd   uq5  gnd  powerec type=0  
Uec11  bus11  ud11  gnd  uq11  gnd  powerec type=0  
Uec12  bus12  ud12  gnd  uq12  gnd  powerec type=0  
Uec13  bus13  ud13  gnd  uq13  gnd  powerec type=0  
Uec22  bus22  ud22  gnd  uq22  gnd  powerec type=0  
Uec31  bus31  ud31  gnd  uq31  gnd  powerec type=0  
Uec32  bus32  ud32  gnd  uq32  gnd  powerec type=0  

Uec41  bus41  ud41  gnd  uq41  gnd  powerec type=0
Uec42  bus42  ud42  gnd  uq42  gnd  powerec type=0
Uec43  bus43  ud43  gnd  uq43  gnd  powerec type=0
Uec46  bus46  ud46  gnd  uq46  gnd  powerec type=0
Uec47  bus47  ud47  gnd  uq47  gnd  powerec type=0
Uec51  bus51  ud51  gnd  uq51  gnd  powerec type=0
Uec61  bus61  ud61  gnd  uq61  gnd  powerec type=0
Uec62  bus62  ud62  gnd  uq62  gnd  powerec type=0
Uec63  bus63  ud63  gnd  uq63  gnd  powerec type=0
Uec71  bus71  ud71  gnd  uq71  gnd  powerec type=0
Uec72  bus72  ud72  gnd  uq72  gnd  powerec type=0
 
U1   ud1  uq1  tap1  ULTC TW=20+10*rand()
U2   ud2  uq2  tap2  ULTC TW=20+10*rand()
U3   ud3  uq3  tap3  ULTC TW=20+10*rand()
U4   ud4  uq4  tap4  ULTC TW=20+10*rand()
U5   ud5  uq5  tap5  ULTC TW=20+10*rand()
U11 ud11 uq11 tap11  ULTC TW=20+10*rand()
U12 ud12 uq12 tap12  ULTC TW=20+10*rand()
U13 ud13 uq13 tap13  ULTC TW=20+10*rand()
U22 ud22 uq22 tap22  ULTC TW=20+10*rand()
U31 ud31 uq31 tap31  ULTC TW=20+10*rand()
U32 ud32 uq32 tap32  ULTC TW=20+10*rand()
U41 ud41 uq41 tap41  ULTC TW=20+10*rand()
U42 ud42 uq42 tap42  ULTC TW=20+10*rand()
U43 ud43 uq43 tap43  ULTC TW=20+10*rand()
U46 ud46 uq46 tap46  ULTC TW=20+10*rand()
U47 ud47 uq47 tap47  ULTC TW=20+10*rand()
U51 ud51 uq51 tap51  ULTC TW=20+10*rand()
U61 ud61 uq61 tap61  ULTC TW=20+10*rand()
U62 ud62 uq62 tap62  ULTC TW=20+10*rand()
U63 ud63 uq63 tap63  ULTC TW=20+10*rand()
U71 ud71 uq71 tap71  ULTC TW=20+10*rand()
U72 ud72 uq72 tap72  ULTC TW=20+10*rand()

model ULTC   nport veriloga="ultc.va" verilogaprotected=no \
                   verilogatrace=["Vmag","Vref","Upper","Lower"]

; +----------+
; | Couplers |
; +----------+

#ifdef HVDC

parameters Vd#4011=438.96k Vq#4011=-24.982k \
           Vabs#4011=sqrt(Vd#4011^2+Vq#4011^2) \
	   Vph#4011=atan2(Vq#4011,Vd#4011)/pi*180

Ec#4011 bus4011 tas tbs tcs powerec type=3 \
        f0=F0 vrating=Vabs#4011 v0=1 theta0=Vph#4011 // HVDC Sender

parameters Vd#4045=458.54k Vq#4045=-212.99k \
           Vabs#4045=sqrt(Vd#4045^2+Vq#4045^2) \
           Vph#4045=atan2(Vq#4045,Vd#4045)/pi*180

Ec#4045 bus4045 tar tbr tcr powerec type=3 \
        f0=F0 vrating=Vabs#4045 v0=1 theta0=Vph#4045 // HVDC Receiver

#endif

end

#ifdef HVDC

; +--------------+
; | HVDC Circuit |
; +--------------+

; +--------+
; | Sender |
; +--------+

; Transformer

Tas     tas      gnd   as  gnd  transformer t1=K t2=1
Tbs     tcs      gnd   bs  gnd  transformer t1=K t2=1
Tcs     tbs      gnd   cs  gnd  transformer t1=K t2=1

; VSC converter and LCL filter instance

LCLs   as   bs   cs   dcps  dcns  vds  vqs  idrefs  iqrefs  LCL_VSC KI=G_KI KP=G_KP

; Set-point sender

Xdc   xdc     gnd   dcps  dcns  vcvs gain1=1
Vdc   vdc     gnd   vsource vdc=VSC_DC

Xx    idc     gnd   ccvs sensedev="Lp" gain1=1
PiVdc xdc     vdc   dcout  PI  KP=10 KI=100
Xp      p     gnd   dcout  idc iqrefs vcvs func=v(dcout)*v(idc)

Idrefs  idrefs   gnd   vds  vqs  p   vcvs func= \
    (v(p)*v(vds)+Q*v(vqs))/((3/2)*max(10,(v(vds)*v(vds)+v(vqs)*v(vqs))))

Iqrefs  iqrefs   gnd   vds  vqs  p   vcvs func= \
    (v(p)*v(vqs)-Q*v(vds))/((3/2)*max(10,(v(vds)*v(vds)+v(vqs)*v(vqs))))

; +----------+
; | Receiver |
; +----------+

; Transformer 

Tar     tar      gnd   ar  gnd  transformer t1=K t2=1
Tbr     tcr      gnd   br  gnd  transformer t1=K t2=1
Tcr     tbr      gnd   cr  gnd  transformer t1=K t2=1

; VSC converter and LCL filter instance

LCLr   ar   br   cr   dcpr  dcnr  vdr  vqr  idrefr  iqrefr  LCL_VSC KI=G_KI KP=G_KP

; Set-point receiver
Idrefr  idrefr   gnd   vdr  vqr   vcvs func= \
    (P*v(vdr)+Q*v(vqr))/((3/2)*max(10,(v(vdr)*v(vdr)+v(vqr)*v(vqr))))

Iqrefr  iqrefr   gnd   vdr  vqr   vcvs func= \
    (P*v(vqr)-Q*v(vdr))/((3/2)*max(10,(v(vdr)*v(vdr)+v(vqr)*v(vqr))))

; +---------+
; | DC Link |
; +---------+

parameters R_KM=0.42m L_KM=15.9u C_KM=23.1u KM=700

Cdcpr   dcpr   gnd  capacitor c=100u*1 ic=VSC_DC/2
Cdcnr    gnd  dcnr  capacitor c=100u*1 ic=VSC_DC/2
Rdcpr   dcpr   gnd  resistor  r=1G
Rdcnr   dcnr   gnd  resistor  r=1G

Rp      dcpr    xp  resistor  r=R_KM*KM
Lp        xp  dcps  inductor  l=L_KM*KM

Rn      dcnr    xn  resistor  r=R_KM*KM
Ln        xn  dcns  inductor  l=L_KM*KM

Cdcps   dcps   gnd  capacitor c=100u*1 ic=VSC_DC/2
Cdcns    gnd  dcns  capacitor c=100u*1 ic=VSC_DC/2
Rdcps   dcps   gnd  resistor  r=1G
Rdcns   dcns   gnd  resistor  r=1G

; +----------------------+
; | Subcircuits & Models |
; +----------------------+

; VSC converter and LCL filter subcircuit

subckt LCL_VSC a b c dcp dcn vd vq idref iqref
parameters KI=50 KP=20

// VSC instance

VSC1        vha  vhb  vhc  ya  yb  yc  dcp  dcn  VSC 

// Breaker

Swa   vha   xha   swc   gnd  VSW
Swb   vhb   xhb   swc   gnd  VSW
Swc   vhc   xhc   swc   gnd  VSW

Vsw   swc   gnd   vsource v1=0 v2=V_DIG td=3/F0 tr=1u \
                          width=25*TSTOP+1m period=30*TSTOP
// LCL Filter

RLS1 xha xhb xhc xa xb xc iha ihb ihc RL_CURRENT_SENSE_NO_CUTSET  R=R1 L=L1
F1    xa xb xc       FILTER
RL1   xa xb xc a b c RL      R=R1 L=L1

// abctodq0 
I2     xa     xb    xc   theta  vd  vq  vzero  abctodq0  
I3    iha    ihb   ihc   theta  id  iq  izero  abctodq0  

// PLL
I4     vq  theta  phase  THETA FREQ=F0

// Controller
CNTR1  idref  iqref  theta  vd  vq  id  iq  ya  yb  yc  VSC_CONTROLLER KI=KI KP=KP

ends

; VSC controller block

subckt VSC_CONTROLLER idref iqref theta vd vq id iq a b c
parameters KI=50 KP=20

P1     id   idref    pi1  PI KP=KP KI=KI
Id      d     gnd    vd   gnd  pi1  gnd  iq  gnd  vcvs gain1=1 \
                                                       gain2=1 \
                                                       gain3=-F0*2*pi*L1

P2     iq   iqref    pi2  PI KP=G_KP KI=G_KI
Iq      q     gnd    vq   gnd  pi2  gnd  id  gnd  vcvs gain1=1 \
                                                       gain2=1 \
                                                       gain3= F0*2*pi*L1

Ic      d    q    gnd   theta   a  b  c  dq0toabcWithGain GAIN=2/VSC_DC
ends

; Voltage Source Converter block

subckt VSC  pha phb phc refa refb refc dcp dcn

#ifdef SWC
; Pwm modulator
Vpwma    pwma     gnd   vsource t=0 v=0 \
                                t=PWM_PERIOD/4     v= 1 \
				t=PWM_PERIOD/4*3   v=-1 \
				t=PWM_PERIOD       v= 0 \
				period=PWM_PERIOD

Cmpa     cmpa     gnd   pwma  refa vcvs func=(v(refa,pwma) > 0) ? V_DIG : 0 \
                                        digital=yes trtime=TR_TIME 

Cmpb     cmpb     gnd   pwma  refb vcvs func=(v(refb,pwma) > 0) ? V_DIG : 0 \
                                        digital=yes trtime=TR_TIME

Cmpc     cmpc     gnd   pwma  refc vcvs func=(v(refc,pwma) > 0) ? V_DIG : 0 \
                                        digital=yes trtime=TR_TIME

Va        pha     phb   cmpa  cmpb dcp dcn  vcvs func=v(cmpa,cmpb)*v(dcp,dcn)
Vb        phb     phc   cmpb  cmpc dcp dcn  vcvs func=v(cmpb,cmpc)*v(dcp,dcn)

#else 

; Pwm modulator
Va        pha     phb   refa  refb  dcp  dcn  vcvs func=v(refa,refb)*v(dcp,dcn)/2
Vb        phb     phc   refb  refc  dcp  dcn  vcvs func=v(refb,refc)*v(dcp,dcn)/2

#endif

Xdc       dcp     dcn   nport sensedev="Va" sensedev="Vb" \
                              sensen="pha" sensen="phb" sensen="phc" \
    func1=max(VSC_DC/2,v(p1))*i(p1)+i(Va)*v(pha,phb)+i(Vb)*v(phb,phc)

ends

; dq0toabcWithGain block

subckt dq0toabcWithGain d q zero wt a b c
parameters GAIN=1/450 MIN=-0.95 MAX=0.95

Mt1   a    gnd    wt  d  q  zero  vcvs  func=GAIN*(-sin(v(wt))*v(q) + \
                                                    cos(v(wt))*v(d) + \
                                                    v(zero)) min=MIN max=MAX
Mt2   b    gnd    wt  d  q  zero  vcvs  func=GAIN*(-sin(v(wt)-2*pi/3)*v(q) + \
                                                    cos(v(wt)-2*pi/3)*v(d) + \
                                                    v(zero)) min=MIN max=MAX
Mt3   c    gnd    wt  d  q  zero  vcvs  func=GAIN*(-sin(v(wt)+2*pi/3)*v(q) + \
                                                    cos(v(wt)+2*pi/3)*v(d) + \
                                                    v(zero)) min=MIN max=MAX
ends

; abctodq0 block

subckt abctodq0  a b c wt d q zero
Xd     d    gnd    wt  a  b  c  vcvs  func= 2/3 * \
                        (cos(v(wt))       *v(a) +   \
                         cos(v(wt)-2/3*pi)*v(b) +   \
                         cos(v(wt)+2/3*pi)*v(c))

Xq     q    gnd    wt  a  b  c  vcvs  func= -2/3 * \
                        (sin(v(wt))       *v(a) +   \
                         sin(v(wt)-2/3*pi)*v(b) +   \
                         sin(v(wt)+2/3*pi)*v(c))

Xzero zero  gnd    a  b  c  vcvs  func=sqrt(2)/3*(v(a)+v(b)+v(c))
ends

subckt PI inneg inpos out
parameters KP=20 KI=50
C1    c   gnd   capacitor c=1*1m
X1    c   gnd   inpos    inneg            vccs  gain1=-KI*1m
X2  out   gnd   inpos    inneg   c   gnd  vcvs  gain1= KP gain2=1

ends

subckt RL ia ib ic oa ob oc
parameters L=1.8m R=100m

La   ia   xa    inductor l=L
Ra   xa   oa    resistor r=R
Rda  ia   xa    resistor r=100e6

Lb   ib   xb    inductor l=L
Rb   xb   ob    resistor r=R
Rdb  ib   xb    resistor r=100e6

Lc   ic   xc    inductor l=L
Rc   xc   oc    resistor r=R
Rdc  ic   xc    resistor r=100e6

ends

subckt RL_CURRENT_SENSE_NO_CUTSET ia ib ic oa ob oc sa sb sc
parameters L=1.8m R=100m

La    ia   xa    inductor l=L
Ra    xa   oa    resistor r=R

Lb    ib   xb    inductor l=L
Rb    xb   ob    resistor r=R

Rc    xc   oc    resistor r=R

#ifdef STAR
Lc    ic   xc    inductor l=L

Ic    sc  gnd   ccvs sensedev="Lc" gain1=1
#else
Lc    ic   xc    ia   xa   ib   xb  vcvs gain1=-1 gain2=-1

Ic    sc  gnd   ccvs sensedev="La" sensedev="Lb" gain1=-1 gain2=-1
#endif

Ia    sa  gnd   ccvs sensedev="La" gain1=1
Ib    sb  gnd   ccvs sensedev="Lb" gain1=1

ends

subckt FILTER a b c 
parameters R=0.1 C=27u

Ra     a   ca   resistor  r=R
Ca    ca  gnd   capacitor c=C

Rb     b   cb   resistor  r=R
Cb    cb  gnd   capacitor c=C

Rc     c   cc   resistor  r=R
Cc    cc  gnd   capacitor c=C

ends

model THETA nport veriloga="theta.va" verilogaprotected=no
model VSW vswitch ron=10m roff=10M voff=0.4*V_DIG von=0.6*V_DIG

#endif
