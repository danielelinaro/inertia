ground electrical gnd

parameter PRAND=1M
parameter TSTOP=10
parameter FRAND=10
parameters UC=1000 UO=1000

options outintnodes=yes ; pivcaching=0

Al_dummy_tstop  alter param="TSTOP"  rt=yes
Al_dummy_frand  alter param="FRAND"  rt=yes

#ifdef PAN

Ctrl control begin

    dt = 1/FRAND;
    T  = [dt:dt:TSTOP+dt];

    noise_samples_bus_3  = [ T, randn( length(T) ) ];

endcontrol

Dc dc nettype=1 print=yes sparse=1

#ifndef LOAD_FLOW_ONLY

Pz pz nettype=1 mem=["invmtrx"]

#ifdef WITH_TRAN
Tr tran stop=TSTOP nettype=1 restart=1 annotate=3 method=1 timepoints=1/FRAND forcetps=1 maxiter=65 saman=yes sparse=2
#endif

#endif // LOAD_FLOW_ONLY

#endif // PAN

begin power

include ieee39_PF.inc

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

; the powerec used for connecting the stochastic load
Pec13 bus3 dload gnd qload gnd powerec type=0

; used to give a reference to the electric angular frequency of each bus
Coi powercoi type=2 gen="G02" \
                  attach="G01" attach="G03" attach="G04" \
                  attach="G05" attach="G06" attach="G07" \
                  attach="G08" attach="G09" attach="G10"

end

; the stochastic load
Rnd dload qload rndload RAND_L P=PRAND VRATING=345k VMAX=1.2*345k VMIN=0.8*345k
Wav rndload gnd vsource wave="noise_samples_bus_3"

model RAND_L   nport veriloga="randl.va"
; model IEEEG1Tg nport veriloga="ieeeg1tg.va"
; model IEEEG3Tg nport veriloga="ieeeg3tg.va"

