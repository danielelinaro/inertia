ground electrical gnd

options outintnodes=yes ; pivcaching=0

Dc dc nettype=1 print=yes sparse=1
;Pz pz nettype=1 mem=["invmtrx"]

begin power

include ieee39_PF.inc

end

