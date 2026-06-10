# MPC baseline: acados SQP-RTI + HPIPM on the quarter-car environment
#
# ocp.py          CasADi symbolic quarter-car ODE + acados OCP/solver builder
# controller.py   MPCController: caches solver per road geometry, runs SQP-RTI
# mpc.py          CLI runner: episodes, comparison table, JSON output
#
# usage:
#   python src/baseline/mpc/mpc.py
#   python src/baseline/mpc/mpc.py --n-episodes 50 --horizon 60
#   python src/baseline/mpc/mpc.py --out eval/results/mpc_h50.json
