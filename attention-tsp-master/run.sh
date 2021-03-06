
python run.py --graph_size 50 --run_name f0_adap --experiment adaptive --adaptive_parameter tsp50_f0_adap.json --baseline rollout
python run.py --graph_size 50 --run_name f1_adap --experiment adaptive --adaptive_parameter tsp50_f1_adap.json --baseline rollout

python run.py --graph_size 50 --run_name f0_fixed --experiment supervised --supervised_parameter tsp50_f0_fixed.json --baseline rollout
python run.py --graph_size 50 --run_name f1_fixed --experiment supervised --supervised_parameter tsp50_f1_fixed.json --baseline rollout
