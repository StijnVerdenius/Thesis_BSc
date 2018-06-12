









python mailing.py started_on_tsp20_f0_f1_fixed
python run.py --graph_size 20 --run_name f0_f1_fixed --experiment supervised --supervised_parameter tsp20_f0_f1_fixed.json --baseline rollout
python mailing.py started_on_tsp20_f0_f1_adap
python run.py --graph_size 20 --run_name f0_f1_adap --experiment adaptive --adaptive_parameter tsp20_f0_f1_adap.json --baseline rollout
python mailing.py started_on_tijdsexp
python run.py --graph_size 50 --run_name time_experiment --experiment supervised --supervised_parameter time_experiment.json --baseline rollout
python mailing.py started_on_tsp50_f0_f1_fixed
python run.py --graph_size 50 --run_name f0_f1_fixed --experiment supervised --supervised_parameter tsp50_f0_f1_fixed.json --baseline rollout
python mailing.py started_on_tsp50_f0_f1_adap
python run.py --graph_size 50 --run_name f0_f1_adap --experiment adaptive --adaptive_parameter tsp50_f0_f1_adap.json --baseline rollout
python mailing.py started_on_tsp20_f1_fixed
python run.py --graph_size 20 --run_name f1_fixed --experiment supervised --supervised_parameter tsp20_f1_fixed.json --baseline rollout
python mailing.py started_on_tsp20_f0_adap
python run.py --graph_size 20 --run_name f0_adap --experiment adaptive --adaptive_parameter tsp20_f0_adap.json --baseline rollout
python mailing.py started_on_tsp20_f1_adap
python run.py --graph_size 20 --run_name f1_adap --experiment adaptive --adaptive_parameter tsp20_f1_adap.json --baseline rollout
python mailing.py finished_all