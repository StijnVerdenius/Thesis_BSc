python mailing.py 50base
python run.py --graph_size 50 --run_name baselinefor50 --experiment supervised --supervised_parameter begin_juni_1.json --baseline rollout 
python mailing.py 50sup
python run.py --graph_size 50 --run_name sup50 --experiment supervised --supervised_parameter begin_juni_2.json --baseline rollout 
python mailing.py 50ap
python run.py --graph_size 50 --run_name adap50 --experiment adaptive --adapative_parameter begin_juni_3.json --baseline rollout 
python mailing.py 20siz
python run.py --graph_size 20 --run_name valsize --experiment supervised --supervised_parameter begin_juni_4.json --baseline rollout 
python mailing.py 20entr
python run.py --graph_size 20 --run_name valentr --experiment supervised --supervised_parameter begin_juni_5.json --baseline rollout
python mailing.py done_now