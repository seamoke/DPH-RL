# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import re
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))
import threading
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
import argparse
from abc import ABC, abstractmethod
from loguru import logger


extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.path.pardir, os.path.pardir))
sys.path.append(extra_path)

class Scorer(ABC):
    """
    preprocess question and response then return score
    """

    def __init__(self, ** kargs) -> None:
        super().__init__()

    @abstractmethod
    def score(self, trajs, **kargs):
        pass

    @abstractmethod
    def close(self):
        pass


class SqlRewardModelBase(Scorer):
    """
    preprocess question and response then return score
    """
    KEY = 'sql_execution_reward'

    def __init__(self, database_path, **kwargs) -> None:
        self.database_path = database_path
        SqlRewardModelBase.KEY = kwargs.get('key', 'sql_execution_reward')
        
        try:
            global extra_path
            print(f"extra_path: {extra_path}")
            sys.path.append(extra_path)
            from rl.scorer.sql_executor import Excecutor as SQLExcecutor
        except ImportError:
            SQLExcecutor = None
            print("== [WARN] Can not import SQLExecutor.")
        assert SQLExcecutor is not None, "SQLExcecutor is not initialized."
        self.excecutor = SQLExcecutor()
        self.eq_func = SQLExcecutor.result_equal

    async def score(self, trajs):
        queries = [response for response in trajs['gen_responses']]
        
        gold_sql = trajs['extra']['gold_sql']
        db_id = os.path.join(self.database_path, trajs['extra']['db_id'])
        
        cmp_method = trajs['extra'].get('cmp_method', 'spider')
        gold_answer = self.excecutor.exec_sql(db_id, [gold_sql],
                                               keep_distinct=True,
                                               do_post_process=False)
        if gold_answer[0][0] != 'result':
            print(f"db_id:{db_id} gold_sql: {gold_sql} is not executable. result:{gold_answer}")
            return [0.0 for _ in queries]
        exec_res = []
        for query in queries:
            res = self.excecutor.exec_sql(db_id, [query], keep_distinct=True)
            #print(f"db_id:{db_id}\nquery: {query}\nres: {res}\ngold_sql: {gold_sql}\ngold_answer: {gold_answer}", flush=True)
            exec_res.append(res)
        
        comp_res = []
        for res in exec_res:
            if res[0][0] != 'result':
                comp_res.append(False)
                print(f"execution error:{res}", flush=True)
            else:
                if cmp_method == "spider":
                    order_matters = "orderby" in re.sub(r"\s+", gold_sql.lower(), "")
                    comp = self.eq_func(res[0][-1], gold_answer[0][-1], order_matters)
                    comp_res.append(comp)
                else:
                    comp = set(res[0][-1]) == set(gold_answer[0][-1])
                    comp_res.append(comp)
        comp_res = [1.0 if res else 0.0 for res in comp_res]
        assert len(comp_res) == len(queries)
        return list(zip(comp_res, exec_res))
    
    def close(self):
        return super().close()
    
class SqlRewardModel(SqlRewardModelBase):
    # Inherits score and __init__ from the base class.
    pass

class SqlDetailRewardModel(SqlRewardModelBase):
    # This model provides finer-grained scoring for SQL tasks.
    # It matches sub-columns between prediction (pred) and ground truth (gold).
    # Example: pred = [column1, column2, column3], gold = [column1, column2],
    # common(pred, gold) = {column1, column2}.
    # Score formula:
    #     score = 0.8 * len(common(pred, gold)) / (len(pred) * len(gold))
    # Note: Column order is not considered in matching.
    # The factor 0.8 is used to distinguish partially correct preds from fully correct ones.
    async def score(self, trajs):
        queries = [response for response in trajs['gen_responses']]
        
        gold_sql =trajs['extra']['gold_sql']  
        db_id = os.path.join(self.database_path, trajs['extra']['db_id'])  
        
        cmp_method = trajs['extra'].get('cmp_method','bird')
        gold_answer = self.excecutor.exec_sql(db_id, [gold_sql],keep_distinct=True,do_post_process=False)
        if gold_answer[0][0]!='result':
            print(f"db_id:{db_id} gold_sql: {gold_sql} is not executable. result:{gold_answer} ")
            return [0.0 for _ in queries]

        exec_res = []
        total_gold_list = []

        for query in  queries:
            
            res = self.excecutor.exec_sql(db_id, [query],keep_distinct=True)
            exec_res.append(res)
            total_gold_list.append(gold_answer)
        comp_res = []
        for res in exec_res:
            if res[0][0]!='result':
                comp_res.append(0.0)
                print( f"execution error:{res}", flush=True)
            else:

                if cmp_method == "spider":
                    order_matters = "orderby" in re.sub(r"\s+", gold_sql.lower(), "")
                    comp = self.eq_func(res[0][-1], gold_answer[0][-1], order_matters)
                    
                else:
                    comp = set(res[0][-1] ) == set(gold_answer[0][-1])
                if comp:
                    comp = 1.0
                else:
                    comp = 0.0
                if comp == 0.0:
                    gold_list = copy.deepcopy(gold_answer[0][-1])
                    res_list = copy.deepcopy(res[0][-1])
                    gold_list = list(set(gold_list)) 
                    res_list = list(set(res_list))

                    indices = list(range(len(gold_list[0]))) if gold_list else []
                    gold_dict = {i: {t[i] for t in gold_list} for i in indices}

                    indices = list(range(len(res_list[0]))) if res_list else []
                    res_dict = {i: {t[i] for t in res_list} for i in indices}


                    def A_in_B(gold_dict, res_dict):
                        """
                        For each set in gold_dict, check whether there exists at least one set in res_dict that contains all its elements.
                        This function is reserved for future use.
                        """
                        for gold_set in gold_dict.values():
                            if not any(gold_set==res_set for res_set in res_dict.values()):
                                return False
                        return True
                    def get_product(gold_dict, res_dict):
                        """
                        Compute the Cartesian product of gold and res.
                        """
                        count = 0
                        if (len(gold_dict) * len(res_dict)) == 0:
                            return 0.0
                        for gold_set in gold_dict.values():
                            if any(gold_set==res_set for res_set in res_dict.values()):
                                count += 1
                        score = count * count / (len(gold_dict) * len(res_dict))
                        score = min(score,1.0) * 0.8
                        return score
                    comp = get_product(gold_dict,res_dict)

                comp_res.append(comp)   
       
        assert len(comp_res) == len(queries)
                                  
        # Remove the comment from this line to return gold.
        #return list(zip(comp_res,exec_res,total_gold_list))
        return list(zip(comp_res, exec_res))

    def close(self):
        return super().close()
    

app = FastAPI()
exiting = False


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.opt(exception=e).error('Error in custom_exception_handler')


class ScorerServer(object):
    def __init__(self, args=None):
        self.chain_map = {}

        assert 'Chains' in args, f"config must contain Chains field."
        assert 'Scorers' in args, f"config must contain Scorers field."

        chains_args = args['Chains']
        scorers_args = args['Scorers']
        logger.info(f"scorers_args: {scorers_args}")

        _scorer1_args = scorers_args['Scorer1']
        _scorer_name = _scorer1_args.pop("class_name")
        assert _scorer_name == SqlDetailRewardModel.__name__

        self.scorer = SqlDetailRewardModel(**_scorer1_args)
        self.logging = 'log_file' in args['ScorerServer']

    async def process_trajectories(self, trajs):
        results = {}

        # if "tag" not in trajs:
        #     assert len(self.chain_map) == 1, f"when no specify tag in data, only support one chain."
        #     results = await self.scorer.score(trajs)
        # else:
        results = await self.scorer.score(trajs)

        logging_results = {}
        logging_results['results'] = results
        logging_results['extra']=trajs.get('extra','')
        logging_results['type']=trajs.get('type','')
        logging_results['idx']=trajs.get('idx','')
        logging_results['round']=trajs.get('round','')

        if self.logging:
            logger.info(logging_results)
        _results = {}
        _results['Scorer1'] = results
        return _results


class ShutDownThread(threading.Thread):
    def __init__(self,server):
        threading.Thread.__init__(self)
        self.server = server
    def run(self):
        while True:
            # Call print_time
            # print_time()
            global exiting
            if exiting:
                for scorer_name, scorer_instances in self.server.scorers_map.items():
                    for score_key, scorer in scorer_instances:
                        print(f'close scorer {scorer_name} {score_key}', scorer, flush=True)
                        close_handle = scorer.close.remote()
                os._exit(0)
            time.sleep(10)

server_instance = None

@app.on_event("startup")
async def startup_event():
    global server_instance

    try:
        args = get_args()
        server_instance = ScorerServer(args)

        shutdown_thread = ShutDownThread(server_instance)
        shutdown_thread.start()
    except Exception as e:
        raise Exception(f"{e.__repr__()}")


@app.put("/api")
async def score_trajectories(request: Request):
    try:
        trajs = await request.json()
        results = await server_instance.process_trajectories(trajs)
        return JSONResponse(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print (f"Scorer Server Exception: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing trajectories: {str(e)}")

@app.get("/heartbeat")
async def heartbeat():
    return {"status": "healthy"}

@app.put("/shutdown")
async def shutdown():
    global exiting
    exiting = True
    return {"message": "Server shutting down..."}


# Define a function that prints the current time
def print_time():
    print(f"Current time: {datetime.now().isoformat()}",flush=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', "-c", type=str, default="/path/to/config.yaml", help='Config YAML')
    parser.add_argument('--address', type=str, default="0.0.0.0", help='Server address')
    parser.add_argument('--port', type=int, default=5001, help='Server port')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if config['ScorerServer'] == None:
        config['ScorerServer'] = {}
    if args.address:
        config['ScorerServer']['address'] = args.address
    if args.port:
        config['ScorerServer']['port'] = args.port

    config['ScorerServer'] = OmegaConf.merge(vars(args), config['ScorerServer'])

    return config


def set_logging(config):    
    if 'log_file' in config['ScorerServer']:
        logger.add(config['ScorerServer'].log_file, rotation=1*10e9,format="{time} | {level} | {message}")


if __name__ == "__main__":
    try:
        import uvicorn
        args = get_args()
        set_logging(args)
        address = args['ScorerServer'].address
        port = args['ScorerServer'].port

        num_instances = args["Scorers"]["Scorer1"]["num_instances"]
        uvicorn.run(app="scorer_server_without_ray:app", host=address, port=port, workers=num_instances)
    except Exception as e:
        raise Exception(f"{e.__repr__()}")
