import os
import argparse
import yaml
from easydict import EasyDict

import numpy as np

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from flask import Flask, request

from genrobo3d.evaluation.robot_pipeline import RobotPipeline

def preprocess_batch_format(batch):
    # batch = {
    #     'task_str': task_str,
    #     'variation': variation,
    #     'step_id': step_id,
    #     'obs_state_dict': obs_state_dict,
    #     'episode_id': demo_id,
    #     'instructions': instructions,
    #     'cache': cache,
    # }
    cache = batch.get("cache", None)
    if cache is not None and isinstance(cache, dict):
        batch["cache"] = EasyDict(cache)

    return batch


def main(args):
    app = Flask(__name__)
    
    with open('genrobo3d/configs/rlbench/robot_pipeline_realrobot.yaml', 'r') as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config = EasyDict(pipeline_config)

    mp_checkpoint_file = os.path.join(
        pipeline_config.motion_planner.expr_dir, 'ckpts', 
        f'model_step_{pipeline_config.motion_planner.ckpt_step}.pt'
    )
    if not os.path.exists(mp_checkpoint_file):
        print(mp_checkpoint_file, 'not exists')
        return
    pipeline_config.motion_planner.checkpoint = mp_checkpoint_file
    pipeline_config.motion_planner.config_file = os.path.join(
        pipeline_config.motion_planner.expr_dir, 'logs', 'training_config.yaml'
    )

    pipeline_config.motion_planner.pred_dir = os.path.join(pipeline_config.motion_planner.expr_dir, 'preds_debug')
    os.makedirs(pipeline_config.motion_planner.pred_dir, exist_ok=True)
    
    actioner = RobotPipeline(pipeline_config)

    @app.route('/predict', methods=['POST'])
    def predict():
        #(step_id, obs_state_dict) 
        data = request.data
        print("Hello:")
        #import pudb; pudb.set_trace()
        batch = msgpack_numpy.unpackb(data)
        # print(batch.keys())
        # print('cache', batch.get('cache', None))
        # # fetch the current observation, and predict one action
        # batch = {
        #     'task_str': task_str,
        #     'variation': variation,
        #     'step_id': step_id,
        #     'obs_state_dict': obs_state_dict,
        #     'episode_id': demo_id,
        #     'instructions': instructions,
        #     'cache': cache,
        # }
        batch = preprocess_batch_format(batch)
        output = actioner.predict(**batch)
        # action = output["action"]
        #cache =  output["cache"]
        print("Received: Step id", batch['step_id'])
        print("obs state dict:", batch['obs_state_dict'].keys())
        return msgpack_numpy.packb(output)
    
    app.run(host=args.ip, port=args.port, debug=args.debug)

'''
    Instructions:
        run in cleps: salloc -c 8 --gres=gpu:a100:1 --hint=multithread -p gpu
        then in your local machine: ssh -N -v -v  -L 8001:gpu013:8001 cleps -i ~/.ssh/jz_rsa
        then run the code: python server.py --ip $(hostname)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D-LOTUS server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    main(args)


