import time
from serverless_actor import ServerlessActor


def handler(event, context):
    start_time = time.time()

    # From input parameters
    redis_host = event['redis_host']
    redis_port = event['redis_port']
    redis_password = event['redis_password']
    algo_name = event['algo_name']
    env_name = event['env_name']
    num_envs_per_worker = int(event['num_envs_per_worker'])
    rollout_fragment_length = int(event['rollout_fragment_length'])

    # From AWS Lambda context
    aws_request_id = context.aws_request_id

    # Run the actor
    actor = ServerlessActor(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        algo_name=algo_name,
        env_name=env_name,
        num_envs_per_worker=num_envs_per_worker,
        rollout_fragment_length=rollout_fragment_length,
    )
    actor.init_redis_client()
    model_weights = actor.redis_get_model_weights()
    actor.set_model_weights(model_weights)
    sample_batch = actor.sample()
    
    # Record time
    end_time = time.time()
    lambda_duration = end_time - start_time
    
    actor.redis_hset_sample_batch("sample_batch", aws_request_id, sample_batch)
    actor.redis_hset_lambda_duration("lambda_duration", aws_request_id, lambda_duration)

    return {
        "aws_request_id": aws_request_id,
        "lambda_duration": lambda_duration
    }
