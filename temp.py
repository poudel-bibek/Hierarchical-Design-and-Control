    
    update_count = 0
    action_timesteps = 0
    best_reward = float('-inf') 
    best_loss = float('inf')
    best_eval = float('inf')
    avg_eval = 200.0 # arbitrary large number
    eval_veh_avg_wait = 200.0 
    eval_ped_avg_wait = 200.0    

    # Every iteration, save all the sampled actions to a json file (by appending to the file).
    # A newer policy does importance sampling only every iteration. 
    actions_file_path = os.path.join(save_dir, f'sampled_actions.json')
    open(actions_file_path, 'w').close()
    sampled_actions = []

    all_memories = Memory()
    for iteration in range(0, total_iterations): # Starting from 1 to prevent policy update in the very first iteration.
        print(f"\nStarting iteration: {iteration + 1}/{total_iterations} with {global_step} total steps so far\n")
        
        old_policy = control_ppo.policy_old.to(device)
        old_policy.share_memory() # Dont pickle separate policy_old for each worker. Despite this, the old policy is still stale.
        old_policy.eval() # So that dropout, batnorm, laternote etc. are not used during inference

        #print(f"Shared policy weights: {control_ppo.policy_old.state_dict()}")
        train_queue = mp.Queue()
        train_processes = []
        active_train_workers = []
        for rank in range(control_args['num_processes']):

            worker_seed = SEED + iteration * 1000 + rank
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    old_policy,
                    control_args_worker,
                    train_queue,
                    worker_seed,
                    shared_state_normalizer,
                    shared_reward_normalizer,
                    device)
                )
            p.start()
            train_processes.append(p)
            active_train_workers.append(rank)
        
        while active_train_workers:
            print(f"Active workers: {active_train_workers}")
            rank, memory = train_queue.get()

            if memory is None:
                print(f"Worker {rank} finished")
                active_train_workers.remove(rank)
            else:
                current_action_timesteps = len(memory.states)
                print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}")
                all_memories.actions.extend(torch.from_numpy(np.asarray(memory.actions)))
                all_memories.states.extend(torch.from_numpy(np.asarray(memory.states)))
                all_memories.values.extend(memory.values)
                all_memories.logprobs.extend(memory.logprobs)
                all_memories.rewards.extend(memory.rewards)
                all_memories.is_terminals.extend(memory.is_terminals)

                sampled_actions.append(memory.actions[0].tolist())
                action_timesteps += current_action_timesteps
                global_step += current_action_timesteps * train_config['action_duration'] 
                print(f"Action timesteps: {action_timesteps}, global step: {global_step}")
                del memory #https://pytorch.org/docs/stable/multiprocessing.html

                # Update PPO every n times (or close to n) action has been taken 
                if action_timesteps >= control_args['update_freq']:
                    print(f"Updating PPO with {len(all_memories.actions)} memories") 

                    update_count += 1
                    # Anneal after every update
                    if control_args['anneal_lr']:
                        current_lr = control_ppo.update_learning_rate(update_count, total_updates)

                    avg_reward = sum(all_memories.rewards) / len(all_memories.rewards)
                    print(f"\nAverage Reward (across all memories): {avg_reward}\n")
                    #print(f"\nAll memories rewards: {all_memories.rewards}")

                    loss = control_ppo.update(all_memories)

                    # Reset all memories
                    del all_memories
                    all_memories = Memory() 
                    action_timesteps = 0
                    print(f"Size of all memories after update: {len(all_memories.actions)}")

                    # Save both during sweep and non-sweep
                    # Save (and evaluate the latest policy) every save_freq updates
                    if update_count % control_args['save_freq'] == 0:
                        latest_policy_path = os.path.join(control_args['save_dir'], f'policy_at_step_{global_step}.pth')
                        save_policy(control_ppo.policy, shared_state_normalizer, latest_policy_path)
                    
                        print(f"Evaluating policy: {latest_policy_path} at step {global_step}")
                        eval_json = eval(control_args_worker, ppo_args, eval_args, policy_path=latest_policy_path, tl= False) # which policy to evaluate?
                        _, eval_veh_avg_wait, eval_ped_avg_wait, _, _ = get_averages(eval_json)
                        eval_veh_avg_wait = np.mean(eval_veh_avg_wait)
                        eval_ped_avg_wait = np.mean(eval_ped_avg_wait)
                        avg_eval = ((eval_veh_avg_wait + eval_ped_avg_wait) / 2)
                        print(f"Eval veh avg wait: {eval_veh_avg_wait}, eval ped avg wait: {eval_ped_avg_wait}, avg eval: {avg_eval}")

                    # Save best policies 
                    if avg_reward > best_reward:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_reward_policy.pth'))
                        best_reward = avg_reward
                    if loss['total_loss'] < best_loss:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_loss_policy.pth'))
                        best_loss = loss['total_loss']
                    if avg_eval < best_eval:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_eval_policy.pth'))
                        best_eval = avg_eval

                    # logging
                    if is_sweep: # Wandb for hyperparameter tuning
                        wandb.log({ "iteration": iteration,
                                        "avg_reward": avg_reward, # Set as maximize in the sweep config
                                        "update_count": update_count,
                                        "policy_loss": loss['policy_loss'],
                                        "value_loss": loss['value_loss'], 
                                        "entropy_loss": loss['entropy_loss'],
                                        "total_loss": loss['total_loss'],
                                        "current_lr": current_lr if control_args['anneal_lr'] else ppo_args['lr'],
                                        "approx_kl": loss['approx_kl'],
                                        "eval_veh_avg_wait": eval_veh_avg_wait,
                                        "eval_ped_avg_wait": eval_ped_avg_wait,
                                        "avg_eval": avg_eval,
                                        "global_step": global_step })
                        
                    else: # Tensorboard for regular training
                        writer.add_scalar('Training/Average_Reward', avg_reward, global_step)
                        writer.add_scalar('Training/Total_Policy_Updates', update_count, global_step)
                        writer.add_scalar('Training/Policy_Loss', loss['policy_loss'], global_step)
                        writer.add_scalar('Training/Value_Loss', loss['value_loss'], global_step)
                        writer.add_scalar('Training/Entropy_Loss', loss['entropy_loss'], global_step)
                        writer.add_scalar('Training/Total_Loss', loss['total_loss'], global_step)
                        writer.add_scalar('Training/Current_LR', current_lr if control_args['anneal_lr'] else ppo_args['lr'], global_step)
                        writer.add_scalar('Training/Approx_KL', loss['approx_kl'], global_step)
                        writer.add_scalar('Evaluation/Veh_Avg_Wait', eval_veh_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Ped_Avg_Wait', eval_ped_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Avg_Eval', avg_eval, global_step)
                    print(f"\nLogged data at step {global_step}\n")

                    # At the end of update, save normalizer stats
                    state_normalizer_mean = shared_state_normalizer.mean.numpy()  
                    state_normalizer_M2 = shared_state_normalizer.M2.numpy()  
                    state_normalizer_count = shared_state_normalizer.count.value  

        # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in train_processes:
            p.join() 
        print(f"All processes joined\n\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del train_queue

        # Save all the sampled actions to a json file
        with open(actions_file_path, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[iteration] = sampled_actions
            f.seek(0)
            #print(f"Sampled actions: {sampled_actions}")
            json.dump(data, f, indent=4)
            f.truncate()
            f.close()
        sampled_actions = []

    if not is_sweep:
        writer.close()