```mermaid
sequenceDiagram
    autonumber
    participant S as train_eagle3.sh
    participant D as DeepSpeed Launcher
    participant M as eagle3_trainer.main()
    participant T as Eagle3TrainerDeepSpeed
    participant F as freq_map(d2t/t2d)
    participant B as Base Model Weights
    participant DS as Train Dataset(.pt/.ckpt)
    participant C as EagleDataCollator
    participant H as Frozen Base LM Head
    participant E as Eagle3 Draft Model
    participant O as DeepSpeed Optimizer
    participant W as Tracking/WandB
    participant V as Val Loader
    participant K as Checkpoint

    S->>D: deepspeed eagle3_trainer.py ...
    D->>M: parse args + init_distributed()
    M->>T: new Eagle3TrainerDeepSpeed(args)

    T->>F: load d2t/t2d vocab mapping
    alt use_target_model=True
        T->>B: load tokenizer + target_model(eval,frozen)
    end

    T->>DS: scan data_path, collect files, split train/val(95/5)
    T->>E: build Eagle3 model config(num_hidden_layers=1, draft_vocab_size)
    T->>B: load embed_tokens + lm_head weights
    B-->>E: init embed / mapped draft lm_head
    T->>E: freeze embed_tokens
    alt use_target_model=False (current default)
        T->>H: build frozen base_model_lm_head
    end

    T->>D: deepspeed.initialize(model, train_dataset, collator)
    T->>K: try load latest checkpoint
    M->>T: train()

    loop each epoch
        loop each batch
            D->>DS: __getitem__()
            alt current default offline hidden-state mode
                DS-->>D: input_ids, hidden_states, last_hidden_states, loss_mask
            else online target model mode
                DS->>B: generate hidden_states from target_model
                DS-->>D: input_ids, hidden_states, loss_mask
            end
            D->>C: collate + pad
            C-->>T: batch(input_ids, hidden_states, attention_mask, loss_mask, ...)

            T->>T: move batch to CUDA / cast dtype
            T->>T: _compute_loss(batch)

            alt current default offline hidden-state mode
                T->>H: last_hidden_states -> target_logits
                H-->>T: target_logits
                T->>T: right-shift padding(target_logits)
            else online target model mode
                T->>B: full_input_ids -> target logits
                B-->>T: target_logits
            end

            T->>E: forward(hidden_states,input_ids,attention_mask,target,prediction_length)
            E-->>T: loss_list, accuracy_list
            T->>T: total_loss = sum((0.8^i) * loss_i)
            T->>O: backward(total_loss)
            O->>E: optimizer step
            T->>W: log train metrics

            opt save_steps hit
                T->>K: save_checkpoint(step)
            end
        end

        T->>V: run validation
        V->>E: same forward path, no grad
        T->>W: log val metrics
        T->>K: save_checkpoint(epoch)
    end

```
```mermaid
sequenceDiagram
    autonumber
    participant PPO as RayPPOTrainer
    participant AR as Actor/Rollout Worker
    participant DB as DataBuffer
    participant DM as RolloutDrafterManager
    participant BG as EagleBackgroundTrainer
    participant SM as FSDP SGLang ShardingManager
    participant ENG as SGLang Engine

    PPO->>AR: generate_sequences()
    AR-->>PPO: rollout结果

    PPO->>AR: compute_log_prob(return_hidden_states=True)
    AR-->>PPO: old_log_probs + hidden_states
    PPO->>AR: add_drafter_data_to_buffer(batch, hidden_states)
    AR->>DB: add_batch(...)

    PPO->>AR: RL step结束
    AR->>DM: increment_rl_step()

    alt 到达drafter训练间隔
        DM->>BG: activate_training_model()
        loop background training
            BG->>DB: 取最近N步样本
            DB-->>BG: input_ids + hidden_states
            BG->>BG: 组batch并训练
            BG->>BG: optimizer.step()
        end
    end

    AR->>SM: wake_up()
    SM->>SM: 读取drafter_module.state_dict()
    SM->>ENG: update_drafter_weights(...)
    ENG-->>SM: flush_cache()
    Note over ENG: 后续rollout使用新草稿模型

```

```mermaid
flowchart LR
    A[RayPPOTrainer] --> B[FSDP ActorRollout Worker]
    B --> C[FSDP SGLang ShardingManager]
    B --> D[SGLang Rollout]
    D --> E[RolloutDrafterManager]
    E --> F[EagleBackgroundTrainer]
    F --> G[Drafter Module FSDP]
    C --> H[SGLang Inference Engine]

    A -->|触发generate / compute_log_prob| B
    B -->|训练侧模型参数| C
    C -->|同步actor权重| H

    D -->|rollout样本| A
    D -->|收集hidden_states| F
    F -->|更新drafter参数| G
    C -->|读取drafter_module参数| G
    C -->|update_drafter_weights| H

```


```mermaid
sequenceDiagram
    autonumber
    participant PPO as RayPPOTrainer
    participant WG as ActorRollout WorkerGroup
    participant W as FSDP ActorRollout Worker
    participant SR as SGLang Rollout
    participant DM as RolloutDrafterManager
    participant ENG as SGLang Engine
    participant DB as DataBuffer
    participant BG as EagleBackgroundTrainer
    participant SM as FSDP SGLang ShardingManager

    loop each RL step
        PPO->>WG: generate_sequences(gen_batch)
        WG->>W: generate_sequences(prompts)
        W->>SR: rollout.generate_sequences(...)

        alt should_collect_data_this_step() and collect_hidden_states_from_sgl=true
            SR->>ENG: generate(..., return_hidden_states=true)
            ENG-->>SR: responses + meta_info.hidden_states
            SR->>DM: check collect interval
            SR->>BG: collect_online_data(filtered_batch, engine_hidden_states)
            BG->>DB: add_batch(input_ids, prompts, responses, hidden_states)
        else normal rollout
            SR->>ENG: generate(..., return_hidden_states=false)
            ENG-->>SR: responses
        end

        SR-->>W: rollout batch
        W-->>WG: DataProto
        WG-->>PPO: gen_batch_output

        PPO->>WG: compute_log_prob(batch)
        WG->>W: compute_log_prob(data)
        W-->>WG: old_log_probs + entropy
        WG-->>PPO: old_log_prob result

        PPO->>WG: increment_rl_step()
        WG->>W: increment_rl_step()
        W->>SR: increment_rl_step()
        SR->>DM: increment_rl_step()
        SR->>BG: data_buffer.increment_step()

        alt current_rl_step % training_interval_steps == 0
            DM->>BG: activate_training_model(...)
            loop background drafter training
                BG->>DB: get_data_from_last_n_steps(...)
                DB-->>BG: samples with hidden_states
                BG->>BG: prepare_training_batch()
                BG->>BG: forward + vloss + ploss
                BG->>BG: backward + optimizer.step()
            end
        end
    end

    Note over SM: next rollout wake_up phase
    W->>SR: wake_up()
    SR->>SM: sharding_manager.wake_up()
    SM->>SM: read drafter_module.state_dict()
    SM->>ENG: update_drafter_weights(...)
    ENG-->>SM: flush_cache()

```




```mermaid
sequenceDiagram
    participant C as Controller / Trainer
    participant W as ActorRolloutRefWorker
    participant S as FSDPSGLangShardingManager
    participant R as SGLang Rollout
    participant E as SGLang Engine
    participant A as Actor/Critic PPO
    participant D as DrafterManager
    participant B as BackgroundTrainer
    participant M as Drafter FSDP Module

    C->>W: 初始化 actor / rollout / drafter
    W->>W: _build_drafter_model()
    W->>B: 持有 drafter_module_fsdp
    W->>S: sharding_manager.drafter_module = drafter_module_fsdp

    loop 每个 RL iteration
        C->>W: 开始本轮 rollout

        W->>S: wake_up()
        S->>S: 提取最新 actor 权重
        S->>E: update_weights(actor params)
        S->>S: 提取最新 drafter 权重 from drafter_module
        S->>E: update_drafter_weights(drafter params, is_draft_model=True)
        E-->>S: flush_cache()
        S-->>W: rollout engine ready

        W->>R: generate with current actor + current drafter
        R->>E: speculative decoding
        E-->>R: responses + hidden_states
        R->>B: collect_online_data(batch, hidden_states)

        Note over R,B: 收集本轮 drafter 在线蒸馏样本

        R-->>A: rollout outputs
        A->>A: reward / advantage / returns / old_logprob
        A->>A: PPO update(actor / critic / ref)
        A->>B: increment_rl_step()

        Note over A,B: 本轮 RL 主线完成，标记一个 RL step 边界

        D->>B: 启动或继续 background training loop
        B->>B: activate_training_model() if needed

        loop drafter background steps
            B->>B: 从 DataBuffer / deque 采样并组 batch
            alt Eagle
                B->>M: forward -> hidden state regression + token distillation
            else EAGLE3
                B->>B: frozen base lm_head(target_hidden_states) -> target_logits
                B->>M: forward -> multi-step distillation
            end
            M-->>B: loss
            B->>M: backward()
            B->>M: optimizer.step()

            Note over B,M: 这里完成的是 drafter 训练侧权重更新
        end

        Note over M,S: 下一次 sharding_manager.wake_up() 时读取最新 drafter_module 参数
        Note over S,E: 再通过 update_drafter_weights() 热同步到 SGLang engine
        Note over E,R: 后续 rollout 使用更新后的 drafter 权重
    end

```



主线

初始化 actor、rollout、drafter、background trainer
rollout 用当前 actor + drafter 做生成
SGLang engine 返回 responses 和 hidden states
hidden states 被送进 BackgroundTrainer.collect_online_data()
RL 主线拿 rollout 结果做 reward/advantage/PPO update
drafter 支线异步启动 training loop
trainer 从 DataBuffer/deque 组 batch
Eagle 或 EAGLE3 做在线蒸馏更新
定期保存 checkpoint，结束时 cleanup/offload
更新后的 drafter 继续服务下一轮 rollout



RL 主线：rollout -> reward/advantage -> PPO 更新 actor/critic
drafter 支线：rollout hidden_states -> collect_online_data -> 后台蒸馏训练 drafter