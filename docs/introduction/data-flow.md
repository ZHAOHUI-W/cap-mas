# CAP-MAS 数据流图

> 生成日期：2026-07-22

## 1. 系统整体数据流

```mermaid
flowchart TB
    subgraph INPUT["输入层"]
        TASK["自然语言任务"]
        YAML["CAP-X YAML 配置"]
    end

    subgraph BACKEND["后端适配层 backends/"]
        FACTORY["build_capx_runtime_from_yaml()"]
        CAPXBE["CAPXRobotBackend"]
        CAPXOBS["CAPXObservationProvider"]
        FACTORY --> CAPXBE
        FACTORY --> CAPXOBS
    end

    subgraph LLM_LAYER["LLM 层 llm/"]
        CLIENT["CAPXCompatibleLLMClient<br/>HTTP+重试+结构化输出"]
        MGD["MissionGraphDecoder<br/>JSON→schema→场景→图验证"]
        MTD["MissionTopologyDecoder<br/>JSON→schema→场景→拓扑验证"]
        LSD["LocalSubgraphDecoder<br/>JSON→schema→子图验证"]
        PROMPTS["Prompts<br/>Schema生成+Prompt构建"]
        CLIENT --> MGD
        CLIENT --> MTD
        CLIENT --> LSD
        PROMPTS --> CLIENT
    end

    subgraph AGENT["智能体层 agents/"]
        SMM["SimpleMissionManager<br/>确定性子目标"]
        LLM_MM["LLMMissionManager<br/>LLM→完整MissionGraph"]
        LLM_TM["LLMTopologyManager<br/>阶段一: LLM→紧凑拓扑"]
        LLM_PA["LLMGraphPolicyAgent<br/>LLM→局部子图(从完整图)"]
        LLM_SPA["LLMStagedGraphPolicyAgent<br/>阶段二: LLM→直接局部图"]
        ARB["CandidateArbiter<br/>验证+多因子评分+选择"]
    end

    subgraph CONTRACTS["合约层 contracts/"]
        AC["ActionContract"]
        SS["SceneSnapshot"]
        MG["MissionGraph"]
        SG["SubgraphSpec"]
        MT["MissionTopology"]
        TS["TopologySubgoal"]
        CE["CandidateEvidence"]
        VR["VerificationResult"]
        FA["FailureArtifact"]
    end

    subgraph GRAPH["图策略层 graph/"]
        GV["GraphValidator"]
        TV["TopologyValidator"]
        SER["Graph Serialization"]
        STAGED["Staged Serialization<br/>+ Topology.assemble()"]
    end

    subgraph RUNTIME["运行时层 runtime/"]
        ORC["RuntimeOrchestrator<br/>单周期编排"]
        ER["EpisodeRunner / MultiCycleRunner"]
        FI["FixedGraphInterpreter<br/>图→合约→执行"]
        LLM_SCH["LLMGraphScheduler<br/>多智能体编译+执行"]
        FGS["FixedGraphScheduler"]
        ALM["ActionLeaseManager"]
        SSTORE["InMemoryStateStore"]
        EBS["EventBus + ArtifactStore"]
        REC["MappingRecoverySelector"]
    end

    subgraph VERIFY["验证层 verification/"]
        PBV["PredicateBasedVerifier"]
        LOV["LiberoObservableVerifier"]
    end

    subgraph SKILL["技能系统 skills/"]
        SR["SkillRegistry"]
    end

    subgraph EVAL["评估层 evaluation/"]
        REWARD["CAPXBinaryReward"]
        PARITY["ParityComparison<br/>CAP-X vs CAP-MAS"]
    end

    %% 主数据流
    TASK --> SMM
    TASK --> LLM_MM
    TASK --> LLM_TM
    YAML --> FACTORY

    %% Legacy 协议流
    LLM_MM -->|"MissionGraph"| LLM_SCH
    LLM_MM --> MG
    LLM_PA -->|"SubgraphSpec 候选"| ARB
    ARB -->|"选中候选"| LLM_SCH

    %% Staged 协议流
    LLM_TM -->|"MissionTopology"| LLM_SCH
    LLM_TM --> MT
    MT -->|"子目标分发"| LLM_SPA
    LLM_SPA -->|"SubgraphSpec 候选"| ARB
    MT -->|"assemble(subgraphs)"| MG

    %% 图执行流
    LLM_SCH -->|"编译后 MissionGraph"| FI
    FI -->|"ActionContract (降级)"| FGS
    FGS --> ORC

    %% 运行时编排流
    ORC -->|"1. 验证"| PBV
    PBV --> ORC
    ORC -->|"2. 租约"| ALM
    ORC -->|"3. 执行技能"| CAPXBE
    ORC -->|"4. 观察"| CAPXOBS
    CAPXOBS --> SSTORE
    ORC -->|"5. 提交验证"| PBV

    %% 场景流
    SSTORE -->|"latest()"| LLM_MM
    SSTORE -->|"latest()"| LLM_TM
    SSTORE -->|"latest()"| LLM_PA
    SSTORE -->|"latest()"| FI

    %% 恢复流
    ORC -->|"失败"| REC
    REC -->|"RecoveryDecision"| FI

    %% 评估流
    ORC --> PARITY

    %% LLM 层
    PROMPTS --> LLM_MM
    PROMPTS --> LLM_TM
    PROMPTS --> LLM_PA
    PROMPTS --> LLM_SPA

    style AGENT fill:#e1f5fe
    style RUNTIME fill:#fff3e0
    style LLM_LAYER fill:#f3e5f5
    style GRAPH fill:#e8eaf6
    style CONTRACTS fill:#e0f2f1
```

---

## 2. Legacy 协议数据流（Manager 输出完整图）

```mermaid
sequenceDiagram
    participant M as LLMMissionManager
    participant LLM as LLM Client
    participant Dec as MissionGraphDecoder
    participant PA as LLMGraphPolicyAgent(s)
    participant Arb as CandidateArbiter
    participant Sch as LLMGraphScheduler
    participant Int as FixedGraphInterpreter
    participant Orc as RuntimeOrchestrator
    participant Ver as Verifier
    participant BE as RobotBackend

    M->>LLM: propose_graph 请求
    LLM-->>Dec: LLM Response
    Dec->>Dec: JSON解析→Schema验证→场景检查→图验证
    Dec-->>M: MissionGraph (完整图)

    loop 每个子图
        Sch->>PA: propose_subgraph(子目标)
        PA->>LLM: 局部图请求
        LLM-->>PA: LLM Response
        PA->>Dec: 解码+提取目标子图
        PA-->>Sch: SubgraphSpec 候选
    end

    Sch->>Arb: 候选集合
    Arb->>Arb: 验证+评分+选择
    Arb-->>Sch: 选中候选

    Sch->>Int: 编译后 MissionGraph
    Int->>Int: 按控制流遍历

    loop 每个 action 节点
        Int->>Int: 降级为 ActionContract
        Int->>Orc: run_cycle(contract)
        Orc->>Ver: approve(contract, scene)
        Ver-->>Orc: approve
        Orc->>BE: execute_skill(skill, args, budget)
        BE-->>Orc: SkillExecutionResult
        Orc->>BE: observe()
        BE-->>Orc: 新 SceneSnapshot
        Orc->>Ver: commit(contract, before, after)
        Ver-->>Orc: commit / recover
    end
```

---

## 3. Staged 协议数据流（Manager 只输出拓扑）

```mermaid
sequenceDiagram
    participant M as LLMTopologyManager
    participant LLM as LLM Client
    participant TDec as MissionTopologyDecoder
    participant PA as LLMStagedGraphPolicyAgent(s)
    participant SDec as LocalSubgraphDecoder
    participant Arb as CandidateArbiter
    participant Sch as LLMGraphScheduler
    participant Top as MissionTopology
    participant Int as FixedGraphInterpreter
    participant Orc as RuntimeOrchestrator

    M->>LLM: propose_topology 请求
    LLM-->>TDec: LLM Response
    TDec->>TDec: JSON解析→Schema验证→场景检查→拓扑验证
    TDec-->>M: MissionTopology (紧凑拓扑)

    par 并行: 每个子目标
        Sch->>PA: propose_subgraph(拓扑子目标)
        PA->>LLM: 局部图请求
        LLM-->>SDec: LLM Response
        SDec->>SDec: JSON解析→Schema验证→子图验证
        SDec-->>PA: SubgraphSpec
        PA-->>Sch: SubgraphSpec 候选
    end

    Sch->>Arb: 候选集合
    Arb->>Arb: 验证+评分+选择
    Arb-->>Sch: 选中候选

    Sch->>Top: assemble(选中子图集合)
    Top->>Top: 验证子图匹配+推断绑定+归一化边
    Top-->>Sch: 完整 MissionGraph

    Sch->>Int: 编译后 MissionGraph
    Int->>Int: 按控制流执行
```

---

## 4. 单周期执行数据流

```mermaid
sequenceDiagram
    participant Runner as Episode Runner
    participant Policy as Policy Agent
    participant Orch as RuntimeOrchestrator
    participant Verifier as Verifier
    participant Lease as Lease Manager
    participant Backend as Robot Backend
    participant Store as State Store
    participant Registry as Skill Registry

    Runner->>Store: latest() → SceneSnapshot
    Runner->>Policy: propose(context)
    Policy-->>Runner: ActionContract

    Runner->>Orch: run_cycle(contract)

    Note over Orch: 阶段1: 场景版本检查
    Orch->>Orch: 检查 contract.parent_scene_version == current.scene_version

    Note over Orch: 阶段2: 技能验证
    Orch->>Registry: validate_contract(contract)
    Registry->>Registry: 检查技能存在+参数合法

    Note over Orch: 阶段3: 前置验证
    Orch->>Verifier: approve(contract, scene)
    Verifier->>Verifier: 检查前置条件+安全不变量
    Verifier-->>Orch: approve ✓ / reject ✗

    Note over Orch: 阶段4: 获取租约
    Orch->>Lease: acquire(holder, contract_id, duration)
    Lease-->>Orch: ActionLease

    Note over Orch: 阶段5: 技能执行
    loop 每个技能调用
        Orch->>Orch: 解析 SkillOutputRef 引用
        Orch->>Backend: execute_skill(skill, args, budget)
        Backend-->>Orch: SkillExecutionResult
        alt 技能失败
            Orch->>Backend: observe()
            Orch->>Store: compare_and_commit()
            Orch-->>Runner: CycleResult(committed=False)
        end
    end

    Note over Orch: 阶段6: 观察新场景
    Orch->>Backend: observe()
    Backend-->>Orch: 新 SceneSnapshot

    Note over Orch: 阶段7: 提交场景
    Orch->>Store: compare_and_commit(parent, new_scene)

    Note over Orch: 阶段8: 后置验证
    Orch->>Verifier: commit(contract, before, after, trace)
    Verifier->>Verifier: 检查后置条件
    Verifier-->>Orch: commit ✓ / recover ✗

    Note over Orch: 阶段9: 释放租约
    Orch->>Lease: release(lease_id)

    Orch-->>Runner: CycleResult
```

---

## 5. LLM 严格解码数据流

```mermaid
flowchart LR
    subgraph INPUT["LLM 响应"]
        STRUCTURED["response.structured<br/>(结构化输出)"]
        CONTENT["response.content<br/>(JSON文本)"]
    end

    subgraph DECODE["解码管线"]
        PARSE["JSON 解析"]
        SCHEMA["Schema 版本验证"]
        SCENE["场景版本检查"]
        VALIDATE["图/拓扑结构验证"]
    end

    subgraph OUTPUT["解码结果"]
        ACCEPT["GraphDecodeResult<br/>accepted=True<br/>graph=MissionGraph"]
        REJECT["GraphDecodeResult<br/>accepted=False<br/>rejections=[...]"]
    end

    STRUCTURED --> PARSE
    CONTENT --> PARSE
    PARSE --> SCHEMA
    SCHEMA --> SCENE
    SCENE --> VALIDATE
    VALIDATE --> ACCEPT
    PARSE -->|"JSON_INVALID"| REJECT
    SCHEMA -->|"GRAPH_SCHEMA_INVALID"| REJECT
    SCENE -->|"STALE_SCENE / MISSING_PARENT_SCENE"| REJECT
    VALIDATE -->|"图验证错误"| REJECT

    style DECODE fill:#fff3e0
    style ACCEPT fill:#e8f5e9
    style REJECT fill:#ffebee
```

---

## 6. 候选仲裁数据流

```mermaid
flowchart TB
    subgraph INPUT["多个 Policy Agent 候选"]
        C1["Candidate 1<br/>confidence=0.9<br/>evidence=..."]
        C2["Candidate 2<br/>confidence=0.7<br/>evidence=..."]
        C3["Candidate 3<br/>confidence=0.8<br/>evidence=..."]
    end

    subgraph ARBITER["CandidateArbiter"]
        CHECK1["1. 去重检查"]
        CHECK2["2. 场景版本检查"]
        CHECK3["3. 图结构验证"]
        CHECK4["4. 子目标一致性检查"]
        SCORE["5. 多因子评分"]
    end

    subgraph SCORING["评分公式"]
        FORMULA["evidence score = sum(declared available_metrics)<br/>profile weights × declared evidence<br/>- latency/recovery penalties when declared<br/><br/>P3.2 LIBERO: perception only<br/>confidence only in fallback mode"]
    end

    subgraph OUTPUT["仲裁结果"]
        SELECTED["选中候选<br/>(最高分)"]
        REJECTED["拒绝列表<br/>+ 拒绝原因"]
    end

    C1 --> CHECK1
    C2 --> CHECK1
    C3 --> CHECK1
    CHECK1 --> CHECK2
    CHECK2 --> CHECK3
    CHECK3 --> CHECK4
    CHECK4 --> SCORE
    SCORE --> FORMULA
    FORMULA --> SELECTED
    CHECK1 -->|"DUPLICATE"| REJECTED
    CHECK2 -->|"STALE_SCENE"| REJECTED
    CHECK3 -->|"图验证失败"| REJECTED
    CHECK4 -->|"SUBGOAL_MISMATCH"| REJECTED

    style ARBITER fill:#e1f5fe
    style SCORING fill:#fff3e0
```

---

## 7. 恢复数据流

```mermaid
flowchart TB
    subgraph FAILURE["失败来源"]
        PRE["PRECONDITION_FAILED"]
        EXEC["EXECUTION_ERROR"]
        POST["POSTCONDITION_FAILED"]
        COLL["COLLISION_RISK"]
        STALE["STALE_STATE"]
    end

    subgraph CLASSIFY["失败分类"]
        FA["FailureArtifact<br/>failure_class + message<br/>+ recoverable + retry_count"]
    end

    subgraph RECOVERY["恢复策略"]
        RS["MappingRecoverySelector<br/>failure_class → target_subgraph"]
        RD["RecoveryDecision<br/>target_subgraph + rationale"]
    end

    subgraph ROUTE["恢复路由"]
        R1["PRECONDITION_FAILED → replan"]
        R2["EXECUTION_ERROR → retry"]
        R3["POSTCONDITION_FAILED → replan"]
        R4["COLLISION_RISK → retreat"]
        R5["* → default"]
    end

    subgraph EXEC["恢复执行"]
        FI["FixedGraphInterpreter<br/>跳转到 target_subgraph"]
    end

    FAILURE --> FA
    FA --> RS
    RS --> RD
    RD --> FI

    PRE --> R1
    EXEC --> R2
    POST --> R3
    COLL --> R4
    STALE --> R5

    style FAILURE fill:#ffebee
    style RECOVERY fill:#e8f5e9
```

---

## 8. 关键数据流不变量

| 不变量 | 保证机制 |
|--------|----------|
| 控制平面永不等待智能体平面 | 场景快照异步发布，控制器消费最新有效快照 |
| 同一时刻只有一个合约控制机器人 | ActionLeaseManager 互斥租约 |
| 场景版本单调递增 | InMemoryStateStore CAS 提交 |
| LLM 输出必须通过严格解码 | MissionGraphDecoder / MissionTopologyDecoder / LocalSubgraphDecoder |
| 候选必须通过图验证才能被选中 | CandidateArbiter 先验证后评分 |
| 失败必须分类后才能恢复 | FailureArtifact + RecoverySelector |
| 图环路必须有界 | LoopSpec.max_visits + FixedGraphInterpreter 检查 |
| 技能调用只能引用已注册技能 | SkillRegistry.validate_contract() |
