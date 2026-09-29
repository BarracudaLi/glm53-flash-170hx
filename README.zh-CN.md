# GLM-5.3-Flash 在 4× CMP 170HX 上的推理服务

> 🌐 语言 / Language：**简体中文** · [English](./README.md)

生产环境部署 GLM-5.3-Flash（320B 参数 MoE，18B 激活，原生多模态）于四张 NVIDIA CMP 170HX GPU（SM80，PCIe Gen2 x4，无 P2P/NVLink）：

- 流水线并行 4（PP4）；此互联拓扑下 TP 不可行
- EXL3 4bpw 权重 + 离线生成的 Marlin INT4 解码 sidecar
- DFlash2 投机解码，k=7，贪心草稿，自适应验证长度（{4,7} EMA 画像）
- FULL_AND_PIECEWISE CUDA 图覆盖 Ampere（PP4）的整个解码步骤——本机解码提速的最大杠杆
- Strided KDA 循环输入（移植 vLLM PR #55736）：解码 KDA 原地消费合并投影的列切片，无需逐层打包拷贝
- 512K 上下文窗口；KV 缓存在 GPU HBM（约 1.58M token）与 128 GiB 固定内存 `/dev/shm` CPU 层（约 3M token）之间分层——总计约 4.5M token
- 崩溃安全的 CPU KV offload 区域：所有流水线 rank 映射后即 unlink 底层 mmap，SIGKILL 或卡死也不会泄漏 `/dev/shm` 文件
- 跨容器重启的持久化 Triton/vLLM 编译缓存（热启动 15–16 分钟 → 9.3 分钟）
- 基于 Docker 的构建与部署

本仓库是 [hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized)（下文简称 *upstream*）的部署补丁层。upstream 原样无法在这套硬件上正确部署；`patches/` 中的补丁修复了遇到的各类失败，详见[技术修改](#技术修改)。`exllamav3` 依赖从 0.0.43 升级到 1.4.8，将预填充吞吐提升 5–8×。

## 结果（4× CMP 170HX 64 GB，200 W 功率上限）

| 指标 | v22（此前） | v24（当前） |
|---|---|---|
| 预填充吞吐（冷，131k） | 2,078–2,084 tok/s | 2,077 tok/s（持平） |
| 预填充吞吐（热前缀命中） | 最高 14.7k tok/s @64k | 机制不变 |
| 单流解码 C1 | json 97 / code 63–71 / prose 22–24 tok/s | **json 158–161 / code 96–100 / prose 40–42 / counting 158–163 tok/s** |
| 单流长上下文解码 | 47–48 @32k / 42–45 @128k | **69–70 @32k / 68–70 @128k** |
| 聚合解码 | N4=153 / N8=185–213 | **N4=199 / N8=241–256** |
| 6 路并发长上下文解码 | 262–267 tok/s | **265–272 tok/s** |
| 投机接受长度 | 最高 8.0/8（DFlash2 k=7） | 不变 |
| 长上下文 | needle 64k/128k/256k 通过；4× 467K 常驻 | 不变 |
| KV 缓存池 | 约 4.5M token（1.58M GPU + 约 3M CPU 层） | 不变（util 0.93） |
| 热重启 | 15–16 分钟 | **9.3 分钟**（持久化编译缓存） |
| 生产验收 | 质量 6/6、工具调用、推理、视觉、0 错误浸泡 | v24 复验（0 错误 / 0 NaN） |

本机测试后否决的项（记录下来以免他人盲目重试）：`-lgc` 时钟锁定（79 vs 158 tok/s counting，不锁频的 DVFS 胜出）；NCCL `Ring/Simple` 固定（PP 下 FULL 图不需要；`PROTO=Simple` 拖慢了 16 MB 的 PP 隐藏态传输，冷预填充损失 7–15%）。

相同硬件上的参照点：NVFP4 方案（预填充 3.7k tok/s，单流 36–44 tok/s，64K 上下文以上不稳定）以及无投机解码的 AWQ 方案（约 40 tok/s）。本方案预填充与之持平，同时解码/并发吞吐翻倍，并把上下文窗口扩展到 512K。

## 硬件要求

- 4× 计算能力 8.0 的 GPU，每张 ≥64 GB（CMP 170HX 解锁到 64 GB；驱动 610.43.03 open + cmpunlocker）
- PCIe Gen2 x4 链路足够：PP4 不做逐层 all-reduce。此拓扑下 TP4 预填充仅约 800 tok/s，不支持。
- ≥229 GB 系统内存（加载期页面缓存峰值）
- 约 530 GB 磁盘：EXL3 checkpoint 164 GB + Marlin sidecar 151 GB + DFlash2 draft 2.3 GB + 容器镜像
- upstream 源码树 + 本仓库的补丁（或本仓库自带的、已包含构建修复的 Dockerfile）

## 仓库结构

```
patches/    针对 upstream HEAD 的 7 个 diff（按编号顺序打补丁）。
            006 为 AGPL-3.0（见 License）；其余为 Apache-2.0。
deploy/     build.sh（镜像构建）、launch_exl3.sh（推理服务）、
            lanes2.sh（4 卡上生成 sidecar）
bench/      投机解码、并发、needle、预填充阶梯、生产验收、
            以及流式真实解码速率（agent_ab.py / agent_conc.py）测试脚本
docs/       results.md — 完整基准测试表
```

## 快速开始

```bash
# 1) 获取 upstream 源码并打补丁
git clone https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized.git
cd vllm_170hx_glm53-flash-exl3-optimized
git apply ../patches/001-*.patch ../patches/002-*.patch ../patches/003-*.patch \
          ../patches/004-*.patch ../patches/005-*.patch ../patches/006-*.patch \
          ../patches/007-*.patch

# 2) 构建镜像（首次构建约 5 小时，28 核）
#    Dockerfile 从本地副本安装 flashinfer 0.6.17 wheel
#    （需预先放到 fi/ 目录下；原因见 003 补丁）。
bash ../deploy/build.sh

# 3) 下载权重
#    EXL3 checkpoint（164 GB）：Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw（ModelScope）
#      （与 Hugging Face 的 brandonmusic/GLM-5.3-Flash-tr3-4bpw 配置逐字节一致）
#    DFlash2 draft（2.3 GB）：incoai/GLM-5.3-Flash-DFlash2
#      （CC BY-NC-ND — 商用前请审阅其许可证）

# 4) 生成 Marlin sidecar（42 层 × 3.6 GB，4 卡约 25 分钟）
bash ../deploy/lanes2.sh

# 5) 为 CPU KV 层扩容 /dev/shm（默认 tmpfs 太小）：
sudo mount -o remount,size=170G /dev/shm

# 6) 启动（512K 窗口，util 0.94，DFlash2 k=7 + adaptive-k，128 GiB CPU KV
#    层，200 W 功率上限）
bash ../deploy/launch_exl3.sh
```

## 技术修改

所有修改都是针对 upstream HEAD 的 diff，放在 `patches/` 中。每一条都对应一个已复现的具体失败：

1. **sidecar 切换后的 EXL3 张量释放（补丁 001，`exl3.py`）**
   upstream 在 Marlin sidecar 上线后，用空张量替换每层的 EXL3 参数，但原张量未及时释放，每个 MoE 层净增约 3.6 GB。加载在第 8 个 MoE 层确定性 OOM（复现 3/3）。补丁在切换后强制 `gc.collect()` + `cuda.synchronize()` + `empty_cache()`；之后每设备显存稳定在约 41–44 GB，贯穿全部 42 层。

2. **`VLLM_PRETEND_NO_DEEP_GEMM=1`（补丁 002，`import_utils.py`）**
   容器镜像在 arch 列表中包含 SM80 编译了内置 DeepGEMM，使 `has_deep_gemm()` 返回 True；其 attention API 在运行时断言 SM90+（表现为 `deepgemm-src/csrc/apis/attention.hpp:270` 处的启动中止）。补丁给 `has_deep_gemm()` 加了环境钩子，将包报告为不存在，把所有调用点路由到 Triton 兜底——与 upstream 作者（未安装 DeepGEMM）的环境一致。

3. **真实的 `exllamav3.model.config` 加载与搜索路径顺序（补丁 001，`_exl3_module()`）**
   upstream 在 `sys.modules` 里装命名空间桩以绕过 `exllamav3` 包初始化器，其中包括一个只定义了 dummy `Config` 的合成 `exllamav3.model.config`。在 exllamav3 ≥1.4 下，`LinearEXL3.__init__` 会惰性从该模块导入 `NullConfig`，对着桩会失败。此外，导入真实 `config.py` 会连带导入 `ext.py`，除非预编译 `.so` 目录已在 `sys.path` 上（运行时镜像没有 nvcc，无法 JIT 构建），否则会触发扩展的 JIT 重编译。补丁（a）在装桩之前先把预编译扩展目录插入 `sys.path`，（b）通过 importlib 从文件加载真实 `config.py`，激活 `ext.py` 的预编译扩展分支。

4. **按 rank 错峰 H2D（`VLLM_EXL3_H2D_STAGGER_S=12`，补丁 001）**
   4 个流水线 rank 同时做 sidecar 的 host-to-device 传输，会触发 Gen2 x4 上的 PCIe 链路重训练，表现为异步越界写（Xid 31，各 rank 在不同层失败）。补丁把每个 rank 的首次 sidecar 传输延迟 `rank × stagger` 秒。这是针对本互联拓扑的缓解手段；在 NVLink 系统上不必要但无害。

5. **Dockerfile 构建修复（补丁 003）**
   - upstream 固定 FlashInfer `v0.6.18rc10`，该版本缺少 cubin wheel（构建时 HTTP 404）；改为固定 0.6.17 并从本地提供的 wheel 安装，避免瞬时网络故障。
   - upstream Dockerfile 中 `max_jobs` 默认为 2；提高到 20（多核主机上 CUDA 编译约快 10 倍）。
   - 禁用 500 MB wheel 体积检查（`RUN_WHEEL_CHECK=false`），该检查仅对 PyPI 发布有意义。
   - 配置 AlmaLinux/PyPI 镜像，供中国大陆构建使用。

6. **移除 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
   upstream 作者的配方设置了这个分配器模式，但它与 mmap 支持的 CPU KV 层（补丁 004）不兼容：CUDA 分配器的可扩展段与对共享 mmap 的 pinning/registering 视图冲突。启动脚本特意不设置它。

7. **CPU KV offload 层（补丁 004，六个文件）**
   把 `OffloadingConnector` 接入可用的 PP4 配置：
   - 移植 vLLM PR #50653（统一各流水线 rank 的 `num_cpu_blocks`——否则异构 PP rank 会算出不同的 CPU block 数，连接器启动时死锁）；
   - `num_cpu_blocks` 贯通 `KVCacheConfig` 与 offloading 配置；
   - 放宽 `build_offloading_config` 中的整除断言，跳过从不参与前缀缓存的 ring-buffer 组（如 Kpool tail spec）；
   - 运行时标志：`--enable-prefix-caching --prefix-caching-hash-algo xxhash --prefix-match-unit 4`。match unit 是各组 block 大小的 GCD（576 宽对齐需要一个 upstream 尚未实现的「排除非参与组」特性）。
   效果：KV 池 1.53M → 约 4.5M token；4× 467K 请求常驻；冷长上下文预填充下降 30–50%（簿记 + 细粒度哈希），热共享前缀预填充升至 14.7k tok/s。

8. **崩溃安全的 offload mmap（补丁 005，移植 vLLM PR #52596）**
   `SharedOffloadRegion` 获得一个由 `CPUOffloadingSpec` 提供的 `barrier`（gloo world-group）；一旦每个流水线 rank 都映射了文件，创建者就 unlink 路径。unlink 前建立的映射仍然有效，因此内核在最后一个 worker 退出时回收 128 GiB 区域——包括 SIGKILL 或 GPU 卡死硬停机。这消除了「残留的 vllm_offload_*.mmap 卡住下次启动」这类失败，顺带解决了两个引擎实例间的命名冲突。与本地 `VLLM_OFFLOAD_MMAP_MIN_BYTES` 下限合并（创建者把文件大小定为至少满足最大 rank 的期望，异构 PP rank 布局所需）。

9. **DFlash2 自适应验证长度（补丁 006，AGPL-3.0，移植自 MiaAI-Lab）**
   drafter 保留其训练好的 8-token 块；调度器只验证每步前缀，该前缀从每个请求的已接受草稿 EMA 中选出（2/4/7 候选，batch-uniform，每个候选长度多一个 uniform 解码 CUDA 图）。环境变量门控（`GLM53_ADAPTIVE_K`，默认关），带运行时 JSON 覆盖。本硬件生产画像：`{4,7}, alpha=0.15, margin=1.0`——6 路并发长上下文负载下聚合解码 +12–13%，128k 单流 −4%。单流时本机受延迟约束（步进速率与 k 无关，裁剪只会损失接受率）；收益出现在验证计算成为瓶颈时（≥6 并发流）。画像扫描与流式测量方法见 docs/results.md。

10. **v24 解码提速包（补丁 + 配置）**
    - 补丁 007（移植 vLLM PR #55736）：融合的 KDA 循环内核按 token 步长遍历 q/k/v/beta，解码时原地消费合并投影的列切片，省去每层每步 4 次打包拷贝；输出缓冲区按稠密分配（内核稠密写 o）。
    - `cudagraph_mode=FULL_AND_PIECEWISE`，capture 列表最多 64 token：在 FULL_DECODE_ONLY 下只有最小解码批次会回放图，更大批次走 eager，PP 阶段流水线变得 CPU 派发受限。不做 NCCL 固定：PP4 上 FULL 图无需它也能稳定回放（而 `PROTO=Simple` 会拖慢大 PP 传输）。
    - DFlash2 `draft_sample_method=greedy`：温度 0 时贪心草稿把草稿概率固定为 1，使概率比测试严格更容易通过。
    - 从宿主机挂载的持久化编译缓存（`/root/.triton`、`/root/.cache/vllm`）：热重启跳过 Triton JIT 与 inductor 编译。
    本机综合效果：单流解码 +48-80%，N8 聚合 +20-37%，冷预填充持平，热重启 15-16 → 9.3 分钟。

### exllamav3 0.0.43 → 1.4.8

upstream 固定 exllamav3 0.0.43。升级到 [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) 1.4.8 需要修改 3，除此之外 API 兼容（`BC_BlockSparseMLP` 迁移到 `libtorch/blocksparse_mlp_bc.h`）。实测效果：预填充 480 → 2,040–3,926 tok/s（5–8×），解码与并发吞吐不变。

### 投机深度选择（DFlash2）

在 k = 2/4/6/7 上实测（tok/s）：json 43→70→92→98，code 39→54→55→74，math 在 k=6 峰值（51），prose 不敏感（约 23，接受率 1.6）。默认 k=7；数学为主的工作负载首选 k=6。KV 池随 k 略缩（k4 = 1.37M / k7 = 1.25M token @util 0.92；1.53M @util 0.94）。补丁 006（adaptive-k）保持 num_speculative_tokens=7，改为每步裁剪已验证前缀——见修改 9。

## 基准测试

完整表格见 [docs/results.md](docs/results.md)：预填充曲线、k 扫描、并发、needle 检索、流式延迟、64K 输出、按配置的 KV 池，以及生产验收计分卡（通过推理链路上的 litellm 网关测得）。

## 已知限制与运维

- **崩溃后的 GPU 卡死。** Xid 31 故障会让受影响的 GPU 处于 `cudaSetDevice` 失败（`cudaErrorDevicesUnavailable`）的状态，直到断电；本产品不支持 `nvidia-smi -r`。恢复方式是冷断电重启（含引擎重启约 10 分钟）。观测频率约为每 10 次引擎启动 1 次。
- **util 0.94 是上限。** 在 0.95 及以上，加载与长预填充阶段会随机出现越界写失败（本板 Gen2 x4 拓扑的显存余量极限）。
- **内核升级风险。** 无人值守升级会安装不带 cmpunlocker 模块的内核。`GRUB_DEFAULT=saved` 加 `grub-set-default` 把启动固定到带驱动的内核；出现 `driver not loaded` 症状时，先查 `uname -r`。
- **原生 MTP 未接通。** checkpoint 自带完整的 NextN 层（layer 45，EXL3 量化），但 vLLM MTP 草稿路径需要把 `_make_fused_bsz1` 移植到 1.4.8 的 `BC_BlockSparseMLP` 构造器（签名从约 45 个参数涨到约 55 个，新增 z/hyper-connection 捆绑）；预计半天工作量。参照测量也显示 DFlash2 草稿在该模型家族上优于 MTP。
- **较小的 `max_tokens` 预算。** 模型总是先输出推理再输出内容；`max_tokens < 100` 时响应可能只含推理。客户端应请求 ≥400 token，或依赖网关注入的默认值。
- **视觉。** 颜色识别、OCR、计数、轴对齐图表估算均通过。测试固件的轴标签必须与绘制几何一致——模型按所画内容读图，而非按意图。

## 生产集成

引擎在 8093 端口暴露 OpenAI 兼容 API（服务模型名 `glm-flash`）。参考部署中，它由 litellm 网关前置，使用既有的模型别名，客户端无需改动。已通过网关端到端验证：工具调用、推理分离为 `reasoning` 字段（可在网关归一化为 `reasoning_content`）、以及 `chat_template_kwargs`（`enable_thinking`、`reasoning_effort`、`clear_thinking`）。

## 致谢

- [hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized) — 基础 fork
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3 格式与内核；1.4.8 的 MGEMM 调度工作是预填充提升的来源
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — 自适应验证长度设计（补丁 006 为移植；该仓库为 AGPL-3.0）
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — SM80 backport 生态与 MTP 修复参考
- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) — 启发了本工作的 8 路 NVFP4 部署
- dkpoulsen 的 170HX 实验笔记 — 加载序列化与崩溃恢复实践

## 许可证

Apache-2.0（vLLM 衍生），有一处例外：`patches/006-adaptive-k-dflash2.patch` 是 [AGPL-3.0 许可](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)作品的衍生，仍保持 AGPL-3.0——将其纳入产品前请审阅其条款。模型权重受其各自许可证约束：EXL3 包受 ShapleyMCG 约束，DFlash2 draft 受 CC BY-NC-ND 约束——商用前请审阅二者。
