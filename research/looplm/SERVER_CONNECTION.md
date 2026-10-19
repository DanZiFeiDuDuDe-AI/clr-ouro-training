# 当前 CLR 训练服务器与权重同步

当前 O3 rebase 产物位于 Linux 服务器 `ssh.zzai.scnet.cn:11810`，账号 `root`。
`CHEEMS_CONNECTION.md` 是旧 Windows 训练机的说明；本轮使用当前 Linux 服务器。

认证从 `.local/scnet_password.dpapi` 读取，使用当前 Windows 用户的 DPAPI 加密。
也可以通过当前进程的 `LOOPLM_SSH_PASSWORD` 环境变量提供认证。
连接校验已有 SSH known_hosts；密码不写入实验配置、下载清单和日志。

连接检查：

```powershell
python tools/remote_weights.py exec "pwd"
```

每次远程训练必须通过 `tools/train_and_pull.py` 执行，或在完成后立即调用
`tools/remote_weights.py pull <远程实验目录> <本地实验目录>`。
包装器在训练命令正常返回后关闭原 SSH 连接，重新连接并下载 checkpoint、配置、指标；
下载通过 SHA-256 校验后才原子落盘。网络中断会重试；不同版本权重必须使用不同目录。

本轮 `clr_o3_rebase2_vgen2_failure_localization/run_diagnostics.py` 逐项执行这个流程，
上一个训练的权重下载完成后才开始下一项。每个目录中的 `download_verification.json`
记录验证结果。该流程在本地启动器运行时生效；任意其他启动方式需显式调用相同包装器。
后续 Codex 训练任务也应遵守 `AGENTS.md` 中的同步约定。

禁止 allocator、router/memory reconnect、integrated O3 evaluation，保持 Candidate B 和 Rule A 不变。
